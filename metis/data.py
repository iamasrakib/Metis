"""
Μῆτις (Metis) — Advanced Data Pipeline
=========================================
Professional data pipeline with:
  • BPE tokenizer (tiktoken-based) with character-level fallback
  • Memory-mapped datasets for GB-scale corpora
  • Pre-tokenization caching
  • Dynamic batch sizing
  • Multi-file directory loading
  • Vocabulary statistics and validation
"""

import hashlib
import json
import logging
import os
import pickle
import re
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from .packing import STREAM, PackedDataset

logger = logging.getLogger("metis.data")


# ──────────────────────────────────────────────────────────────────────────────
# BPE Tokenizer
# ──────────────────────────────────────────────────────────────────────────────

class BPETokenizer:
    """BPE tokenizer with tiktoken backend, character-level fallback,
    and full compatibility with the original CharTokenizer API.

    Special tokens:
        <pad> — Padding token
        <unk> — Unknown / fallback
        <bos> — Beginning of sequence
        <eos> — End of sequence

    In character mode (and the plain :class:`CharTokenizer`) the special ids
    are the fixed 0..3. In BPE mode they are allocated **above** the native
    vocabulary by :meth:`_build_tiktoken_encoding` — tiktoken's lowest ids are
    real tokens (``'!'`` is 0 in cl100k_base), so ``<pad>`` etc. must never
    share an id with a natural-language token.

    Uses OpenAI's tiktoken for fast BPE encoding. Falls back to a
    character-level tokenizer if tiktoken is unavailable or if the
    user explicitly requests character mode.
    """

    # Char-mode baseline. ``_build_tiktoken_encoding`` remaps these to ids
    # above the native vocab when a real tiktoken encoding is loaded.
    SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize tokenizer.

        Args:
            encoding_name: tiktoken encoding name. Options:
                "cl100k_base" — GPT-4 / GPT-3.5 (100K vocab, default)
                "p50k_base"   — GPT-3 / Codex (50K vocab)
                "r50k_base"   — GPT-2 / Ada (50K vocab)
                "o200k_base"  — GPT-4o (200K vocab, multilingual)
                "char"        — Fallback character-level tokenizer
        """
        self.encoding_name = encoding_name
        self._is_char_mode = encoding_name == "char"
        self._tiktoken_encoding = None
        self._char_encoder = CharTokenizer() if self._is_char_mode else None
        self._special_tokens = dict(self.SPECIAL_TOKENS)

        # Try loading tiktoken
        if not self._is_char_mode:
            try:
                import tiktoken  # noqa: F401  (presence check)
                self._build_tiktoken_encoding(encoding_name)
                logger.info(
                    f"BPE tokenizer initialized: {encoding_name} "
                    f"(vocab_size={self.vocab_size}, "
                    f"special_ids={self._special_tokens})"
                )
            except (ImportError, AttributeError) as e:
                logger.warning(f"tiktoken not available ({e}), falling back to character tokenizer")
                self._is_char_mode = True
                self._char_encoder = CharTokenizer()

        if self._is_char_mode:
            self.vocab_size = len(self.SPECIAL_TOKENS)
            logger.info(
                "Character-level tokenizer initialized "
                "(vocab_size will be updated at fit())"
            )

    @property
    def is_bpe(self) -> bool:
        """True if using BPE, False if character-level fallback."""
        return not self._is_char_mode

    # ── Public API (matches CharTokenizer for drop-in replacement) ──────

    def fit(self, text: str) -> "BPETokenizer":
        """Build vocabulary from training text (only needed for char mode).

        In BPE mode, the vocabulary is already fixed by tiktoken — this
        is a no-op that returns self for API compatibility.
        """
        if self._is_char_mode and self._char_encoder is not None:
            self._char_encoder.fit(text)
            self.vocab_size = self._char_encoder.vocab_size
            logger.info(f"Char fallback fitted — vocab_size={self.vocab_size}")
        else:
            logger.debug(f"BPE tokenizer already has fixed vocab_size={self.vocab_size}")
        return self

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Encode a string into token IDs.

        Args:
            text: Input string.
            add_bos: Prepend <bos> token.
            add_eos: Append <eos> token.

        Returns:
            List of integer token IDs.
        """
        if self._is_char_mode and self._char_encoder is not None:
            ids = self._char_encoder.encode(text, add_bos=False, add_eos=False)
        elif self._tiktoken_encoding is not None:
            # ``disallowed_special=()``: special-token strings (<|eos|>, …)
            # appearing in arbitrary training text are treated as ordinary
            # tokens instead of raising. Specials are inserted programmatically
            # via add_bos/add_eos, never by string matching.
            ids = self._tiktoken_encoding.encode(
                text, allowed_special=set(), disallowed_special=()
            )
        else:
            # Pure fallback: ordinal encoding
            ids = [ord(c) + len(self.SPECIAL_TOKENS) for c in text]

        if add_bos:
            ids = [self._special_tokens["<bos>"]] + ids
        if add_eos:
            ids = ids + [self._special_tokens["<eos>"]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """Decode token IDs back to a string.

        Args:
            ids: List of integer token IDs.
            skip_special: If True, omit special tokens from output.

        Returns:
            Decoded string.
        """
        if skip_special:
            special_ids = set(self._special_tokens.values())
            ids = [i for i in ids if i not in special_ids]

        if not ids:
            return ""

        if self._is_char_mode and self._char_encoder is not None:
            return self._char_encoder.decode(ids, skip_special=False)
        elif self._tiktoken_encoding is not None:
            try:
                return self._tiktoken_encoding.decode(ids)
            except Exception:
                pass

        # Pure fallback: ordinal decode
        return "".join(
            chr(i - len(self.SPECIAL_TOKENS)) if i >= len(self.SPECIAL_TOKENS) else ""
            for i in ids
        )

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode multiple texts efficiently (batched BPE)."""
        if self._is_char_mode or self._tiktoken_encoding is None:
            return [self.encode(t) for t in texts]

        # Use tiktoken's batched encode (special literals → ordinary tokens,
        # matching :meth:`encode`).
        results = []
        for text in texts:
            results.append(self._tiktoken_encoding.encode(
                text, allowed_special=set(), disallowed_special=()
            ))
        return results

    def decode_batch(self, batch: list[list[int]]) -> list[str]:
        """Decode multiple ID lists efficiently."""
        if self._is_char_mode or self._tiktoken_encoding is None:
            return [self.decode(ids) for ids in batch]

        results = []
        for ids in batch:
            results.append(self._tiktoken_encoding.decode(ids))
        return results

    def count_tokens(self, text: str) -> int:
        """Efficiently count tokens without returning the full list."""
        if self._is_char_mode and self._char_encoder is not None:
            return len(self._char_encoder.encode(text))
        elif self._tiktoken_encoding is not None:
            return len(self._tiktoken_encoding.encode(
                text, allowed_special=set(), disallowed_special=()
            ))
        return len(text)

    def save(self, path: str) -> None:
        """Save tokenizer configuration to a JSON file."""
        data = {
            "type": "bpe" if self.is_bpe else "char",
            "encoding_name": self.encoding_name,
            "vocab_size": self.vocab_size,
            "version": "3.0",
        }
        if self._is_char_mode and self._char_encoder is not None:
            data["char_stoi"] = self._char_encoder.stoi
            data["char_itos"] = {str(k): v for k, v in self._char_encoder.itos.items()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: str) -> "BPETokenizer":
        """Load tokenizer from a JSON file.

        Supports:
          - New BPE format (version 3.0)
          - Legacy CharTokenizer JSON format (version 2.0)
          - Legacy pickle format

        Returns:
            self (for chaining).
        """
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        tokenizer_type = data.get("type", "char")
        version = data.get("version", "2.0")

        if tokenizer_type == "bpe" and version == "3.0":
            self.encoding_name = data["encoding_name"]
            self._is_char_mode = False
            self._load_tiktoken(self.encoding_name)
            # Special ids derive deterministically from the encoding, so the
            # vocab size is recomputed rather than trusting a possibly-stale
            # value saved by a pre-fix checkpoint.
            self.vocab_size = max(self._special_tokens.values()) + 1
        elif tokenizer_type == "char" and version == "3.0":
            # BPETokenizer saved in char mode
            self._is_char_mode = True
            self._char_encoder = CharTokenizer()
            if "char_stoi" in data:
                self._char_encoder.stoi = data["char_stoi"]
                self._char_encoder.itos = {int(k): v for k, v in data["char_itos"].items()}
                self._char_encoder.vocab_size = data["vocab_size"]
            self.vocab_size = data["vocab_size"]
            self.encoding_name = "char"
        elif version == "2.0":
            # Legacy CharTokenizer JSON format
            self._is_char_mode = True
            self._char_encoder = CharTokenizer()
            self._char_encoder.stoi = data.get("stoi", {})
            self._char_encoder.itos = {int(k): v for k, v in data.get("itos", {}).items()}
            self.vocab_size = data.get("vocab_size", len(self.SPECIAL_TOKENS))
            self.encoding_name = "char"
        else:
            raise ValueError(f"Unknown tokenizer format: type={tokenizer_type}, version={version}")

        logger.info(
            f"Tokenizer loaded from {path} — "
            f"type={'bpe' if self.is_bpe else 'char'}, "
            f"vocab_size={self.vocab_size}"
        )
        return self

    def _load_tiktoken(self, encoding_name: str) -> None:
        """Load tiktoken encoding by name (registers Metis special tokens)."""
        if encoding_name == "char":
            self._is_char_mode = True
            self._char_encoder = CharTokenizer()
            return
        try:
            self._build_tiktoken_encoding(encoding_name)
        except Exception:
            logger.warning(f"Failed to load tiktoken '{encoding_name}', trying cl100k_base")
            self._build_tiktoken_encoding("cl100k_base")
            self.encoding_name = "cl100k_base"

    def _build_tiktoken_encoding(self, encoding_name: str) -> None:
        """Load ``encoding_name`` with Metis special tokens ABOVE its native
        vocabulary, and update ``self._special_tokens`` / ``vocab_size``.

        tiktoken's mergeable ranks cover the whole native vocab (ids 0..N-1),
        so its lowest ids are *real* tokens — in cl100k_base id 0 is ``'!'``,
        id 1 is ``'"'``. Registering ``<pad>`` & co. at ids 0-3 (the original
        code) made ``<pad>`` and ``'!'`` the same id, corrupting decode,
        padding, and the bos/eos markers. Every special token is therefore
        placed at ids strictly above the highest id already in use (including
        any specials the base encoding owns, e.g. ``<|endoftext|>``).

        Called identically from ``__init__`` and ``_load_tiktoken`` so a fresh
        tokenizer and a checkpoint-loaded one always agree on the ids.
        """
        import tiktoken

        base = tiktoken.get_encoding(encoding_name)
        high = max(base._mergeable_ranks.values(), default=-1)
        if base._special_tokens:
            high = max(high, max(base._special_tokens.values()))
        first = high + 1
        special_map = {
            f"<|{name}|>": first + i
            for i, name in enumerate(("pad", "unk", "bos", "eos"))
        }
        self._tiktoken_encoding = tiktoken.Encoding(
            name=encoding_name,
            pat_str=base._pat_str,
            mergeable_ranks=base._mergeable_ranks,
            special_tokens={**base._special_tokens, **special_map},
        )
        self._special_tokens = {
            "<pad>": special_map["<|pad|>"],
            "<unk>": special_map["<|unk|>"],
            "<bos>": special_map["<|bos|>"],
            "<eos>": special_map["<|eos|>"],
        }
        # Cover the highest special id; native ids are all below ``first``.
        self.vocab_size = max(special_map.values()) + 1

    # ── Properties (compatible with CharTokenizer) ──────────────────────

    @property
    def pad_id(self) -> int:
        return self._special_tokens["<pad>"]

    @property
    def unk_id(self) -> int:
        return self._special_tokens["<unk>"]

    @property
    def bos_id(self) -> int:
        return self._special_tokens["<bos>"]

    @property
    def eos_id(self) -> int:
        return self._special_tokens["<eos>"]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(type={'bpe' if self.is_bpe else 'char'}, "
            f"vocab_size={self.vocab_size})"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Legacy CharTokenizer (kept for backward compatibility)
# ──────────────────────────────────────────────────────────────────────────────

class CharTokenizer:
    """Character-level tokenizer — maintained for backward compatibility.

    New code should use ``BPETokenizer`` instead.
    See ``BPETokenizer`` for full documentation.
    """

    SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}

    def __init__(self):
        self.stoi: dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.itos: dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.vocab_size: int = len(self.SPECIAL_TOKENS)

    def fit(self, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        for ch in chars:
            if ch not in self.stoi:
                self.stoi[ch] = self.vocab_size
                self.itos[self.vocab_size] = ch
                self.vocab_size += 1
        logger.info(
            f"CharTokenizer fitted — vocab_size={self.vocab_size} "
            f"({self.vocab_size - len(self.SPECIAL_TOKENS)} chars + "
            f"{len(self.SPECIAL_TOKENS)} special tokens)"
        )
        return self

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi.get(c, self.SPECIAL_TOKENS["<unk>"]) for c in text]
        if add_bos:
            ids = [self.SPECIAL_TOKENS["<bos>"]] + ids
        if add_eos:
            ids = ids + [self.SPECIAL_TOKENS["<eos>"]]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special else set()
        return "".join(
            self.itos.get(i, "<unk>") for i in ids if i not in special_ids
        )

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "version": "2.0",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Tokenizer saved to {path}")

    def load(self, path: str) -> "CharTokenizer":
        if path.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            self.stoi = data["stoi"]
            self.itos = {int(k): v for k, v in data["itos"].items()}
            self.vocab_size = data["vocab_size"]
        else:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.stoi = state["stoi"]
            self.itos = state["itos"]
            self.vocab_size = state["vocab_size"]
        logger.info(f"Tokenizer loaded from {path} — vocab_size={self.vocab_size}")
        return self

    @property
    def pad_id(self) -> int:
        return self.SPECIAL_TOKENS["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.SPECIAL_TOKENS["<unk>"]

    @property
    def bos_id(self) -> int:
        return self.SPECIAL_TOKENS["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.SPECIAL_TOKENS["<eos>"]

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"


# ──────────────────────────────────────────────────────────────────────────────
# Dataset — MMapDataset (memory-mapped, GB-scale)
# ──────────────────────────────────────────────────────────────────────────────

class MMapDataset(Dataset):
    """Memory-mapped dataset for tokenized language modeling data.

    Wraps a flat numpy memmap array of token IDs so datasets larger than
    system RAM can be trained on without loading everything into memory.

    Each sample is a contiguous (input, target) pair of length ``seq_len``
    where ``target[i] = input[i+1]``.
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int,
        memmap_mode: str = "r",
        dtype: np.dtype = np.uint16,
    ):
        """
        Args:
            data_path: Path to a .bin (memmap) or .npy file, or a raw tensor.
            seq_len: Sequence length for each training sample.
            memmap_mode: Memory map file access mode ('r', 'r+', 'c').
            dtype: NumPy dtype for the memory-mapped array.
        """
        self.seq_len = seq_len
        self.dtype = dtype
        self._data_tensor = None

        # Load data
        if isinstance(data_path, torch.Tensor):
            self._data = data_path.numpy()
            self._data_tensor = data_path
        elif isinstance(data_path, np.ndarray):
            self._data = data_path
        elif data_path.endswith(".npy"):
            self._data = np.load(data_path, mmap_mode=memmap_mode)
        elif data_path.endswith(".bin"):
            # Raw binary — metadata must be in a companion .json or passed externally
            self._data = np.memmap(data_path, dtype=dtype, mode=memmap_mode)
        else:
            raise ValueError(f"Unsupported data format: {data_path}")

        self._len = max(0, len(self._data) - seq_len)
        if self._len == 0:
            raise ValueError(
                f"Data length ({len(self._data)}) must be > seq_len ({seq_len}). "
                f"Provide more training data."
            )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self._data[idx: idx + self.seq_len].astype(np.int64))
        y = torch.from_numpy(self._data[idx + 1: idx + self.seq_len + 1].astype(np.int64))
        return x, y

    def to_tensor(self) -> torch.Tensor:
        """Load the full dataset into a tensor (defeats memory-mapping)."""
        if self._data_tensor is None:
            self._data_tensor = torch.from_numpy(np.array(self._data))
        return self._data_tensor


class TextDataset(Dataset):
    """In-memory fixed-length sequence dataset (original implementation).

    Useful for smaller datasets (<500 MB). For larger data, use MMapDataset.
    """

    def __init__(self, data: torch.Tensor, seq_len: int):
        if len(data) <= seq_len:
            raise ValueError(
                f"Data length ({len(data)}) must be greater than seq_len ({seq_len}). "
                f"Provide more training data."
            )
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


class StreamingTextDataset(IterableDataset):
    """Streaming dataset for arbitrarily large text corpora.

    Reads and tokenizes data in chunks — never holds the full corpus
    in memory. Useful for multi-GB datasets that don't fit in RAM.
    """

    def __init__(
        self,
        file_paths: list[str],
        tokenizer: BPETokenizer,
        seq_len: int,
        chunk_size: int = 100_000,
    ):
        """
        Args:
            file_paths: List of text file paths to stream from.
            tokenizer: Fitted tokenizer.
            seq_len: Sequence length.
            chunk_size: Characters to read per chunk.
        """
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.chunk_size = chunk_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        buffer: list[int] = []
        pending = ""   # char tail carried across chunk boundaries
        for file_path in self.file_paths:
            with open(file_path, encoding="utf-8") as f:
                while True:
                    chunk = f.read(self.chunk_size)
                    if not chunk:
                        break
                    text = pending + chunk
                    # Cut at the last whitespace so a BPE token never spans a
                    # chunk boundary (a mid-token split changes the tokenization
                    # vs. a monolithic encode). The tail after the cut is
                    # carried into the next chunk. Chunks with no whitespace are
                    # tokenized whole — pathological, never hit for prose.
                    cut = max(text.rfind(" "), text.rfind("\n"), text.rfind("\t"))
                    if cut < 0:
                        body, pending = text, ""
                    else:
                        body, pending = text[:cut + 1], text[cut + 1:]
                    buffer.extend(self.tokenizer.encode(body))

                    # Yield complete sequences from buffer
                    while len(buffer) >= self.seq_len + 1:
                        x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                        y = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                        buffer = buffer[self.seq_len:]
                        yield x, y
            # Flush the carried tail at end-of-file (tokenized whole).
            if pending:
                buffer.extend(self.tokenizer.encode(pending))
                pending = ""
                while len(buffer) >= self.seq_len + 1:
                    x = torch.tensor(buffer[:self.seq_len], dtype=torch.long)
                    y = torch.tensor(buffer[1:self.seq_len + 1], dtype=torch.long)
                    buffer = buffer[self.seq_len:]
                    yield x, y


# ──────────────────────────────────────────────────────────────────────────────
# Tokenization Cache
# ──────────────────────────────────────────────────────────────────────────────

def _content_fingerprint(text: str) -> str:
    """Cheap content fingerprint for cache invalidation.

    Hashes the length plus the head/tail of the corpus. Editing a data file
    always changes one of these, so a stale cache is never silently reused —
    while a multi-GB corpus is never read into a full hash.
    """
    h = hashlib.sha256()
    h.update(str(len(text)).encode("utf-8"))
    h.update(text[:65_536].encode("utf-8", "replace"))
    h.update(text[-65_536:].encode("utf-8", "replace"))
    return h.hexdigest()[:12]


def _tokenizer_cache_name(tokenizer) -> str:
    """Cache-name for a tokenizer: tiktoken encoding name, or ``"char"``.

    ``is_bpe`` / ``encoding_name`` only exist on :class:`BPETokenizer`, so the
    access is guarded — the plain :class:`CharTokenizer` (no such attributes)
    must map to the same ``"char"`` tag the caller uses elsewhere.
    """
    return tokenizer.encoding_name if getattr(tokenizer, "is_bpe", False) else "char"


def _cache_path(dataset_path: str, tokenizer_name: str, seq_len: int,
                content_hash: str) -> str:
    """Generate a deterministic cache file path for a tokenized dataset.

    The key covers the path, tokenizer, sequence length AND the corpus content
    fingerprint — the original key (path + tokenizer only) silently reused the
    stale cache whenever a file changed, and made a train split and its val
    split (same path, different text) collide on one cache file.
    """
    raw_hash = hashlib.sha256(
        f"{dataset_path}:{tokenizer_name}:{content_hash}".encode()
    ).hexdigest()[:12]
    base = os.path.splitext(os.path.basename(dataset_path))[0]
    cache_dir = os.path.join("cache", "tokenized")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{base}_{tokenizer_name}_{raw_hash}_sl{seq_len}_v2.npy")


_CHUNK_CHARS = 64_000_000  # ~16M tokens per chunk — keeps transient lists small


def _tokenize_large(tokenizer: BPETokenizer, text: str) -> np.ndarray:
    """Encode a large corpus in chunks, returning a compact numpy array.

    ``BPETokenizer.encode`` returns a Python list of one int per token
    (~28 bytes per int object on 64-bit CPython).  On a multi-hundred-million-
    token corpus that list alone is many GB and OOMs Colab's ~12 GB RAM
    (the 500M-token FineWeb-Edu corpus → ~14 GB).  Chunking keeps each
    transient list to ~16M tokens (~0.5 GB); chunk boundaries fall on
    newlines so BPE tokens never straddle an edge.
    """
    arrays: list[np.ndarray] = []
    max_id = 0
    n = len(text)
    start = 0
    while start < n:
        end = min(start + _CHUNK_CHARS, n)
        if end < n:  # not the final chunk — break on a newline for a clean edge
            nl = text.rfind("\n", start, end)
            if nl > start:
                end = nl + 1
        ids = tokenizer.encode(text[start:end])
        if ids:
            max_id = max(max_id, max(ids))
            arrays.append(np.array(ids, dtype=np.uint32))
        start = end
    dtype = np.uint32 if max_id > 65535 else np.uint16
    if not arrays:
        return np.array([], dtype=dtype)
    data = np.concatenate(arrays)
    return data if dtype == np.uint32 else data.astype(dtype)


def _file_fingerprint(path: str) -> str:
    """Content fingerprint for cache invalidation, without reading the file whole.

    The in-memory variant (:func:`_content_fingerprint`) hashes a loaded str;
    this hashes a file's size plus head/tail bytes so a multi-GB corpus is
    never read into RAM just to build a cache key.
    """
    size = os.path.getsize(path)
    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))
    with open(path, "rb") as f:
        h.update(f.read(65_536))
        f.seek(max(0, size - 65_536))
        h.update(f.read())
    return h.hexdigest()[:12]


def _tokenize_file_streaming(path: str, tokenizer: BPETokenizer) -> np.ndarray:
    """Encode a large corpus file in newline-aligned chunks, never loading it whole.

    Loading a multi-GB corpus as one ``str`` is itself the OOM: a 2 GB file
    becomes 4-8 GB in RAM (CPython pads a string to 2/4 bytes per char once any
    char exceeds ASCII, and web text is full of unicode), and train/val slicing
    makes copies of that. Reading the file in 64M-char windows keeps peak RAM to
    the compact token array (~2 GB per 500M tokens) plus one window; boundaries
    land on newlines so BPE tokens stay clean across edges.
    """
    arrays: list[np.ndarray] = []
    max_id = 0
    carry = ""
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            chunk = f.read(_CHUNK_CHARS)
            window = carry + chunk
            if not window:
                break
            eof = not chunk
            if not eof and "\n" not in window:
                # No newline in the window — hold it over for more data, but cap
                # carry so a pathological newline-free run stays bounded.
                carry = window
                if len(carry) >= 4 * _CHUNK_CHARS:
                    carry, piece = carry[_CHUNK_CHARS:], carry[:_CHUNK_CHARS]
                    ids = tokenizer.encode(piece)
                    if ids:
                        max_id = max(max_id, max(ids))
                        arrays.append(np.array(ids, dtype=np.uint32))
                continue
            cut = (window.rfind("\n") + 1) if not eof else len(window)
            piece, carry = window[:cut], window[cut:]
            ids = tokenizer.encode(piece)
            if ids:
                max_id = max(max_id, max(ids))
                arrays.append(np.array(ids, dtype=np.uint32))
    dtype = np.uint32 if max_id > 65535 else np.uint16
    if not arrays:
        return np.array([], dtype=dtype)
    data = np.concatenate(arrays)
    return data if dtype == np.uint32 else data.astype(dtype)


def tokenize_and_cache(
    text: str,
    tokenizer: BPETokenizer,
    seq_len: int,
    dataset_path: str,
    force_rebuild: bool = False,
) -> tuple[np.ndarray, str]:
    """Tokenize text and cache the result as a memory-mapped numpy array.

    If a valid cache exists, loads from cache instead of re-tokenizing.

    Returns:
        Tuple of (token_ids_array, cache_path).
    """
    tokenizer_name = _tokenizer_cache_name(tokenizer)
    cache_path = _cache_path(dataset_path, tokenizer_name, seq_len,
                             _content_fingerprint(text))

    if not force_rebuild and os.path.exists(cache_path):
        logger.info(f"Loading tokenized cache: {cache_path}")
        data = np.load(cache_path, mmap_mode="r")
        return data, cache_path

    logger.info(f"Tokenizing {len(text):,} chars...")
    # Chunked encoding: encode() on the full corpus materializes a Python list
    # of one int per token (~28 bytes each) — a multi-hundred-million-token
    # corpus becomes ~14 GB and OOMs Colab's ~12 GB RAM. _tokenize_large keeps
    # each transient list to a ~16M-token chunk and returns a compact array.
    data = _tokenize_large(tokenizer, text)
    np.save(cache_path, data)
    logger.info(
        f"Tokenized cache saved: {cache_path} "
        f"({len(data):,} tokens, dtype={data.dtype.name})"
    )
    return data, cache_path


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading Utilities
# ──────────────────────────────────────────────────────────────────────────────

def load_text(path: str) -> str:
    """Load and validate a text file from path or directory.

    If *path* is a directory, loads **all** ``.txt`` files within it,
    concatenated with double-newlines.

    Args:
        path: File path or directory path.

    Returns:
        Concatenated text content.
    """
    if os.path.isdir(path):
        txt_files = sorted(f for f in os.listdir(path) if f.endswith(".txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in directory: {path}")
        texts = []
        for fname in txt_files:
            fpath = os.path.join(path, fname)
            with open(fpath, encoding="utf-8") as f:
                text = f.read()
            texts.append(text)
            logger.info(f"Loaded file: {fname} ({len(text):,} chars)")
        combined = "\n\n".join(texts)
        logger.info(
            f"Loaded directory: {path} ({len(txt_files)} files, "
            f"{len(combined):,} total chars)"
        )
        return combined

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if len(text) < 100:
        logger.warning(f"Dataset is very small ({len(text)} chars). Model may not learn well.")
    logger.info(f"Loaded dataset: {len(text):,} characters from {path}")
    return text


def train_val_split(
    text: str,
    train_ratio: float = 0.9,
) -> tuple[str, str]:
    """Split text into training and validation sets.

    Args:
        text: Full corpus text.
        train_ratio: Fraction of data for training (default 0.9).

    Returns:
        Tuple of (train_text, val_text).
    """
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    logger.info(
        f"Data split — train: {len(train_text):,} chars, "
        f"val: {len(val_text):,} chars "
        f"({train_ratio:.0%}/{1 - train_ratio:.0%})"
    )
    return train_text, val_text


def split_text_into_documents(text: str, min_len: int = 1) -> list[str]:
    """Split raw text into documents on paragraph (blank-line) boundaries.

    Paragraphs are the natural unit of a corpus that is a plain concatenation
    of short pieces (chat logs, web snippets, news articles). Empty / whitespace
    paragraphs are dropped; if the text has no paragraph breaks it is returned
    whole.

    Args:
        text: Raw corpus text.
        min_len: Minimum non-whitespace length for a paragraph to count.

    Returns:
        List of document strings (leading/trailing whitespace trimmed).
    """
    parts = re.split(r"\n\s*\n", text)
    docs = [p.strip() for p in parts if len(p.strip()) >= min_len]
    return docs or [text]


def load_documents(path: str) -> list[str]:
    """Load a corpus as a list of documents.

    Directory paths yield one document per ``.txt`` file (each file is a
    complete document). A single file is split into paragraphs via
    :func:`split_text_into_documents`.

    This is the packing-native counterpart of :func:`load_text` — dynamic
    sequence packing needs document boundaries to segment attention; the flat
    :func:`load_text` pipeline discards them.
    """
    if os.path.isdir(path):
        txt_files = sorted(f for f in os.listdir(path) if f.endswith(".txt"))
        if not txt_files:
            raise FileNotFoundError(f"No .txt files found in directory: {path}")
        docs = []
        for fname in txt_files:
            fpath = os.path.join(path, fname)
            with open(fpath, encoding="utf-8") as f:
                docs.append(f.read())
            logger.info(f"Loaded document: {fname} ({len(docs[-1]):,} chars)")
        return docs

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    docs = split_text_into_documents(text)
    logger.info(
        f"Loaded {len(docs):,} documents from {path} "
        f"({sum(len(d) for d in docs):,} chars)"
    )
    return docs


def split_documents(
    documents: list[str], train_ratio: float = 0.9
) -> tuple[list[str], list[str]]:
    """Split a document list into train and validation sets (by count).

    ``n_train`` is clamped so the validation set never comes back empty when
    there are ≥ 2 documents (the old ``max(1, int(n * ratio))`` produced an
    empty val for every 1- and 2-document corpus, breaking the validation-loss
    stage). With a single document the val set is empty and a warning is logged
    so callers can decide whether that is acceptable.
    """
    n_total = len(documents)
    if n_total < 2:
        logger.warning(
            "split_documents: fewer than 2 documents — validation set is empty."
        )
        return list(documents), []
    n_train = max(1, min(n_total - 1, int(n_total * train_ratio)))
    train_docs = documents[:n_train]
    val_docs = documents[n_train:]
    logger.info(
        f"Document split — train: {len(train_docs):,} docs, "
        f"val: {len(val_docs):,} docs ({train_ratio:.0%}/{1 - train_ratio:.0%})"
    )
    return train_docs, val_docs


def create_packed_dataloader(
    text_or_documents: str | list[str],
    tokenizer: BPETokenizer | CharTokenizer,
    seq_len: int,
    batch_size: int,
    strategy: str = STREAM,
    shuffle: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """Create a DataLoader that yields dynamically packed batches.

    Accepts either a raw text string (split into documents) or an explicit list
    of document strings. Each document is tokenized with the supplied tokenizer
    and packed into fixed-length sequences (see ``metis/packing.py``); every
    yielded batch is a :class:`PackedBatch` carrying ``input_ids``/``labels``,
    a block-diagonal causal ``attention_mask``, per-segment RoPE
    ``position_ids``, and ``cu_seqlens``.

    The tokenizer interface is preserved: documents are encoded with
    ``tokenizer.encode`` and ``<eos>`` / ``<pad>`` come straight from the
    tokenizer's ``eos_id`` / ``pad_id`` properties.

    Args:
        text_or_documents: Corpus text or list of document strings.
        tokenizer: Fitted BPETokenizer (or CharTokenizer).
        seq_len: Fixed packed-sequence length.
        batch_size: Packed sequences per batch.
        strategy: ``"stream"`` (contiguous, zero padding) or ``"bin"``
            (whole-document first-fit-decreasing).
        shuffle: Re-compose batches each epoch.
        seed: RNG seed for reproducible packing / shuffling.

    Returns:
        DataLoader yielding :class:`PackedBatch` objects (``batch_size=None``;
        each item is already a full batch).
    """
    documents = (
        split_text_into_documents(text_or_documents)
        if isinstance(text_or_documents, str)
        else list(text_or_documents)
    )
    doc_ids = [tokenizer.encode(doc) for doc in documents]
    dataset = PackedDataset(
        doc_ids, seq_len, batch_size,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        strategy=strategy,
        shuffle=shuffle,
        seed=seed,
    )
    # ``pin_memory=True`` is intentionally omitted: with ``batch_size=None``
    # DataLoader returns the PackedBatch dataclass unchanged and never touches
    # its inner tensors, so pinning would be a silent no-op. The overlapped
    # pipeline pins each batch explicitly on the staging thread instead.
    return DataLoader(dataset, batch_size=None, num_workers=0, pin_memory=False)


def _make_loader(dataset, batch_size: int, shuffle: bool, drop_last: bool,
                 num_workers: int, rank: int = 0, world_size: int = 1) -> DataLoader:
    """Build a DataLoader over an indexable ``dataset``, sharded per DDP rank.

    With ``world_size > 1`` a :class:`DistributedSampler` gives each rank a
    disjoint slice of the corpus, so the effective global batch is
    ``batch_size * world_size`` of *unique* data (without it every rank would
    iterate the same full corpus and the extra GPUs would train on duplicated
    examples). ``rank`` must be passed by each DDP process.
    """
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=shuffle, drop_last=drop_last,
        )
        return DataLoader(
            dataset, batch_size=batch_size, sampler=sampler,
            drop_last=drop_last, pin_memory=True, num_workers=num_workers,
        )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, pin_memory=True, num_workers=num_workers,
    )


def create_dataloader(
    text: str,
    tokenizer: BPETokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
    use_mmap: bool = True,
    num_workers: int = 0,
    dataset_path: str = "",
    force_recache: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create a DataLoader from raw text with caching support.

    Args:
        text: Raw text to tokenize and batch.
        tokenizer: Fitted BPETokenizer (or CharTokenizer).
        seq_len: Sequence length for each training sample.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle data.
        use_mmap: Use memory-mapped dataset (saves RAM for large corpora).
        num_workers: DataLoader worker processes (>0 for faster loading).
        dataset_path: Original dataset path (for cache key). If empty,
                      caching is disabled.
        force_recache: Force re-tokenization even if cache exists.
        rank: DDP rank (0 in single-process training) for data sharding.
        world_size: Number of DDP processes (1 = no sharding).

    Returns:
        PyTorch DataLoader yielding (input, target) batches.
    """
    # Try loading from cache first
    if dataset_path and use_mmap:
        cache_path = _cache_path(
            dataset_path,
            _tokenizer_cache_name(tokenizer),
            seq_len,
            _content_fingerprint(text),
        )
        try:
            data_array, _ = tokenize_and_cache(
                text, tokenizer, seq_len,
                dataset_path=dataset_path,
                force_rebuild=force_recache,
            )
            dataset = MMapDataset(data_array, seq_len)
        except (OSError, ValueError, EOFError) as e:
            # Unusable cache (corrupt .npy, wrong dtype, mmap failure) is the
            # one recoverable condition: drop the bad cache file so the next
            # run rebuilds it fresh, and fall back to in-memory tokenization.
            # Anything outside these types is a real bug (e.g. in encode) and
            # must surface instead of being masked by a silent fallback.
            logger.warning(f"MMap dataset failed ({e}); falling back to in-memory")
            try:
                if os.path.exists(cache_path):
                    os.remove(cache_path)
                    logger.info(f"Removed unusable cache: {cache_path}")
            except OSError:
                pass
        else:
            return _make_loader(
                dataset, batch_size, shuffle, drop_last=True,
                num_workers=num_workers, rank=rank, world_size=world_size,
            )

    # Fallback: in-memory dataset (chunked encode — same RAM safety as the
    # mmap path, so --no-mmap on a large corpus can't OOM either).
    data = _tokenize_large(tokenizer, text)
    data_tensor = torch.tensor(data, dtype=torch.long)
    dataset = TextDataset(data_tensor, seq_len)
    return _make_loader(
        dataset, batch_size, shuffle, drop_last=True,
        num_workers=num_workers, rank=rank, world_size=world_size,
    )


def create_dataloaders_from_file(
    dataset_path: str,
    tokenizer: BPETokenizer,
    seq_len: int,
    batch_size: int,
    train_ratio: float = 0.9,
    shuffle: bool = True,
    num_workers: int = 0,
    force_recache: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders by stream-tokenizing a corpus file.

    The memory-safe counterpart of :func:`create_dataloader` for large
    single-file corpora: the file is encoded in newline-aligned chunks straight
    from disk (:func:`_tokenize_file_streaming`), so the full corpus is never
    held as a Python ``str``. The tokenized array is cached on disk keyed by a
    file fingerprint, so resume runs load it without re-encoding; the array is
    split at ``train_ratio`` for train/val (both are views — no copies).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    cache_path = _cache_path(
        dataset_path, _tokenizer_cache_name(tokenizer), seq_len,
        _file_fingerprint(dataset_path),
    )
    if not force_recache and os.path.exists(cache_path):
        logger.info(f"Loading tokenized cache: {cache_path}")
        data = np.load(cache_path, mmap_mode="r")
    else:
        logger.info(
            f"Tokenizing {os.path.getsize(dataset_path):,} bytes "
            f"(streamed — corpus never loaded whole)..."
        )
        data = _tokenize_file_streaming(dataset_path, tokenizer)
        np.save(cache_path, data)
        logger.info(
            f"Tokenized cache saved: {cache_path} "
            f"({len(data):,} tokens, dtype={data.dtype.name})"
        )
    split = int(len(data) * train_ratio)
    if split < seq_len + 1 or len(data) - split < seq_len + 1:
        raise ValueError(
            f"Corpus tokenizes to {len(data)} tokens, too small to split into "
            f"train/val sequences of length {seq_len}. Provide more data."
        )
    train_loader = _make_loader(
        MMapDataset(data[:split], seq_len), batch_size, shuffle,
        drop_last=True, num_workers=num_workers, rank=rank, world_size=world_size,
    )
    val_loader = _make_loader(
        MMapDataset(data[split:], seq_len), batch_size, shuffle=False,
        drop_last=True, num_workers=num_workers, rank=rank, world_size=world_size,
    )
    return train_loader, val_loader


# Backward compatibility aliases
get_dataloader = create_dataloader
