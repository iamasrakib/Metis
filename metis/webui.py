"""
Μῆτις (Metis) — Web UI (Gradio)
==================================
Browser-based chat interface for the Metis language model.

Usage:
    metis ui
    python -m metis.webui --checkpoint-dir checkpoints
"""

import sys
import logging
import argparse

from metis import load_model_and_tokenizer, generate_text
from metis.config import setup_logging

logger = logging.getLogger("metis.webui")


def main():
    parser = argparse.ArgumentParser(description="Metis Web UI")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--device", type=str, default=None, help="Device override")
    parser.add_argument("--port", type=int, default=7860, help="Gradio port")
    parser.add_argument("--share", action="store_true", help="Create public link")
    args = parser.parse_args()

    # Load model
    print("Loading model...")
    try:
        model, tokenizer, config = load_model_and_tokenizer(args.checkpoint_dir, args.device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print(f"   Train one first: metis train --dataset data/input.txt")
        sys.exit(1)

    # Lazily import gradio
    try:
        import gradio as gr
    except ImportError:
        print("❌ gradio not installed. Run: pip install gradio")
        sys.exit(1)

    def chat_fn(message: str, history: list, temperature: float, max_tokens: int,
                top_k: int, top_p: float, rep_penalty: float) -> str:
        """Chat response function for Gradio."""
        # Build prompt from history
        prompt = ""
        for user_msg, bot_msg in history:
            prompt += f"User: {user_msg}\nMetis: {bot_msg}\n"
        prompt += f"User: {message}\nMetis:"

        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            device=config.device,
        )

        response = generated[len(prompt):].strip().split("\n")[0]
        return response

    # Build the Gradio interface
    with gr.Blocks(
        title="Μῆτις (Metis) — Chat",
        theme=gr.themes.Soft(),
        css="""
        .logo-text { font-size: 1.5em; font-weight: bold; }
        .subtitle { color: #666; }
        """
    ) as demo:
        gr.Markdown(
            f"""
            # 🏛️ Μῆτις (Metis) v3.0
            *A modern tiny language model — built from scratch*
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    chat_fn,
                    additional_inputs=[
                        gr.Slider(0.0, 2.0, value=0.8, label="Temperature", step=0.1),
                        gr.Slider(32, 1024, value=200, step=32, label="Max Tokens"),
                        gr.Slider(0, 100, value=40, step=1, label="Top-K"),
                        gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-P"),
                        gr.Slider(1.0, 3.0, value=1.1, step=0.1, label="Repetition Penalty"),
                    ],
                    title="Chat with Metis",
                    description="Ask anything — Metis will respond based on its training data.",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Model Info")
                info = gr.JSON({
                    "Parameters": config.n_params,
                    "Device": config.device,
                    "Tokenizer": getattr(tokenizer, "encoding_name", "char"),
                    "Max Seq Len": config.max_seq_len,
                    "Vocab Size": config.vocab_size,
                })

                gr.Markdown("### Commands")
                gr.Markdown("""
                In chat you can use:
                - `/clear` — Reset conversation
                - `/temp 0.5` — Set temperature
                """)

    print(f"  Starting Web UI on port {args.port}...")
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
