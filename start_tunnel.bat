@echo off
REM ─────────────────────────────────────────────────────────────
REM  Metis - expose the local teacher gateway (OmniRoute :20128)
REM  so Google Colab can reach it for `metis distill`.
REM
REM  Run this, wait ~10 seconds, then look for the line:
REM      Your quick Tunnel has been created! Visit it at
REM      https://SOMETHING.trycloudflare.com
REM  Copy that URL, ADD "/v1" to the end, and paste it as the
REM  TEACHER_URL Colab secret (left sidebar -> key icon).
REM
REM  --protocol http2 is REQUIRED on this machine: the default QUIC
REM  (UDP) tunnel registers but drops requests (HTTP 522) on this
REM  network. http2 uses TCP and works reliably.
REM  --url uses 127.0.0.1 (not localhost) because the gateway binds
REM  IPv4 only, and localhost resolves to ::1 here.
REM
REM  Close this window (or run stop_tunnel.bat) when done distilling.
REM  NOTE: this tunnel has NO password. Anyone who finds the URL can
REM  use your gateway, so close it when you are not distilling.
REM ─────────────────────────────────────────────────────────────
echo Starting tunnel to http://127.0.0.1:20128 (HTTP/2) ...
cloudflared tunnel --url http://127.0.0.1:20128 --protocol http2
