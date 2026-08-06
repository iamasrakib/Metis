@echo off
REM ─────────────────────────────────────────────────────────────
REM  Metis - start the named tunnel (permanent URL)
REM  https://teacher.alhissn.com/v1
REM
REM  This tunnel survives restarts. Just double-click this bat
REM  whenever you want to distill in Colab. Close this window
REM  (or run stop_tunnel.bat) when done.
REM ─────────────────────────────────────────────────────────────
echo Starting named tunnel: teacher.alhissn.com -> localhost:20128 ...
cloudflared tunnel run metis-teacher
