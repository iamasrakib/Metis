@echo off
REM Stops any running cloudflared tunnel (the public URL goes dead
REM immediately, which also closes the gateway to the internet).
taskkill /IM cloudflared.exe /F 2>NUL
echo Tunnel stopped - gateway is local-only again.
