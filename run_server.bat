@echo off
title Human Tracking Server - Online Mode
color 0B

echo ========================================================
echo       Human Tracking Project - Server Launcher
echo ========================================================
echo.

REM Set Python Path
set PYTHON_PATH=C:\Users\ahmad\AppData\Local\Programs\Python\Python311\python.exe

REM Firewall setup (optional, but kept for local testing)
netsh advfirewall firewall add rule name="Flask Server Port 5000" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1

echo [STATUS] Starting server with Cloudflare Tunnel...
echo [INFO] Once the server starts, look for the 'TryCloudflare' link.
echo [INFO] Share that link to allow anyone to access your site!
echo.
echo ========================================================
echo.

REM Start local browser for testing
start http://127.0.0.1:5000

REM Run the server
%PYTHON_PATH% -m uvicorn app_fastapi:app --host 0.0.0.0 --port 5000 --reload

pause