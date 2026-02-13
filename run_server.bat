@echo off
title Flask Server - Network Mode

echo ================================
echo Starting Flask Server...
echo ================================

REM مسار Python الصحيح
set PYTHON_PATH=C:\Users\ahmad\AppData\Local\Programs\Python\Python311\python.exe

REM فتح المنفذ 5000 في الجدار الناري (إذا لم يكن موجود)
echo Opening firewall port 5000...
netsh advfirewall firewall add rule name="Flask Server Port 5000" dir=in action=allow protocol=TCP localport=5000 >nul 2>&1

REM عرض IP الجهاز
echo.
echo Your Network IP addresses:
ipconfig | findstr /i "IPv4"
echo.
echo Use this address on phone browser:
echo http://YOUR_IP:5000
echo.

REM فتح المتصفح محلياً
start http://127.0.0.1:5000

REM تشغيل السيرفر
echo Running server...
%PYTHON_PATH% app.py

pause