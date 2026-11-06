@echo off
cd /d "%~dp0"
uv run kartlab
if errorlevel 1 pause
