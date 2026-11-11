@echo off
chcp 65001 >nul
cd /d %~dp0gomoku
python play_all.py
pause
