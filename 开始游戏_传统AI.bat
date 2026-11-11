@echo off
chcp 65001 >nul
cd /d %~dp0gomoku
python play_ab.py --time 1.5
pause
