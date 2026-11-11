@echo off
chcp 65001 >nul
cd /d %~dp0gomoku
echo 每 2 小时从 VM 回传一次最新权重，并自动和传统引擎打 6 局基准赛（Ctrl+C 结束）
python sync_vm.py --loop 7200 --bench 6
pause
