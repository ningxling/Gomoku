# 一键推送脚本：双击或右键"使用 PowerShell 运行"
# 推送到 https://github.com/ningxling/Gomoku
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

Set-Location $PSScriptRoot

$git = "D:\Git\cmd\git.exe"
if (-not (Test-Path $git)) { $git = "git" }

Write-Host "=== 1) 检查网络 ===" -ForegroundColor Cyan
$ok = Test-NetConnection github.com -Port 443 -InformationLevel Quiet
if (-not $ok) {
    Write-Host "连不上 github.com（443）。请换网络或挂 VPN 后再试。" -ForegroundColor Red
    Write-Host "测试命令： Test-NetConnection github.com -Port 443 -InformationLevel Quiet"
    Read-Host "按回车退出"
    exit 1
}
Write-Host "网络正常" -ForegroundColor Green

Write-Host "`n=== 2) 当前提交 ===" -ForegroundColor Cyan
& $git log --oneline -3

Write-Host "`n=== 3) 推送 ===" -ForegroundColor Cyan
Write-Host "如需登录：用户名 ningxling，密码填 Personal Access Token（ghp_ 开头）"
& $git push -u origin main

Write-Host "`n=== 4) 结果 ===" -ForegroundColor Cyan
if ($LASTEXITCODE -eq 0) {
    Write-Host "推送成功 -> https://github.com/ningxling/Gomoku" -ForegroundColor Green
} else {
    Write-Host "推送失败，请看上面的报错。" -ForegroundColor Red
    Write-Host "常见原因：网络不通 / 令牌没勾 repo / 远端仓库已有内容（可试 --force）"
}
Read-Host "`n按回车关闭"
