# 打包成 exe（Windows）
# 用法：
#   .\打包exe.ps1            同时打两个版本
#   .\打包exe.ps1 -Only light  只打轻量版（传统引擎，约 15 MB，秒开）
#   .\打包exe.ps1 -Only full   只打完整版（含强化学习，体积大、启动慢）
param(
    [ValidateSet("all", "light", "full")]
    [string]$Only = "all"
)

$ErrorActionPreference = "Stop"
$py = "python"

Write-Host "=== 1. 检查 PyInstaller ===" -ForegroundColor Cyan
& $py -m PyInstaller --version 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "未安装，正在安装（需要网络）…" -ForegroundColor Yellow
    & $py -m pip install pyinstaller
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller 安装失败，请检查网络或代理" }
}

Push-Location $PSScriptRoot\gomoku
try {
    if ($Only -in @("all", "light")) {
        Write-Host "`n=== 2. 打包【轻量版】：只有传统引擎，不需要 torch ===" -ForegroundColor Cyan
        & $py -m PyInstaller --noconfirm --clean --onedir --name GomokuLight `
            --add-data "..\gomoku.html;." `
            launcher_ab.py
        if ($LASTEXITCODE -ne 0) { throw "轻量版打包失败" }
        Write-Host "完成：dist\GomokuLight\GomokuLight.exe" -ForegroundColor Green
    }

    if ($Only -in @("all", "full")) {
        Write-Host "`n=== 3. 打包【完整版】：含强化学习（torch 约 500 MB，耗时几分钟）===" -ForegroundColor Cyan
        & $py -m PyInstaller --noconfirm --clean --onedir --name Gomoku `
            --add-data "..\gomoku.html;." `
            --add-data "checkpoints;checkpoints" `
            --collect-all torch `
            --collect-submodules numpy `
            --hidden-import numpy `
            launcher.py
        if ($LASTEXITCODE -ne 0) { throw "完整版打包失败" }
        Write-Host "完成：dist\Gomoku\Gomoku.exe" -ForegroundColor Green
    }
}
finally {
    Pop-Location
}

Write-Host "`n=== 4. 产物 ===" -ForegroundColor Cyan
if (Test-Path "$PSScriptRoot\gomoku\dist") {
    Get-ChildItem "$PSScriptRoot\gomoku\dist" -Directory | ForEach-Object {
        $exe = Get-ChildItem $_.FullName -Filter *.exe | Select-Object -First 1
        $mb = [math]::Round((Get-ChildItem $_.FullName -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
        if ($exe) { "{0,-14} {1,6} MB   {2}" -f $_.Name, $mb, $exe.FullName }
    }
}
Write-Host "`n把整个文件夹（不是单个 exe）复制给别人，双击 exe 即可开玩。" -ForegroundColor Yellow
