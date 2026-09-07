$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$python = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Semantic ranker environment not found. Run .\setup.ps1 first."
}

& $python .\main.py
