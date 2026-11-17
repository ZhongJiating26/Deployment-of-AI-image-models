param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '.venv\Scripts\python.exe')
)

$ErrorActionPreference = 'Stop'
$PythonPath = (Resolve-Path -LiteralPath $PythonPath).Path
Push-Location -LiteralPath $PSScriptRoot
try {
    & $PythonPath -m app.serve
    $result = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $result
