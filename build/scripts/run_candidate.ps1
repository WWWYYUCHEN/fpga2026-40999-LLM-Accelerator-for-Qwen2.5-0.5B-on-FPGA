# =============================================================================
# run_custom.ps1 -- run qwen_B*.exe with an arbitrary raw-template prompt.
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\run_custom.ps1 `
#       -Exe .\qwen_B.exe -Dir .\runB_mw_green -Max 2 -Prompt "The quick brown fox."
#   Same driver contract as run_raw.ps1 (segments.tsv copy, SIM_SD, in.txt).
# =============================================================================
param([Parameter(Mandatory=$true)][string]$Exe,
      [Parameter(Mandatory=$true)][string]$Dir,
      [Parameter(Mandatory=$true)][string]$Prompt,
      [int]$Max = 2,
      [ValidateSet('raw','chat')][string]$Template = 'raw',
      [int]$Stop = 0,
      [switch]$DumpLayers)
$ErrorActionPreference = 'Continue'
$W    = $PSScriptRoot
if (!(Test-Path $Exe)) { "missing $Exe"; exit 1 }
$Exe = [IO.Path]::GetFullPath((Resolve-Path $Exe).Path)
if ([IO.Path]::IsPathRooted($Dir)) { $Dir = [IO.Path]::GetFullPath($Dir) }
else { $Dir = [IO.Path]::GetFullPath((Join-Path $W $Dir)) }
if (Test-Path $Dir) { throw "Run directory already exists: $Dir" }
New-Item -ItemType Directory $Dir | Out-Null
Copy-Item "$W\verification\segments.tsv" "$Dir\segments.tsv" -Force
$txt = "/template $Template`n/max $Max`n$Prompt`n"
Set-Content "$Dir\in.txt" $txt -NoNewline -Encoding ascii
$env:SIM_SD = "$W\sd"
Remove-Item Env:RTL_LAYER_DIAGNOSTIC -ErrorAction SilentlyContinue
Remove-Item Env:RTL_DUMP_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:RTL_STOP_AFTER -ErrorAction SilentlyContinue
if ($Stop -gt 0) { $env:RTL_STOP_AFTER = "$Stop" }
if ($DumpLayers) { $env:RTL_DUMP_LAYERS = '1' }
Push-Location $Dir
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $Exe -WorkingDirectory $Dir -RedirectStandardInput "$Dir\in.txt" -RedirectStandardOutput "$Dir\run.out" -RedirectStandardError "$Dir\run.err" -PassThru -NoNewWindow
$p.WaitForExit()
$exit = $p.ExitCode
$sw.Stop()
Pop-Location
"[$(Split-Path $Exe -Leaf) max=$Max stop=$Stop] exit=$($exit) wall={0:N1}s" -f $sw.Elapsed.TotalSeconds | Tee-Object "$Dir\timing.txt"
if (Test-Path "$Dir\hardware_tokens.csv") { "tokens:"; Get-Content "$Dir\hardware_tokens.csv" }
Select-String -Path "$Dir\run.out" -Pattern '@BEGIN|@PROMPT|TEMPLATE=' | ForEach-Object { $_.Line }
