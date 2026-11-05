# =============================================================================
# INT8-attention E2E run (same driver as work_gemmwire\run.ps1)
#   -Tag A|B        : which build (qwen_<Tag>.exe)
#   -Mode gen|diag  : gen = full generation with RTL_STOP_AFTER=<Stop>
#                     diag = RTL_LAYER_DIAGNOSTIC=1 (exits at first layer-0 done)
#   Run directory must contain segments.tsv + in.txt (the bridge opens them
#   relative to the CWD); SIM_SD points at the model assets.
# =============================================================================
param([Parameter(Mandatory=$true)][string]$Tag,
      [Parameter(Mandatory=$true)][ValidateSet('gen','diag')][string]$Mode,
      [int]$Stop = 1)
$ErrorActionPreference = 'Continue'
$env:USERPROFILE = 'D:\ds\kv260_mmu\home'; $env:HOME = 'D:\ds\kv260_mmu\home'
$env:HOMEDRIVE = 'D:'; $env:HOMEPATH = '\ds\kv260_mmu\home'
$env:APPDATA = 'D:\ds\kv260_mmu\home\AppData\Roaming'; $env:LOCALAPPDATA = 'D:\ds\kv260_mmu\home\AppData\Roaming'
$W    = 'D:\ds\kv260_mmu\newdma\work_i8attn'
$dir  = "$W\run${Tag}_${Mode}"
$exe  = "$W\qwen_$Tag.exe"
if (!(Test-Path $exe)) { "missing $exe"; exit 1 }
New-Item -ItemType Directory -Force $dir | Out-Null
Copy-Item 'D:\ds\kv260_mmu\newdma\e2e_run\segments.tsv' "$dir\segments.tsv" -Force
if ($Mode -eq 'gen') { Set-Content "$dir\in.txt" "/max 1`nhi`n" -NoNewline -Encoding ascii }
else                 { Set-Content "$dir\in.txt" "hi`n" -NoNewline -Encoding ascii }
$env:SIM_SD = 'D:\ds\kv260_mmu\sd'
Remove-Item Env:RTL_LAYER_DIAGNOSTIC -ErrorAction SilentlyContinue
Remove-Item Env:RTL_DUMP_LAYERS -ErrorAction SilentlyContinue
Remove-Item Env:RTL_STOP_AFTER -ErrorAction SilentlyContinue
if ($Mode -eq 'diag') { $env:RTL_LAYER_DIAGNOSTIC = '1' } else { $env:RTL_STOP_AFTER = "$Stop" }
Push-Location $dir
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$p = Start-Process -FilePath $exe -WorkingDirectory $dir -RedirectStandardInput "$dir\in.txt" -RedirectStandardOutput "$dir\run.out" -RedirectStandardError "$dir\run.err" -PassThru -NoNewWindow
$p.WaitForExit()
$exit = $p.ExitCode
$sw.Stop()
Pop-Location
"[$Tag/$Mode] exit=$($exit) wall={0:N1}s ({1:N2} min)" -f $sw.Elapsed.TotalSeconds, $sw.Elapsed.TotalMinutes | Tee-Object "$dir\timing.txt"
$logi = "$dir\hardware_logits_0.bin"
if (Test-Path $logi) { "logits_0 sha256: " + (Get-FileHash $logi -Algorithm SHA256).Hash }
if (Test-Path "$dir\hardware_tokens.csv") { "tokens:"; Get-Content "$dir\hardware_tokens.csv" }
