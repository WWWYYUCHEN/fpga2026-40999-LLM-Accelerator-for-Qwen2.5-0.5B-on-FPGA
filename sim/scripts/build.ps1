# =============================================================================
# INT8-attention E2E build (A/B), mirrors newdma\build_fast.ps1 (verilator --cc +
# zig per file) exactly like work_gemmwire\build.ps1 did.
#   A = rtl_base  : byte-identical copy of D:\ds\kv260_mmu\rtl   (baseline)
#   B = rtl_stage : same tree + INT8 KV cache (stages 1..3)
# The object directory is DELETED before every build on purpose: build_fast.ps1
# skips existing .o files and has produced bogus binaries from stale objects.
# =============================================================================
param([Parameter(Mandatory=$true)][string]$Tag,
      [Parameter(Mandatory=$true)][ValidateSet('base','stage','stage4')][string]$TreeSel,
      [ValidateSet('unisim','fast')][string]$Dsp = 'fast')
$ErrorActionPreference = 'Continue'
$env:USERPROFILE = 'D:\ds\kv260_mmu\home'; $env:HOME = 'D:\ds\kv260_mmu\home'
$env:HOMEDRIVE = 'D:'; $env:HOMEPATH = '\ds\kv260_mmu\home'
$env:APPDATA = 'D:\ds\kv260_mmu\home\AppData\Roaming'; $env:LOCALAPPDATA = 'D:\ds\kv260_mmu\home\AppData\Roaming'
$W      = 'D:\ds\kv260_mmu\newdma\work_i8attn'
$CAD    = 'C:\Users\ZKHC\tools\oss-cad-suite'
$ZIG    = 'C:\Users\ZKHC\tools\ziglang_verilator\ziglang\zig.exe'
$RT     = "$CAD\share\verilator\include"
$MODELS = 'D:\ds\kv260_mmu\models'
$FW     = 'D:\ds\kv260_mmu\fw'
$env:VERILATOR_ROOT = "$CAD\share\verilator"
$env:PATH = "$CAD\lib;$CAD\bin;$env:PATH"

$sw = [System.Diagnostics.Stopwatch]::StartNew()
# stage4 = the same shadow tree as stage, minus the BF16 array (drop list below).
$srcdir = if ($TreeSel -eq 'stage4') { "$W\rtl_stage" } else { "$W\rtl_$TreeSel" }
$obj  = "$W\obj_$Tag"
$logs = "$W\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
if (!(Test-Path $srcdir)) { "missing $srcdir"; exit 1 }

# ---- HARD REQUIREMENT: no stale objects, ever -------------------------------
if (Test-Path $obj) { Remove-Item -Recurse -Force $obj }
New-Item -ItemType Directory -Force $obj | Out-Null

# ---- source list ------------------------------------------------------------
switch ($TreeSel) {
    'base'   { $drop = @() }
    'stage'  { $drop = @() }
    'stage4' { $drop = @('qwen25_gemm.sv', 'qwen_gemm_array.sv', 'qwen_gemm_pe.sv', 'qwen_pe_weight_select.v',
                         'qwen_fp32_sum8.sv', 'qwen_int8_to_bf16.sv', 'qwen25_weight_dequant.v') }
}
$dspfile = if ($Dsp -eq 'fast') { "$W\sim_dsp48e2_fast.v" } else { "$MODELS\sim_dsp48e2_verilator.v" }
$srcs = @(Get-ChildItem $srcdir -Recurse -File -Include *.v,*.sv |
          Where-Object { $drop -notcontains $_.Name } |
          Sort-Object FullName | ForEach-Object { $_.FullName })
$srcs += @("$MODELS\qwen25_scalar_math_verilator.sv", "$MODELS\qwen25_sincos_verilator.sv",
           "$MODELS\qwen25_xpm_verilator.sv", $dspfile,
           "$MODELS\sim_unisims.v", "$MODELS\tb_qwen25_chat_generation.sv")

$prov = foreach ($s in $srcs) { (Get-FileHash $s -Algorithm SHA256).Hash.ToLower() + '  ' + $s }
$prov | Set-Content "$logs\srcs_$Tag.txt" -Encoding ascii
"[$Tag] tree=$TreeSel dsp=$Dsp sources: $($srcs.Count)  dropped: $($drop -join ',')" | Tee-Object "$logs\build_$Tag.log" -Append

# ---- 1) Verilator front end -------------------------------------------------
$vargs = @('--cc', '--timing', '--exe', '--top-module', 'tb_qwen25_inference', '--prefix', 'Vlayer',
           '--Mdir', $obj, '--output-split', '20000', '-Wno-fatal') + $srcs
Set-Content "$logs\verilate_cmd_$Tag.txt" -Value ($vargs -join "`n") -Encoding ascii
& "$CAD\bin\verilator_bin.exe" @vargs 2>&1 | Out-File "$logs\verilate_$Tag.log" -Encoding utf8
if ($LASTEXITCODE -ne 0) { "[$Tag] VERILATE FAIL"; Get-Content "$logs\verilate_$Tag.log" | Select-Object -Last 40; exit 1 }
"[$Tag] verilate OK  {0:N0}s" -f $sw.Elapsed.TotalSeconds | Tee-Object "$logs\build_$Tag.log" -Append

# ---- 2) C++ (generated + rtl_bridge + runtime + dpi shim) -------------------
$flags = @('-std=c++20', '-O3', '-fno-fast-math', '-ffp-contract=off', '-DVL_TIME_CONTEXT', '-DVL_USER_FINISH',
           '-I', $obj, '-I', $RT, '-I', "$RT\vltstd")
$cpp = @(Get-ChildItem $obj -Filter '*.cpp' | ForEach-Object { $_.FullName })
$cpp += @("$W\rtl_bridge.cpp", "$MODELS\scalar_math_dpi.cpp")
$cpp += @("$RT\verilated.cpp", "$RT\verilated_dpi.cpp", "$RT\verilated_timing.cpp", "$RT\verilated_threads.cpp")

$jobs = @{}; $maxpar = 10; $queue = New-Object System.Collections.Queue
foreach ($p in $cpp) { $queue.Enqueue($p) }
$fail = $false
while ($queue.Count -gt 0 -or $jobs.Count -gt 0) {
    while ($jobs.Count -lt $maxpar -and $queue.Count -gt 0) {
        $p   = $queue.Dequeue()
        $o   = Join-Path $obj ((Split-Path $p -Leaf) -replace '\.cpp$', '.o')
        $err = Join-Path $obj ('cc_' + [IO.Path]::GetFileNameWithoutExtension($p) + '.txt')
        if (Test-Path $o) { Remove-Item $o -Force }
        $a = @('c++') + $flags + @('-c', $p, '-o', $o)
        $r = Start-Process -FilePath $ZIG -ArgumentList $a -PassThru -NoNewWindow -RedirectStandardError $err
        $jobs[$r.Id] = @{ p = $p; proc = $r; out = $o }
    }
    Start-Sleep -Milliseconds 300
    foreach ($k in @($jobs.Keys)) {
        if ($jobs[$k].proc.HasExited) {
            $o = $jobs[$k].out
            if (!(Test-Path $o) -or (Get-Item $o).Length -eq 0) {
                $fail = $true
                "[$Tag] CC FAIL $($jobs[$k].p)" | Tee-Object "$logs\build_$Tag.log" -Append
                Get-Content (Join-Path $obj ('cc_' + [IO.Path]::GetFileNameWithoutExtension($jobs[$k].p) + '.txt')) -ErrorAction SilentlyContinue | Select-Object -First 10
            }
            $jobs.Remove($k)
        }
    }
}
"[$Tag] cpp objects: $((Get-ChildItem $obj -Filter '*.o').Count)  {0:N0}s fail=$fail" -f $sw.Elapsed.TotalSeconds | Tee-Object "$logs\build_$Tag.log" -Append
if ($fail) { exit 1 }

# ---- 3) firmware C ---------------------------------------------------------
$cflags = @('-std=c11', '-O2', '-DUTF8PROC_STATIC', '-DQWEN_HOST_SIM', '-DQWEN_RTL_SIM',
            '-I', "$FW\tests\shim", '-I', $FW, '-I', 'D:\ds\kv260_mmu')
foreach ($s in @(@{f="$FW\main.c"; n='main'; d='-Dmain=qwen_firmware_main'},
                 @{f="$FW\tokenizer.c"; n='tokenizer'; d=$null},
                 @{f="$FW\utf8_stream.c"; n='utf8_stream'; d=$null},
                 @{f="$FW\generation.c"; n='generation'; d=$null},
                 @{f="$FW\vendor\utf8proc\utf8proc.c"; n='utf8proc'; d=$null},
                 @{f="$FW\tests\board_sim.c"; n='board_sim'; d=$null})) {
    $a = @('cc') + $cflags
    if ($s.d) { $a += $s.d }
    $a += @('-c', $s.f, '-o', (Join-Path $obj ('fw_' + $s.n + '.o')))
    & $ZIG @a 2> (Join-Path $obj 'cc_err.txt')
    if ($LASTEXITCODE -ne 0) { "[$Tag] FW FAIL $($s.n)"; Get-Content (Join-Path $obj 'cc_err.txt') | Select-Object -First 8; exit 1 }
}
"[$Tag] firmware OK" | Tee-Object "$logs\build_$Tag.log" -Append

# ---- 4) link ---------------------------------------------------------------
$objs = @(Get-ChildItem $obj -Filter '*.o' | ForEach-Object { $_.FullName })
$exe  = "$W\qwen_$Tag.exe"
& $ZIG c++ @objs -o $exe 2> "$logs\link_$Tag.err"
if ($LASTEXITCODE -ne 0) { "[$Tag] LINK FAIL"; Get-Content "$logs\link_$Tag.err" | Select-Object -Last 12; exit 1 }
"[$Tag] LINK OK -> $exe  total {0:N0}s" -f $sw.Elapsed.TotalSeconds | Tee-Object "$logs\build_$Tag.log" -Append
(Get-FileHash $exe -Algorithm SHA256).Hash | Tee-Object "$logs\build_$Tag.log" -Append
