# =============================================================================
# build_sim.ps1 -- host build of the candidate9 firmware against virtual
#                  UART / SD / MMIO stubs (no Verilator, no RTL)
#
# Mirrors the reference project's build.py native target: the same fw/*.c is
# compiled for the host with QWEN_HOST_SIM so main.c takes its simulation
# branches, and fw/tests/board_sim.c supplies the peripheral stubs plus the
# scripted-token accelerator. PL token output is fixture data, NOT model math.
#
# This is the counterpart to build_a53.ps1: same sources, different target.
# Use it to exercise the firmware loop without a board:
#   $env:SIM_SD   = <dir containing QWEN25\TOKEN.BIN>
#   $env:SIM_PLAN = <file, one line of token ids per request>
#   "hi" | .\build_sim\qwen_board_sim.exe
# =============================================================================
param(
    [string]$Gcc = $env:QWEN_HOST_CC,
    [string]$Out = ''
)
$ErrorActionPreference = 'Continue'
$W = $PSScriptRoot
if (-not $Out) { $Out = Join-Path $W 'build_sim' }
if (-not $Gcc) {
    $cmd = Get-Command 'gcc' -ErrorAction SilentlyContinue
    if (-not $cmd) { throw 'gcc not found: set $env:QWEN_HOST_CC' }
    $Gcc = $cmd.Source
}
$Gcc = (Resolve-Path -LiteralPath $Gcc).Path

$sources = @('main.c', 'tokenizer.c', 'utf8_stream.c', 'generation.c',
             'vendor\utf8proc\utf8proc.c', 'tests\board_sim.c') |
           ForEach-Object { Join-Path $W $_ }
foreach ($s in $sources) { if (-not (Test-Path -LiteralPath $s)) { throw "missing source: $s" } }

New-Item -ItemType Directory -Path $Out -Force | Out-Null
$exe = Join-Path $Out 'qwen_board_sim.exe'

$cflags = @('-std=c11', '-O2', '-Wall', '-Wextra', '-Werror', '-DUTF8PROC_STATIC',
            '-DQWEN_HOST_SIM', '-I', (Join-Path $W 'tests\shim'), '-I', $W)
& $Gcc @cflags @sources -o $exe '-static-libgcc'
if ($LASTEXITCODE -ne 0) { throw 'host build failed' }

"EXE   : $exe"
"BYTES : $((Get-Item -LiteralPath $exe).Length)"
