# =============================================================================
# build_a53.ps1 -- build the candidate9 PS firmware for KV260 (A53 bare metal)
#
# Mirrors the reference project's src/software/interactive/tools/build.py a53:
#   same include order, same -specs, same link group, same linker script edits,
#   same entry/load-region assertions. Output: build_a53\QWENSD.ELF
#
# The RTL simulation path (build_candidate9.ps1) compiles fw/*.c for the host
# with QWEN_HOST_SIM + QWEN_RTL_SIM. This script compiles the same firmware for
# the real PS, so it adds the two board-only pieces the host build stubs out:
#   sd_prepare.c             provides qwen_board_prepare()
#   sd_entry.S               provides ENTRY(sd_entry)
# and links the prebuilt uart_console_override.o, which replaces the weak
# newlib _write/_read hooks.
#
# Usage:
#   $env:QWEN_A53_CC = 'D:\arm\bin\aarch64-none-elf-gcc.exe'
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\build_a53.ps1
# =============================================================================
param(
    [string]$Gcc = $env:QWEN_A53_CC,
    [string]$Bsp = '',
    [string]$Out = ''
)
# 'Continue': the toolchain writes the benign "LOAD segment with RWX permissions"
# notice to stderr, and Stop would turn that into a terminating error. Every
# native call below is checked through $LASTEXITCODE instead.
$ErrorActionPreference = 'Continue'
$W   = $PSScriptRoot
$Top = Split-Path $W -Parent
if (-not $Bsp) { $Bsp = Join-Path $Top 'bsp' }
if (-not $Out) { $Out = Join-Path $W 'build_a53' }
if (-not $Gcc) {
    $cmd = Get-Command 'aarch64-none-elf-gcc' -ErrorAction SilentlyContinue
    if (-not $cmd) { throw 'aarch64-none-elf-gcc not found: set $env:QWEN_A53_CC' }
    $Gcc = $cmd.Source
}
$Gcc = (Resolve-Path -LiteralPath $Gcc).Path

# ---- preflight --------------------------------------------------------------
$linkerSrc = Join-Path $Bsp 'lscript.ld'
$spec      = Join-Path $Bsp 'Xilinx.spec'
$inc       = Join-Path $Bsp 'include'
$lib       = Join-Path $Bsp 'lib'
foreach ($p in @($linkerSrc, $spec, $inc, $lib)) {
    if (-not (Test-Path -LiteralPath $p)) { throw "missing BSP entry: $p" }
}
$override = Join-Path $W 'uart_console_override.o'
if (-not (Test-Path -LiteralPath $override)) { throw "missing $override" }

$sources = @('main.c', 'tokenizer.c', 'utf8_stream.c', 'generation.c',
             'vendor\utf8proc\utf8proc.c', 'sd_prepare.c', 'sd_entry.S') |
           ForEach-Object { Join-Path $W $_ }
foreach ($s in $sources) { if (-not (Test-Path -LiteralPath $s)) { throw "missing source: $s" } }

# ---- linker script: assert contract, then enlarge stack/heap ---------------
$linker = Get-Content -LiteralPath $linkerSrc -Raw -Encoding UTF8
if ($linker -notmatch 'ENTRY\(sd_entry\)')    { throw 'lscript.ld has no ENTRY(sd_entry)' }
if ($linker -notmatch 'ORIGIN = 0x00100000')  { throw 'lscript.ld does not load at 0x00100000' }
$linker = [regex]::Replace($linker, '_STACK_SIZE = .*?;', '_STACK_SIZE = 0x100000;', 1)
$linker = [regex]::Replace($linker, '_HEAP_SIZE = .*?;',  '_HEAP_SIZE = 0x1000000;', 1)

New-Item -ItemType Directory -Path $Out -Force | Out-Null
$lscript = Join-Path $Out 'lscript.ld'
[System.IO.File]::WriteAllText($lscript, $linker, (New-Object System.Text.UTF8Encoding($false)))

# ---- compile ----------------------------------------------------------------
$cflags = @('-DSDT', '-DUTF8PROC_STATIC', '-O2', '-g', '-std=c11',
            '-Wall', '-Wextra', '-Werror', '-isystem', $inc, "-specs=$spec")
$objects = @()
foreach ($s in $sources) {
    $obj = Join-Path $Out (([System.IO.Path]::GetFileNameWithoutExtension($s)) + '.o')
    & $Gcc @cflags -c $s -o $obj
    if ($LASTEXITCODE -ne 0) { throw "compile failed: $s" }
    $objects += $obj
}
# Must come after our objects and before the archives: it is an object file on
# purpose, so _write/_read resolve here and write.c.obj/read.c.obj stay unused.
$objects += $override

# ---- link -------------------------------------------------------------------
$elf = Join-Path $Out 'QWENSD.ELF'
$map = Join-Path $Out 'qwen_uart.map'
& $Gcc @objects -o $elf "-Wl,-T,$lscript" "-Wl,-Map,$map" -L $lib `
    '-Wl,--start-group' '-lxilstandalone' '-lxiltimer' '-lxilffs' '-lxil' '-lgcc' '-lc' '-Wl,--end-group'
if ($LASTEXITCODE -ne 0) { throw 'link failed' }

# ---- report -----------------------------------------------------------------
$prefix = $Gcc -replace 'gcc(\.exe)?$', ''
& ($prefix + 'size.exe') $elf
$hash = (Get-FileHash -LiteralPath $elf -Algorithm SHA256).Hash.ToLower()
"ELF   : $elf"
"SHA256: $hash"
"BYTES : $((Get-Item -LiteralPath $elf).Length)"
