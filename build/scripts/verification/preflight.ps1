# =============================================================================
# preflight.ps1 -- STEP 0 of the stage-5 sign-off checklist (REPORT_STAGE5_SLICE_A.md §5)
#
# Prints PASS/FAIL for every hard precondition of the final A/B.  Nothing here
# changes the tree; it only refuses to let a run start from an unverified state.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\preflight.ps1            # A-only checks
#   powershell -NoProfile -ExecutionPolicy Bypass -File .\preflight.ps1 -WithB     # + B build/source checks
# =============================================================================
param([switch]$WithB)
$ErrorActionPreference = 'Continue'
$W = 'D:\ds\kv260_mmu\newdma\work_i8attn'
Set-Location $W
$fail = 0
function Check([string]$name, [bool]$ok, [string]$detail) {
    if ($ok) { "PASS  $name  $detail" } else { "FAIL  $name  $detail"; $script:fail++ }
}

# ---- 1. the three harness files, against the values the A build recorded ----
$expect = @{
    'rtl_bridge.cpp'                                  = '22247CF142A218B4CD239B4B64978D6F116A5561F26DBBDCC507C4494BA948E8'
    'sim_dsp48e2_fast.v'                              = '8199E39CC48158CDD320187B7B6AAB8BD4B5F252A28EEB24CD48610D0E56CA3E'
    'D:\ds\kv260_mmu\models\tb_qwen25_chat_generation.sv' = '93C45AA0DD59CCF2B342129077F372863F69BD2A9273849267FDF08DD659B484'
}
foreach ($k in $expect.Keys) {
    $got = if (Test-Path $k) { (Get-FileHash $k -Algorithm SHA256).Hash } else { 'MISSING' }
    Check "harness $([IO.Path]::GetFileName($k))" ($got -eq $expect[$k]) "sha=$got"
}

# ---- 2. rtl_base must still be the golden tree (hard rule 1's tree half) ----
python .\check_tree.py rtl_base D:\ds\kv260_mmu\rtl 2>&1 | ForEach-Object { "      $_" }
Check 'rtl_base == D:\ds\kv260_mmu\rtl' ($LASTEXITCODE -eq 0) "check_tree exit=$LASTEXITCODE"

# ---- 3. the A artefact (hard rule 1) ----
$logits = 'runA_gen\hardware_logits_0.bin'
if (Test-Path $logits) {
    $sha = (Get-FileHash $logits -Algorithm SHA256).Hash
    Check 'A logits sha256' ($sha -eq 'A3AFC50CABAD1069826CB4F4EDE3A1B1EA0E455B794B07870A586C3420631103') "sha=$sha"
} else { Check 'A logits present' $false 'runA_gen\hardware_logits_0.bin missing' }
if (Test-Path 'runA_gen\hardware_tokens.csv') {
    $row = (Get-Content 'runA_gen\hardware_tokens.csv' | Select-Object -Last 1)
    Check 'A token/cycles' ($row -eq '0,9707,134082442') "row=$row"
} else { Check 'A tokens present' $false 'runA_gen\hardware_tokens.csv missing' }

# ---- 4. clip counter must be observable before the sign-off run ----
$clip = Select-String -Path 'rtl_stage\storage\qwen25_qkv_controller.v' -Pattern 'KV8_CLIP' -ErrorAction SilentlyContinue
Check 'KV8_CLIP display present in rtl_stage' ($null -ne $clip) 'needed for the clip-count acceptance column'

# ---- 5. source-list provenance for both builds (hard rule 2) ----
foreach ($tag in @('A')) {
    $p = "logs\srcs_$tag.txt"
    if (Test-Path $p) {
        $has1 = Select-String -Path $p -Pattern '93c45aa0dd59ccf2b342129077f372863f69bd2a9273849267fdf08dd659b484'
        $has2 = Select-String -Path $p -Pattern '8199e39cc48158cdd320187b7b6aab8bd4b5f252a28eeb24cd48610d0e56ca3e'
        Check "srcs_$tag harness hashes" (($null -ne $has1) -and ($null -ne $has2)) $p
    } else { Check "srcs_$tag present" $false $p }
}

if ($WithB) {
    # ---- 6. B build must be newer than every source it compiled ----
    if (Test-Path 'qwen_B.exe') {
        $exe = (Get-Item 'qwen_B.exe').LastWriteTime
        $newest = Get-ChildItem 'rtl_stage' -Recurse -Include *.v,*.sv | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        Check 'qwen_B.exe newer than rtl_stage sources' ($exe -gt $newest.LastWriteTime) "exe=$exe newest=$($newest.Name) $($newest.LastWriteTime)"
        $p = 'logs\srcs_B.txt'
        if (Test-Path $p) {
            $has1 = Select-String -Path $p -Pattern '93c45aa0dd59ccf2b342129077f372863f69bd2a9273849267fdf08dd659b484'
            $has2 = Select-String -Path $p -Pattern '8199e39cc48158cdd320187b7b6aab8bd4b5f252a28eeb24cd48610d0e56ca3e'
            Check 'srcs_B harness hashes' (($null -ne $has1) -and ($null -ne $has2)) $p
        } else { Check 'srcs_B present' $false $p }
    } else { Check 'qwen_B.exe present' $false 'build B first' }
}

""
if ($fail -eq 0) { "PREFLIGHT OK -- the run may start" ; exit 0 }
else { "PREFLIGHT FAILED ($fail check(s)) -- do NOT start the run"; exit 1 }
