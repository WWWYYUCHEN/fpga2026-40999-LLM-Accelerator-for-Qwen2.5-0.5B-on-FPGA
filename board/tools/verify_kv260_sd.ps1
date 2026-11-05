param(
    [string]$SdRoot = "",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$ReportRoot = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path "board\measurements\sd_verification")
)

$ErrorActionPreference = "Stop"
$expectedRoot = Join-Path $ProjectRoot "board\deployment"
if (-not (Test-Path -LiteralPath $expectedRoot)) {
    throw "Expected deployment directory not found: $expectedRoot"
}

if ([string]::IsNullOrWhiteSpace($SdRoot)) {
    if (Test-Path -LiteralPath "F:\QWUART.ELF") {
        $SdRoot = "F:\"
    }
    else {
        $candidate = Get-PSDrive -PSProvider FileSystem |
            Where-Object {
                $_.Root -ne "C:\" -and
                (Test-Path -LiteralPath (Join-Path $_.Root "QWUART.ELF")) -and
                (Test-Path -LiteralPath (Join-Path $_.Root "QWEN25"))
            } |
            Select-Object -First 1
        if ($null -eq $candidate) {
            throw "KV260 SD card not found. Expected QWUART.ELF and QWEN25 on a removable drive."
        }
        $SdRoot = $candidate.Root
    }
}

$SdRoot = [System.IO.Path]::GetFullPath($SdRoot)
$expectedFiles = Get-ChildItem -LiteralPath $expectedRoot -Recurse -File
$missing = @()
$mismatch = @()
$matched = @()

foreach ($expected in $expectedFiles) {
    $relative = $expected.FullName.Substring($expectedRoot.Length).TrimStart("\")
    $target = Join-Path $SdRoot $relative
    if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
        $missing += $relative
        continue
    }

    $targetItem = Get-Item -LiteralPath $target
    $expectedHash = (Get-FileHash -LiteralPath $expected.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    $targetHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant()
    $record = [pscustomobject]@{
        RelativePath = $relative
        ExpectedBytes = $expected.Length
        ActualBytes = $targetItem.Length
        ExpectedSHA256 = $expectedHash
        ActualSHA256 = $targetHash
    }
    if ($expected.Length -ne $targetItem.Length -or $expectedHash -ne $targetHash) {
        $mismatch += $record
    }
    else {
        $matched += $record
    }
}

$unexpected = @()
if (Test-Path -LiteralPath $SdRoot) {
    $actualRelative = Get-ChildItem -LiteralPath $SdRoot -Recurse -File |
        Where-Object {
            $_.FullName -notmatch "\\System Volume Information\\" -and
            $_.FullName -notmatch "\\RECYCLER\\"
        } |
        ForEach-Object { $_.FullName.Substring($SdRoot.Length).TrimStart("\") }
    $expectedRelative = $expectedFiles |
        ForEach-Object { $_.FullName.Substring($expectedRoot.Length).TrimStart("\") }
    $unexpected = @($actualRelative | Where-Object { $_ -notin $expectedRelative })
}

$oldFirmware = Test-Path -LiteralPath (Join-Path $SdRoot "QWENSD.ELF")
$allowedExtras = @("CHAT.CSV")
$unexpectedRequired = @($unexpected | Where-Object { $_ -notin $allowedExtras })
$passed = $missing.Count -eq 0 -and $mismatch.Count -eq 0 -and -not $oldFirmware -and $unexpectedRequired.Count -eq 0
$result = [pscustomobject]@{
    Status = if ($passed) { "PASS" } else { "FAIL" }
    SdRoot = $SdRoot
    ExpectedFileCount = $expectedFiles.Count
    MatchedFileCount = $matched.Count
    MissingFiles = $missing
    HashOrSizeMismatches = $mismatch
    UnexpectedFiles = $unexpected
    UnexpectedRequiredFiles = $unexpectedRequired
    OldQwenSdElfPresent = $oldFirmware
    CheckedAt = (Get-Date).ToString("o")
}

New-Item -ItemType Directory -Force -Path $ReportRoot | Out-Null
$jsonPath = Join-Path $ReportRoot "kv260_sd_verification.json"
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

Write-Host ""
Write-Host "KV260 SD verification"
Write-Host "SD root: $SdRoot"
Write-Host "Expected files: $($expectedFiles.Count)"
Write-Host "Matched files: $($matched.Count)"
Write-Host "Missing files: $($missing.Count)"
Write-Host "Hash or size mismatches: $($mismatch.Count)"
Write-Host "Old QWENSD.ELF present: $oldFirmware"
Write-Host "Unexpected extra files: $($unexpected.Count)"
Write-Host "Unexpected required files: $($unexpectedRequired.Count)"
Write-Host "Result: $($result.Status)"
Write-Host "Report: $jsonPath"
Write-Host ""

if ($missing.Count -gt 0) {
    Write-Host "Missing:"
    $missing | ForEach-Object { Write-Host "  $_" }
}
if ($mismatch.Count -gt 0) {
    Write-Host "Mismatched:"
    $mismatch | ForEach-Object { Write-Host "  $($_.RelativePath)" }
}
if ($unexpected.Count -gt 0) {
    Write-Host "Unexpected:"
    $unexpected | ForEach-Object { Write-Host "  $_" }
}

if (-not $passed) {
    exit 1
}
