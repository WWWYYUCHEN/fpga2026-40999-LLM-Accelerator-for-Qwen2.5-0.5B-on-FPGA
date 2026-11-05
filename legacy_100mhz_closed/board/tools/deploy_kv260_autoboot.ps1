param([ValidateSet('F')][string]$DriveLetter = 'F')

# Deploy only the two generated launcher artifacts to the identified SD card.
# All backups and evidence stay on the PC. No raw disk or QSPI operations.
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$board = Split-Path -Parent $PSScriptRoot
$result = Join-Path $board 'measurements\autoboot'
$baselinePath = Join-Path $result 'sd-deployment.json'
if (-not (Test-Path -LiteralPath $baselinePath)) { throw "Deployment baseline not found: $baselinePath" }
$baseline = Get-Content -Raw -LiteralPath $baselinePath | ConvertFrom-Json
$cardRoot = $DriveLetter + ':\'
$recordPath = Join-Path $result 'sd-deployment.json'
$backupRoot = Join-Path $result 'original-sd-launcher'
$expectedNew = @{
    'boot.cmd' = '1d6082144a8c659cdbb14f232b74455bc4a38e766bb037822a73835db3045cdc'
    'boot.scr' = '32c129004f62d2c00f7da829465beb3c9cf74d00d44b6b5841703c474d4881f5'
}
if (Test-Path -LiteralPath $recordPath) { throw 'Deployment evidence already exists; inspect it before rerunning.' }
if (Test-Path -LiteralPath $backupRoot) { throw 'Backup already exists; refusing to overwrite.' }

function Get-CardIdentity {
    $volume = Get-Volume -DriveLetter $DriveLetter
    $partition = Get-Partition -DriveLetter $DriveLetter
    $disk = Get-Disk -Number $partition.DiskNumber
    if ($volume.FileSystem -ne 'FAT32' -or $volume.DriveType -ne 'Removable' -or
        $disk.FriendlyName.Trim() -ne $baseline.disk_friendly_name -or
        $disk.SerialNumber.Trim() -ne $baseline.disk_serial_number -or
        [long]$disk.Size -ne [long]$baseline.disk_size_bytes -or
        $disk.IsBoot -or $disk.IsSystem -or $disk.IsReadOnly -or
        $partition.PartitionNumber -ne 1 -or $partition.Offset -ne 32256) {
        throw 'SD identity/filesystem/safety mismatch; no further writes.'
    }
    [ordered]@{drive=$cardRoot;disk_number=$disk.Number;friendly_name=$disk.FriendlyName.Trim();serial_number=$disk.SerialNumber.Trim();bytes=[long]$disk.Size;filesystem=$volume.FileSystem;partition=$partition.PartitionNumber;offset=$partition.Offset}
}

function Get-Snapshot {
    $paths = @($baseline.files | ForEach-Object { $_.path }) + @('QWUART.ELF','CHAT.CSV')
    foreach ($relative in $paths) {
        $absolute = [IO.Path]::GetFullPath((Join-Path $cardRoot $relative))
        if (-not $absolute.StartsWith($cardRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Path escaped card root.' }
        $item = Get-Item -LiteralPath $absolute
        if ($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw 'Unexpected directory/reparse point.' }
        [pscustomobject]@{path=$relative;bytes=[long]$item.Length;sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath $absolute).Hash.ToLowerInvariant();archive=[bool]($item.Attributes -band [IO.FileAttributes]::Archive)}
    }
}

$record = [ordered]@{started_at=[DateTimeOffset]::Now.ToString('o');status='started';target=$null;before=@();backups=@();written=@();after=@();error=$null;board_cold_boot_verified=$false;formatted=$false;partition_changed=$false;qspi_written=$false;model_or_firmware_modified=$false;sd_added_files=@()}
try {
    $record.target = Get-CardIdentity
    foreach ($name in @('boot.cmd','boot.scr')) {
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $result $name)).Hash.ToLowerInvariant() -ne $expectedNew[$name]) { throw 'Generated launcher mismatch.' }
    }
    Write-Output 'Verifying all existing deployed files and recording CHAT.CSV before deployment...'
    $record.before = @(Get-Snapshot)
    foreach ($old in $baseline.files) {
        $actual = @($record.before | Where-Object { $_.path -eq $old.path })
        if ($actual.Count -ne 1 -or $actual[0].sha256 -ne $old.sha256 -or $actual[0].bytes -ne $old.bytes) { throw ('Original file mismatch: '+$old.path) }
    }
    $candidate = @($record.before | Where-Object { $_.path -eq 'QWUART.ELF' })
    if ($candidate[0].bytes -ne 999656 -or $candidate[0].sha256 -ne '0e22b2af854dcc6cd3b6ee4071125ad1bbcb7b05a4df6f038f3995b5087f293c') { throw 'Candidate identity mismatch.' }
    [void](New-Item -ItemType Directory -Path $backupRoot)
    foreach ($name in @('boot.cmd','boot.scr')) {
        Copy-Item -LiteralPath (Join-Path $cardRoot $name) -Destination (Join-Path $backupRoot $name)
        $backed = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $backupRoot $name)).Hash.ToLowerInvariant()
        if ($backed -ne (@($record.before | Where-Object { $_.path -eq $name })[0].sha256)) { throw 'Backup verification failed.' }
        $record.backups += [ordered]@{path=('original-sd-launcher/'+$name);sha256=$backed;verified=$true}
    }
    $null = Get-CardIdentity
    foreach ($name in @('boot.cmd','boot.scr')) {
        $destination = Join-Path $cardRoot $name
        Copy-Item -LiteralPath (Join-Path $result $name) -Destination $destination -Force
        $item = Get-Item -LiteralPath $destination
        $item.Attributes = $item.Attributes -bor [IO.FileAttributes]::Archive
        if ((Get-FileHash -Algorithm SHA256 -LiteralPath $destination).Hash.ToLowerInvariant() -ne $expectedNew[$name]) { throw ('SD read-back failed: '+$name) }
        $record.written += $name
        Write-Output ('Updated and read-back verified: '+$destination)
    }
    Write-Output 'Rechecking every deployed file; only boot.cmd and boot.scr may differ...'
    $record.after = @(Get-Snapshot)
    foreach ($before in $record.before) {
        $after = @($record.after | Where-Object { $_.path -eq $before.path })[0]
        if ($expectedNew.ContainsKey($before.path)) {
            if ($after.sha256 -ne $expectedNew[$before.path] -or -not $after.archive) { throw 'Launcher final verification failed.' }
        } elseif ($after.sha256 -ne $before.sha256 -or $after.bytes -ne $before.bytes) {
            throw ('Protected file changed: '+$before.path)
        }
    }
    $null = Get-CardIdentity
    $record.status = 'sd_updated_verified_pending_cold_boot'
    $record.root_files_after = @(Get-ChildItem -File -LiteralPath $cardRoot | Select-Object Name,Length,Attributes)
    Write-Output 'PASS: only the two existing launcher files changed; models, PL, both ELF files and CHAT.CSV unchanged.'
} catch {
    $record.status = 'failed_stop_inspect_before_boot'
    $record.error = $_.Exception.Message
    throw
} finally {
    $record.finished_at = [DateTimeOffset]::Now.ToString('o')
    [IO.File]::WriteAllText($recordPath,($record | ConvertTo-Json -Depth 10),[Text.UTF8Encoding]::new($false))
    Write-Output ('PC evidence: '+$recordPath)
}
