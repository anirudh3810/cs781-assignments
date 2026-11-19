# Usage: .\make_submission.ps1 -Name Name -RollNo RollNo [-ProjectDir path]
#    or: .\make_submission.ps1 Name RollNo [path]
# Follow the naming convention from discord. Here are a few examples.
# Example: .\make_submission.ps1 "John" 1234567890
# Example: .\make_submission.ps1 "John-Doe" 1234567890
# Example: .\make_submission.ps1 "John-Barry-Doe" 1234567890

# Builds CS781-A3-Name-RollNo.zip with files at the zip root.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Name,
    [Parameter(Mandatory = $true, Position = 1)][string]$RollNo,
    [Parameter(Position = 2)][string]$ProjectDir = "."
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$Src     = (Resolve-Path $ProjectDir).Path
$ZipName = "CS781-A3-$Name-$RollNo.zip"
$Out     = Join-Path $Src $ZipName

$TopFiles = @(
    "notebook.ipynb", "a1_tokenizer.py", "attention.py", "model_arch.py", "rope.py",
    "optim_muon.py", "model_builtin.py", "common.py", "analysis_log.json",
    "requirements.txt", "download_data.py", "decoder.svg"
)
$RequiredPerQ = @("model.pt", "config.json", "metrics.json")
$Questions    = @("q1", "q2", "q3", "q4", "q5")

$missing = $false

# --- Validate top-level files ---
foreach ($f in $TopFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $Src $f) -PathType Leaf)) {
        Write-Host "MISSING: $f"; $missing = $true
    }
}

# --- Validate per-question files ---
foreach ($q in $Questions) {
    $qdir = Join-Path $Src "submissions\$q"
    if (-not (Test-Path -LiteralPath $qdir -PathType Container)) {
        Write-Host "MISSING: submissions/$q/"; $missing = $true; continue
    }
    foreach ($f in $RequiredPerQ) {
        if (-not (Test-Path -LiteralPath (Join-Path $qdir $f) -PathType Leaf)) {
            Write-Host "MISSING: submissions/$q/$f"; $missing = $true
        }
    }
    if (-not (Get-ChildItem -LiteralPath $qdir -Filter *.py -File -ErrorAction SilentlyContinue)) {
        Write-Host "WARNING: no saved .py source files in submissions/$q/"
    }
}

if ($missing) {
    Write-Error "Aborting: fix missing files above."
    exit 1
}

# --- Build zip directly (forward-slash entry names, so it unzips cleanly on Linux too) ---
if (Test-Path -LiteralPath $Out) { Remove-Item -LiteralPath $Out -Force }

$zip = [System.IO.Compression.ZipFile]::Open($Out, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    function Add-Entry([string]$FilePath, [string]$EntryName) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $FilePath, $EntryName, [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }

    foreach ($f in $TopFiles) {
        Add-Entry (Join-Path $Src $f) $f
    }

    foreach ($q in $Questions) {
        $qdir = Join-Path $Src "submissions\$q"

        foreach ($f in $RequiredPerQ) {
            Add-Entry (Join-Path $qdir $f) "submissions/$q/$f"
        }

        # Saved Python source files (recursive, preserves subdirs, skips caches)
        Get-ChildItem -LiteralPath $qdir -Filter *.py -File -Recurse |
            Where-Object { $_.FullName -notmatch '[\\/](__pycache__|\.ipynb_checkpoints)[\\/]' } |
            ForEach-Object {
                $rel = $_.FullName.Substring($qdir.Length).TrimStart('\', '/') -replace '\\', '/'
                Add-Entry $_.FullName "submissions/$q/$rel"
            }
    }
}
finally {
    $zip.Dispose()
}

# --- Report ---
$sizeMB = "{0:N2} MB" -f ((Get-Item -LiteralPath $Out).Length / 1MB)
Write-Host "Created: $Out ($sizeMB)"
Write-Host "--- Contents ---"
$read = [System.IO.Compression.ZipFile]::OpenRead($Out)
try {
    $read.Entries | Sort-Object FullName | ForEach-Object {
        "{0,12}  {1}" -f $_.Length, $_.FullName
    }
}
finally {
    $read.Dispose()
}