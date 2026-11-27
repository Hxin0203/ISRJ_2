$ErrorActionPreference = 'Stop'
# Upload locally computed WP5b result folders to server B; never overwrite an existing server DONE.json.
$L = 'G:\Project Code\ISRJ_222\outputs\wp5b_local\WP5b\results'
$remote = 'D:/code/Xin/ISRJ_222/outputs/paper2_baseline_suite_20261126/WP5b/results'
Get-ChildItem -LiteralPath $L -Recurse -Filter DONE.json | ForEach-Object {
    $label = $_.Directory.Parent.Name; $tag = $_.Directory.Name
    $exists = & ssh -o ConnectTimeout=15 -o BatchMode=yes rsdr-b "if exist `"$($remote -replace '/','\')\$label\$tag\DONE.json`" (echo YES) else (echo NO)"
    if ($exists -match 'YES') { Write-Output "skip (server already done) $label/$tag"; return }
    & ssh -o ConnectTimeout=15 -o BatchMode=yes rsdr-b "if not exist `"$($remote -replace '/','\')\$label\$tag`" mkdir `"$($remote -replace '/','\')\$label\$tag`"" | Out-Null
    foreach ($f in Get-ChildItem -LiteralPath $_.Directory.FullName -File) {
        & scp -q -o ConnectTimeout=15 -o BatchMode=yes "$($f.FullName)" "rsdr-b:$remote/$label/$tag/$($f.Name)"
    }
    Write-Output "uploaded $label/$tag ($((Get-ChildItem -LiteralPath $_.Directory.FullName -File).Count) files)"
}
