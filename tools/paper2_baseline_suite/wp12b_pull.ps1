# Pull WP12b rerun inputs from server B into the local (gitignored) mirror; detached-friendly.
$L = 'G:\Project Code\ISRJ_222'
$log = Join-Path $L 'outputs\wp12b_local\pull.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$items = @('data/unet_ablation_3seed_256_20261115/equal_pfa/calibration/test.npz',
           'data/unet_ablation_3seed_256_20261115/equal_pfa/evaluation/test.npz')
foreach ($s in 17, 29, 41, 53, 67) {
    $items += "outputs/paper2_baseline_suite_20261126/E3/ReCoVP-Hybrid/seed_$s/ROW.json"
    $items += "outputs/paper2_baseline_suite_20261126/E3/ReCoVP-Hybrid/seed_$s/SCORES.npz"
    if ($s -ne 67) { $items += "outputs/recovp_hybrid_seed$s/seed_$s/epoch_256.pt"; $items += "outputs/recovp_hybrid_seed$s/seed_$s/epoch_256.json" }
}
foreach ($m in 'RSDR-Net', 'UNet-Tail') { foreach ($s in 17, 29, 41, 53, 67) { $items += "outputs/paper2_baseline_suite_20261126/E3/$m/seed_$s/SCORES.npz" } }
"start $(Get-Date -Format o) items=$($items.Count)" | Out-File $log -Encoding utf8
foreach ($r in $items) {
    $dst = Join-Path $L ($r -replace '/', '\')
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    & scp -q -o ConnectTimeout=15 -o BatchMode=yes "rsdr-b:D:/code/Xin/ISRJ_222/$r" "$dst" 2>&1 | Out-File $log -Append -Encoding utf8
    "{0} {1}" -f (Test-Path -LiteralPath $dst), $r | Out-File $log -Append -Encoding utf8
}
"COPY_DONE $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
