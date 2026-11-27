# Pull what the remaining WP5b jobs need into the local gitignored mirror (detached-friendly).
$L = 'G:\Project Code\ISRJ_222'
$log = Join-Path $L 'outputs\wp5b_local\pull.log'
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$items = @('data/paper2_baseline_suite_20261126/wp5b/DATA_MANIFEST.json')
foreach ($c in 'w06','w18','w36','w54','w72','eta_p145','eta_p217') { $items += "data/paper2_baseline_suite_20261126/wp5b/$c/test_-10_20.npz"; $items += "data/paper2_baseline_suite_20261126/wp5b/$c/test_-10_20.receipt.json" }
$conv = @{ 'CorrectionNet-Tail' = @('correction_mf_tail_20261117_v1/arm', @(17, 41));
           'UNet-Tail'          = @('unet_mf_tail_screen_20261117_v2/tail', @(29));
           'UNet-NoiseNorm'     = @('unet_noise_norm_screen_20261117_v1/norm', @(17, 41));
           'UNet-Freq'          = @('unet_frequency_screen_20261117_v1/factor', @(29));
           'UNet-Dilation'      = @('unet_dilation_screen_20261117_v1/factor', @(17, 29, 41)) }
foreach ($f in $conv.Keys) {
    foreach ($s in $conv[$f][1]) {
        foreach ($n in 'epoch_256.pt', 'epoch_256.json', 'TRAINING_COMPLETE.json') { $items += "outputs/$($conv[$f][0])/seed_$s/$n" }
        $items += "outputs/paper2_baseline_suite_20261126/E3/$f/seed_$s/ROW.json"
    }
}
$items += 'outputs/paper2_baseline_suite_20261126/E7/equal_pfa/rsdr_net/ROW.json'
"start $(Get-Date -Format o) items=$($items.Count)" | Out-File $log -Encoding utf8
foreach ($r in $items) {
    $dst = Join-Path $L ($r -replace '/', '\')
    New-Item -ItemType Directory -Force -Path (Split-Path $dst) | Out-Null
    & scp -q -o ConnectTimeout=15 -o BatchMode=yes "rsdr-b:D:/code/Xin/ISRJ_222/$r" "$dst" 2>&1 | Out-File $log -Append -Encoding utf8
    "{0} {1}" -f (Test-Path -LiteralPath $dst), $r | Out-File $log -Append -Encoding utf8
}
"COPY_DONE $(Get-Date -Format o)" | Out-File $log -Append -Encoding utf8
