param([string]$Root = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
$Root = (Resolve-Path -LiteralPath $Root).ProviderPath
$items = (Get-Content -LiteralPath (Join-Path $Root 'geometry.json') -Raw -Encoding UTF8 | ConvertFrom-Json)

function Set-ShapeCell($shape, [string]$name, $value) {
    $shape.CellsU($name).FormulaU = [string]$value
}

function Get-ColorFormula([string]$hex) {
    if ($hex -notmatch '^#[a-fA-F0-9]{6}$') { throw "Invalid RGB color $hex" }
    $rgb = for ($i = 1; $i -le 5; $i += 2) { [Convert]::ToInt32($hex.Substring($i, 2), 16) }
    return 'RGB(' + ($rgb -join ',') + ')'
}

function New-PathShape($page, $item) {
    $points = $item.points
    if ($points.Count -lt 2) { throw 'A path requires at least two points.' }
    $closed = $item.kind -eq 'poly'
    $count = $points.Count
    if ($closed) { $count++ }
    $coords = New-Object double[] ($count * 2)
    for ($i = 0; $i -lt $count; $i++) {
        $point = $points[$i % $points.Count]
        $coords[2 * $i] = [double]$point[0] / 100
        $coords[2 * $i + 1] = (656 - [double]$point[1]) / 100
    }
    if ($count -eq 2) { return $page.DrawLine($coords[0], $coords[1], $coords[2], $coords[3]) }
    return $page.DrawPolyline($coords, 0)
}

function Set-ShapeText($shape, $item) {
    $shape.Text = [string]$item.text
    $font = if ($item.font) { $item.font } else { 'Times New Roman' }
    Set-ShapeCell $shape 'Char.Font' ('FONT("' + $font + '")')
    $size = if ($item.size) { $item.size } else { 9 }
    Set-ShapeCell $shape 'Char.Size' ("$size pt")
    $color = if ($item.color) { $item.color } elseif ($item.kind -eq 'text' -and $item.stroke -like '#*') { $item.stroke } else { '#1c3341' }
    Set-ShapeCell $shape 'Char.Color' (Get-ColorFormula $color)
    Set-ShapeCell $shape 'Char.Style' ([int][bool]$item.bold + 2 * [int][bool]$item.italic)
    Set-ShapeCell $shape 'Para.HorzAlign' ([int]($item.align -ne 'left'))
    Set-ShapeCell $shape 'VerticalAlign' 1
    Set-ShapeCell $shape 'Para.SpLine' '-100%'
    foreach ($margin in 'LeftMargin', 'RightMargin', 'TopMargin', 'BottomMargin') {
        Set-ShapeCell $shape $margin '0 pt'
    }
}

$app = $null
$doc = $null
try {
    $app = New-Object -ComObject Visio.InvisibleApp
    $app.AlertResponse = 7
    $app.ScreenUpdating = 0
    $doc = $app.Documents.Add('')
    $page = $doc.Pages.Item(1)
    $page.Name = 'ReCoVP-Hybrid'
    Set-ShapeCell $page.PageSheet 'PageWidth' '12 in'
    Set-ShapeCell $page.PageSheet 'PageHeight' '6.56 in'
    $layers = @($page.Layers.Add('A - Structured estimation'), $page.Layers.Add('B - Complex refinement'), $page.Layers.Add('C - Training objectives'))
    foreach ($item in $items) {
        if ($item.kind -in 'line', 'poly') {
            $shape = New-PathShape $page $item
        } else {
            $left = [double]$item.x / 100
            $right = ([double]$item.x + [double]$item.w) / 100
            $top = (656 - [double]$item.y) / 100
            $bottom = (656 - [double]$item.y - [double]$item.h) / 100
            switch ($item.kind) {
                'ellipse' { $shape = $page.DrawOval($left, $bottom, $right, $top) }
                { $_ -in 'rect', 'text' } { $shape = $page.DrawRectangle($left, $bottom, $right, $top) }
                default { throw "Unknown item kind $($item.kind)" }
            }
        }
        $y = if ($item.kind -in 'line','poly') { [double]$item.points[0][1] } else { [double]$item.y }
        $idx = if ($y -lt 240) { 0 } elseif ($y -lt 508) { 1 } else { 2 }
        $layers[$idx].Add($shape, 0)
        if ($item.name) { $shape.NameU = [string]$item.name }
        if ($item.kind -ne 'text' -and $item.kind -ne 'line' -and $item.fill -like '#*') {
            Set-ShapeCell $shape 'FillPattern' 1
            Set-ShapeCell $shape 'FillForegnd' (Get-ColorFormula $item.fill)
        } else {
            Set-ShapeCell $shape 'FillPattern' 0
        }
        if ($item.kind -ne 'text' -and $item.stroke -like '#*') {
            Set-ShapeCell $shape 'LineColor' (Get-ColorFormula $item.stroke)
            $weight = if ($null -ne $item.lw) { $item.lw } else { 0.7 }
            Set-ShapeCell $shape 'LineWeight' ("$weight pt")
            Set-ShapeCell $shape 'LinePattern' $(if ($item.dashed) { 2 } else { 1 })
            if ($item.arrow) {
                Set-ShapeCell $shape 'EndArrow' 4
                Set-ShapeCell $shape 'EndArrowSize' 1
            }
        } else {
            Set-ShapeCell $shape 'LinePattern' 0
        }
        if ($item.text) { Set-ShapeText $shape $item }
    }
    $doc.SaveAs((Join-Path $Root 'Hybrid_Visio.vsdx'))
    $doc.ExportAsFixedFormat(1, (Join-Path $Root 'Hybrid_Visio.pdf'), 1, 0)
    Write-Output ("Exported {0} editable Visio shapes." -f $page.Shapes.Count)
} finally {
    if ($doc) { $doc.Saved = $true; $doc.Close() }
    if ($app) { $app.Quit() }
}
