param([string]$Root=$PSScriptRoot)
$ErrorActionPreference='Stop'
$cfg=Get-Content -LiteralPath (Join-Path $Root 'figure_geometry.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$base='RSDR_Net_Architecture'
$out=Join-Path $Root ($base+'.vsdx')
function Cell($s,[string]$n,$f){$s.CellsU($n).FormulaU=[string]$f}
function Color([string]$v){
  if($v -notmatch '^#[0-9a-fA-F]{6}$'){throw "Unsupported color: $v"}
  $r=[Convert]::ToInt32($v.Substring(1,2),16);$g=[Convert]::ToInt32($v.Substring(3,2),16);$b=[Convert]::ToInt32($v.Substring(5,2),16)
  return "RGB($r,$g,$b)"
}
function X([double]$v){return $v/100.0}
function Y([double]$v){return ([double]$cfg.height-$v)/100.0}
function StyleText($s,$o){
  $s.Text=[string]$o.text
  Cell $s 'Char.Font' ('FONT("'+$cfg.font+'")');Cell $s 'Char.AsianFont' ('FONT("'+$cfg.font+'")');Cell $s 'Char.ComplexScriptFont' ('FONT("'+$cfg.font+'")')
  Cell $s 'Char.Size' ([string]$o.size+' pt')
  $ink=$cfg.palette.ink;if($o.color){$ink=$o.color};Cell $s 'Char.Color' (Color $ink)
  $style=0;if($o.bold){$style+=1};if($o.italic){$style+=2};Cell $s 'Char.Style' $style
  $align=1;if($null -ne $o.align){$align=$o.align};Cell $s 'Para.HorzAlign' $align;Cell $s 'Para.SpLine' '-105%';Cell $s 'VerticalAlign' 1
  foreach($margin in 'LeftMargin','RightMargin','TopMargin','BottomMargin'){Cell $s $margin '0 pt'}
  Cell $s 'TxtWidth' 'Width'
}
$app=$null;$doc=$null
try{
  $app=New-Object -ComObject Visio.InvisibleApp;$app.AlertResponse=7
  $doc=$app.Documents.Add('');$page=$doc.Pages.Item(1);$page.Name='RSDR-Net'
  Cell $page.PageSheet 'PageWidth' ([string]$cfg.page_width_in+' in');Cell $page.PageSheet 'PageHeight' ([string]$cfg.page_height_in+' in')
  $groups=@{};$native=@{};$connectors=@();$count=0;$textCount=0
  foreach($o in $cfg.items){
    if(-not $groups.ContainsKey($o.group)){$groups[$o.group]=New-Object System.Collections.Generic.List[object]}
    if($o.kind -eq 'path'){
      $pts=New-Object double[] ($o.points.Count*2)
      for($i=0;$i -lt $o.points.Count;$i++){$pts[2*$i]=X $o.points[$i][0];$pts[2*$i+1]=Y $o.points[$i][1]}
      if($o.points.Count -eq 2){$s=$page.DrawLine($pts[0],$pts[1],$pts[2],$pts[3])}else{$s=$page.DrawPolyline($pts,0)}
      Cell $s 'FillPattern' 0;Cell $s 'LineColor' (Color $o.line);Cell $s 'LineWeight' ([string]$o.weight+' pt');Cell $s 'LinePattern' $(if($o.dash){2}else{1});Cell $s 'LineCap' 0
      if($o.arrow){Cell $s 'EndArrow' 4;Cell $s 'EndArrowSize' 1}
      if($o.start -or $o.end){$connectors+=@{shape=$s;spec=$o}}
    }else{
      $x1=X $o.x;$y1=Y ($o.y+$o.h);$x2=X ($o.x+$o.w);$y2=Y $o.y
      if($o.kind -eq 'oval'){$s=$page.DrawOval($x1,$y1,$x2,$y2)}else{$s=$page.DrawRectangle($x1,$y1,$x2,$y2)}
      if($o.kind -eq 'text'){Cell $s 'FillPattern' 0;Cell $s 'LinePattern' 0}
      else{Cell $s 'FillPattern' 1;Cell $s 'FillForegnd' (Color $o.fill);Cell $s 'LinePattern' $(if($o.dash){2}else{1});Cell $s 'LineColor' (Color $o.line);Cell $s 'LineWeight' ([string]$o.weight+' pt');Cell $s 'Rounding' '0 in'}
      if($o.text){StyleText $s $o;$textCount++}
    }
    $s.NameU=$o.name;$native[$o.name]=$s;$groups[$o.group].Add($s);$count++
  }
  $glued=0
  foreach($entry in $connectors){$s=$entry.shape;$o=$entry.spec;if($s.OneD -eq 0){$s.OneD=1};if($o.start){$s.CellsU('BeginX').GlueToPos($native[$o.start.shape],[double]$o.start.x,[double]$o.start.y);$glued++};if($o.end){$s.CellsU('EndX').GlueToPos($native[$o.end.shape],[double]$o.end.x,[double]$o.end.y);$glued++}}
  foreach($name in $cfg.grouping_order){$sel=$page.CreateSelection(0,0,0);foreach($s in $groups[$name]){$sel.Select($s,2)};$g=$sel.Group();$g.NameU=$name;Cell $g 'SelectMode' 1}
  $doc.SaveAs($out)
  $page.Export((Join-Path $Root ($base+'.svg')))
  $app.Settings.SetRasterExportResolution(2,250,250,0);$page.Export((Join-Path $Root ($base+'.png')))
  $doc.ExportAsFixedFormat(1,(Join-Path $Root ($base+'.pdf')),1,0)
  @{status='EXPORTED';native_items=$count;native_text_shapes=$textCount;panel_groups=$page.Shapes.Count;glued_endpoints=$glued;output=$out;utc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $Root 'BUILD_RECEIPT.json') -Encoding UTF8
  Write-Output "VISIO_EXPORT_COMPLETE: $count native shapes, $textCount text boxes, $($cfg.grouping_order.Count) panels."
}finally{if($doc){$doc.Saved=$true;$doc.Close()};if($app){$app.Quit()}}
