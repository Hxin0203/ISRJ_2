"""Source-grounded CorrectionNet figure; coordinates are in 1/100 inch units.

Only schematic icon points are generated here. No training or result data are used.
The editable master and all exports are produced by build_visio.ps1.
"""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
C = dict(ink="#243746", edge="#526775", blue="#5285A3", teal="#92C7C8",
         teal_light="#DCECEE", blue_light="#DFEAF1", pale="#F5F8FA",
         orange="#B86D4E", orange_light="#FAEEE7", gold="#EBD396",
         gold_light="#FBF6E8", gray="#71838D", rule="#C2D0D9", white="#FFFFFF")
items = []
panel = ""


def add(kind, name, **kw):
    items.append(dict(kind=kind, name=name, group=panel, **kw))
    return name


def box(name, x, y, w, h, text="", fill="white", line="edge", size=17,
        bold=False, dash=False, oval=False, weight=1.05):
    add("oval" if oval else "rect", name, x=x, y=y, w=w, h=h,
        text=text, fill=C.get(fill, fill), line=C.get(line, line), size=size,
        bold=bold, dash=dash, weight=weight)
    return name


def text(name, x, y, w, h, value, size=17, bold=False, color="ink", align=1,
         italic=False):
    add("text", name, x=x, y=y, w=w, h=h, text=value, size=size,
        bold=bold, color=C.get(color, color), align=align, italic=italic)


def path(name, pts, color="ink", arrow=True, weight=1.15, dash=False,
         start=None, end=None):
    add("path", name, points=pts, line=C.get(color, color), arrow=arrow,
        weight=weight, dash=dash, start=start, end=end)


def port(shape, x, y):
    # Relative coordinates are Visio coordinates: bottom-left origin.
    return dict(shape=shape, x=x, y=y)


def arrow(name, a, b, pts, color="ink", dash=False):
    path(name, pts, color=color, dash=dash, start=a, end=b)


def dot(name, x, y, color="ink"):
    box(name, x-3, y-3, 6, 6, fill=color, line=color, oval=True, weight=0.6)


def waveform(name, x, y, w, h, received):
    box(name+"_frame", x, y, w, h, line="rule", weight=.6)
    for channel in range(2):
        cy = y+h*(.29+.43*channel)
        path(name+f"_axis_{channel}", [[x+7, cy], [x+w-7, cy]],
             color="rule", arrow=False, weight=.45)
        pts = []
        for n in range(260):
            t = n/259
            envelope = (1 if .16 < t < .75 else 0)
            v = .35*envelope*math.sin(2*math.pi*(8*t+13*t*t)+channel*math.pi/2)
            if received:
                for a, b in [(.26,.35), (.49,.59), (.73,.83)]:
                    if a < t < b:
                        v += .55*math.sin(2*math.pi*(29*t+18*t*t)+channel*.9)
            v += .035*math.sin(2*math.pi*91*t+channel)
            pts.append([x+8+(w-16)*t, cy-h*.23*v])
        path(name+f"_native_IQ_{channel}", pts, color="blue", arrow=False, weight=.6)


# (a) Inference. The return value of CorrectionNet is c, not the restored y-c.
panel = "A_Inference"
text("title_a", 22, 10, 1110, 40, "(a)  Overall architecture of CorrectionNet", 21, True, align=0)
box("panel_a", 20, 58, 1560, 250, fill="pale", line="rule", weight=.75)
text("input_title", 41, 126, 145, 28, "Observed I/Q", 17)
waveform("input", 47, 160, 135, 72, True)
text("input_y", 48, 235, 133, 29, "y ∈ ℂ<sup>N</sup>", 18, italic=True)
norm = box("rms_dechirp", 238, 158, 190, 76, "RMS + dechirp\nz = Dy / a(y)", size=17)
core = box("learned_core", 500, 154, 270, 84, "Residual backbone\nF<sub>θ</sub>  ·  panel (b)", fill="teal", size=18, bold=True)
inverse = box("inverse_dechirp", 842, 158, 255, 76, "Inverse dechirp + rescale\nc = a(y)D<sup>−1</sup>δ<sub>z</sub>", size=17)
summ = box("subtraction", 1233, 177, 38, 38, "Σ", size=23, oval=True)
waveform("output", 1381, 160, 135, 72, False)
text("output_title", 1349, 126, 200, 28, "Restored I/Q", 17)
text("output_T", 1338, 235, 226, 29, "T<sub>θ</sub>(y) = y − C<sub>θ</sub>(y)", 18, italic=True)
path("input_to_norm", [[182,196],[238,196]], end=port(norm,0,.5))
dot("input_split", 210, 196)
path("identity_bypass", [[210,196],[210,101],[1252,101],[1252,177]], end=port(summ,.5,1))
text("bypass_label", 530, 72, 390, 27, "Observation bypass", 17, color="gray")
text("plus_port", 1264, 148, 28, 28, "+", 18)
text("minus_port", 1196, 199, 28, 27, "−", 20)
arrow("norm_to_core",port(norm,1,.5),port(core,0,.5),[[428,196],[500,196]])
text("z_label", 436, 165, 56, 26, "z", 19, italic=True)
arrow("core_to_inverse",port(core,1,.5),port(inverse,0,.5),[[770,196],[842,196]])
text("delta_label", 780, 162, 53, 28, "δ<sub>z</sub>", 19, italic=True)
arrow("correction_to_subtract",port(inverse,1,.5),port(summ,0,.5),[[1097,196],[1233,196]])
text("correction_label", 1098, 160, 132, 28, "c = C<sub>θ</sub>(y)", 18, italic=True)
path("subtract_to_output",[[1271,196],[1381,196]],start=port(summ,1,.5))
arrow("scale_to_inverse",port(norm,.5,0),port(inverse,.5,0),[[333,234],[333,276],[969.5,276],[969.5,234]],color="blue",dash=True)
text("scale_label", 486, 252, 365, 26, "a(y): per-observation RMS", 16, color="blue")
text("icons_schematic", 46, 275, 235, 23, "Waveforms are schematic", 15, color="gray", align=0)

# (b) Three scales; decoder channels are concatenated, never added.
panel = "B_Residual_Backbone"
text("title_b",22,325,940,36,"(b)  Three-scale residual encoder–decoder F<sub>θ</sub>",21,True,align=0)
box("panel_b",20,372,930,377,fill="pale",line="rule",weight=.75)
inp=box("iq_tensor",42,408,100,56,"I/Q z\n2 × 1800",size=16)
e0=box("enc0",184,403,166,66,"Res 16\n16 × 1800",fill="teal",size=18)
e1=box("enc1",184,518,166,66,"Res 32, s = 2\n32 × 900",fill="teal",size=18)
e2=box("enc2",184,633,166,66,"Res 64, s = 2\n64 × 450",fill="teal",size=18)
c0=box("concat0",520,408,84,56,"Concat\n48 ch",fill="white",size=15)
c1=box("concat1",520,523,84,56,"Concat\n96 ch",fill="white",size=15)
d0=box("dec0",631,403,163,66,"Res 16\n16 × 1800",fill="blue_light",size=18)
d1=box("dec1",631,518,163,66,"Res 32\n32 × 900",fill="blue_light",size=18)
head=box("head",837,406,90,60,"Conv₁\n16 → 2",fill="blue_light",size=16)
arrow("iq_to_enc0",port(inp,1,.5),port(e0,0,.5),[[142,436],[184,436]])
arrow("down0",port(e0,.5,0),port(e1,.5,1),[[267,469],[267,518]])
text("down0_label",277,478,55,28,"↓2",17,color="gray")
arrow("down1",port(e1,.5,0),port(e2,.5,1),[[267,584],[267,633]])
text("down1_label",277,593,55,28,"↓2",17,color="gray")
arrow("skip0",port(e0,1,.5),port(c0,0,.5),[[350,436],[520,436]],color="blue")
arrow("skip1",port(e1,1,.5),port(c1,0,.5),[[350,551],[520,551]],color="blue")
text("skip0_label",391,402,110,26,"skip",16,color="blue")
text("skip1_label",391,517,110,26,"skip",16,color="blue")
arrow("concat_to_dec0",port(c0,1,.5),port(d0,0,.5),[[604,436],[631,436]])
arrow("concat_to_dec1",port(c1,1,.5),port(d1,0,.5),[[604,551],[631,551]])
arrow("upsample1",port(e2,1,.5),port(c1,.5,0),[[350,666],[562,666],[562,579]])
text("upsample1_label",388,630,158,28,"Linear ↑2",17)
arrow("upsample0",port(d1,.5,1),port(c0,.5,0),[[712.5,518],[712.5,491],[562,491],[562,464]])
text("upsample0_label",378,472,165,26,"Linear ↑2",16)
arrow("dec0_to_head",port(d0,1,.5),port(head,0,.5),[[794,436],[837,436]])
path("head_to_delta",[[882,466],[882,498]],start=port(head,.5,0))
text("output_tensor",804,505,138,53,"δ<sub>z</sub>\n2 × 1800",17,italic=True)
text("backbone_legend",49,707,870,28,"Res: block (c);  Concat: channels;  tensor sizes: channels × samples",16,color="gray",align=0)

# (c) Every active block changes channel count, so its skip is a 1x1 projection.
panel = "C_Residual_Block"
text("title_c",984,325,594,36,"(c)  Residual convolution block",21,True,align=0)
box("panel_c",980,372,600,377,fill="pale",line="rule",weight=.75)
text("block_h",1190,382,70,28,"h",19,italic=True)
v5=box("conv5",1142,433,168,52,"Conv₅, stride s",fill="teal",size=17)
ng=box("norm_gelu",1142,503,168,52,"GN → GELU",fill="teal_light",size=17)
v3=box("conv3_gn",1142,573,168,52,"Conv₃ → GN",fill="teal",size=17)
proj=box("projection",1000,503,118,67,"Conv₁\nstride s",fill="blue_light",size=16)
addnode=box("residual_add",1208,658,36,36,"+",size=22,oval=True)
gelu=box("final_gelu",1347,652,130,48,"GELU",fill="teal_light",size=18)
path("h_to_conv5",[[1226,410],[1226,433]],end=port(v5,.5,1))
dot("h_split",1226,419)
path("h_to_projection",[[1226,419],[1059,419],[1059,503]],end=port(proj,.5,1))
arrow("conv5_to_gn",port(v5,.5,0),port(ng,.5,1),[[1226,485],[1226,503]])
arrow("gn_to_conv3",port(ng,.5,0),port(v3,.5,1),[[1226,555],[1226,573]])
arrow("conv3_to_add",port(v3,.5,0),port(addnode,.5,1),[[1226,625],[1226,658]])
arrow("projection_to_add",port(proj,.5,0),port(addnode,0,.5),[[1059,570],[1059,676],[1208,676]],color="blue")
arrow("add_to_gelu",port(addnode,1,.5),port(gelu,0,.5),[[1244,676],[1347,676]])
path("block_output",[[1477,676],[1549,676]],start=port(gelu,1,.5))
text("block_output_h",1490,638,73,30,"h′",19,italic=True)
text("block_legend",997,707,566,28,"GN: 8 groups; s = 2 for downsampling, otherwise 1",16,color="gray",align=0)

# (d) Paired supervision. S,J,E are training variables, never inference inputs.
panel = "D_Paired_Training"
text("title_d",22,770,1230,39,"(d)  Four-state paired supervision (training only)",21,True,align=0)
box("panel_d",20,823,1560,310,fill="white",line="orange",dash=True,weight=.9)
text("shared_realizations",43,834,646,28,"Shared realizations of S, J and E in each quartet",17,color="gray")
shared=box("shared_C",300,887,190,180,"CorrectionNet\nC<sub>θ</sub>\n\nShared weights",fill="teal",size=18,bold=True)
states=[("00","E","0"),("10","S + E","0"),("01","J + E","J"),("11","S + J + E","J")]
text("states_heading",48,864,194,28,"Input y<sub>ab</sub>",16)
text("targets_heading",526,864,183,28,"c<sub>ab</sub> / q<sub>ab</sub>",17)
for i,(index,formula,q) in enumerate(states):
    cy=917+45*i
    state=box("state_"+index,48,cy-18,194,36,formula,size=18,
              fill="white" if i<2 else "orange_light",line="rule",weight=.8)
    t=(1067-cy)/180
    arrow("state_to_C_"+index,port(state,1,.5),port(shared,0,t),[[242,cy],[300,cy]])
    path("C_to_prediction_"+index,[[490,cy],[535,cy]],start=port(shared,1,t))
    text("prediction_"+index,541,cy-18,163,36,f"c<sub>{index}</sub>  /  {q}",18,italic=True)
text("references",45,1082,655,33,"References u<sub>ab</sub>: (E, S + E, E, S + E)",16,color="gray")
box("loss_panel",730,846,825,228,fill="gold_light",line="rule",weight=.8)
text("total_loss",747,853,792,37,"L = L<sub>end</sub> + 0.1 L<sub>PC</sub> + L<sub>damage</sub>",21,True)
path("loss_rule",[[750,896],[1535,896]],color="rule",arrow=False,weight=.6)
text("endpoint_loss",748,910,786,38,"L<sub>end</sub> = ⟨ ‖c<sub>ab</sub> − q<sub>ab</sub>‖² / max(‖u<sub>ab</sub>‖², ε) ⟩<sub>ab</sub>",19)
text("pc_loss",748,959,786,38,"L<sub>PC</sub> = ⟨ ‖H(c<sub>ab</sub> − q<sub>ab</sub>)‖² / max(‖Hu<sub>ab</sub>‖², ε) ⟩<sub>ab</sub>",19)
text("damage_loss",748,1008,786,43,"L<sub>damage</sub> = ‖c<sub>11</sub> − c<sub>01</sub>‖² / max(‖S‖², ε)",19,color="orange")
text("loss_legend",740,1081,804,35,"H: matched filter;  ⟨·⟩<sub>ab</sub>: mean over four states;  ε = 10<sup>−12</sup>",16,color="gray")

# A full line-sized mark-up run is never rasterized: text, subscripts and
# superscripts are passed to Visio Characters as editable character runs.
def plain_markup(s):
    return re.sub(r"</?(sub|sup)>", "", s)


cfg=dict(width=1600,height=1150,page_width_in=16,page_height_in=11.5,
         font="Times New Roman",palette=C,items=items,
         grouping_order=["A_Inference","B_Residual_Backbone","C_Residual_Block","D_Paired_Training"],
         schematic_waveforms=True)
(ROOT/"figure_geometry.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
protected=[WORKSPACE/"paper2/network.py",WORKSPACE/"paper2/vendor/legacy_model.py",
           WORKSPACE/"manuscript/ISRJ_2_初稿.md",
           Path("G:/Files/Markdown/ISRJ/草稿/草稿1113/模型图输入波形修订_v3_20261115/RSDR-Net_editable_input_v3_20261115.vsdx"),
           Path("C:/Users/admin/AppData/Local/Temp/codex-clipboard-d5da421d-de90-42ad-8e1c-a9a13803cbee.png")]
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
manifest_path=ROOT/"SOURCE_SHA256.json"
if not manifest_path.exists():
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
else:
    assert json.loads(manifest_path.read_text(encoding="utf-8"))==manifest,"Protected source changed"
print(f"Prepared {len(items)} native objects in four semantic panels.")
