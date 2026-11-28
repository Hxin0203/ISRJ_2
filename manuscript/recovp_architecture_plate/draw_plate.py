"""Typeset the four-panel figure using the previously published signal insets."""
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
STEM = HERE / "ReCoVP_Hybrid_plate"
W, H = 1480, 741
items = []
ink, muted, edge = "#203c4a", "#476773", "#b8c9cc"
teal, blue, gold, rose = "#b4d7d0", "#bcd2df", "#f1d99e", "#efc7b8"
green, rust = "#367d78", "#a2644e"
asset_dir = "../jest_signal_insets/receiver_view/"

def rect(x,y,w,h,fill,stroke="none",sw=1):
    items.append(dict(type="rect",x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,sw=sw))

def poly(points, fill, stroke="none", sw=1):
    items.append(dict(type="poly",points=points,fill=fill,stroke=stroke,sw=sw))

def line(points, stroke=ink, sw=1.65, dash=False, arrow=False):
    items.append(dict(type="line",points=points,stroke=stroke,sw=sw,dash=dash))
    if arrow:
        (a,b),(x,y) = points[-2:]
        n = math.hypot(x-a,y-b)
        u,v = (x-a)/n,(y-b)/n
        poly([(x,y),(x-7*u+3.2*v,y-7*v-3.2*u),(x-7*u-3.2*v,y-7*v+3.2*u)],stroke)

def text(x,y,s,size=17,anchor="start",color=ink,bold=False,italic=False,font="Calibri"):
    items.append(dict(type="text",x=x,y=y,text=s,size=size,anchor=anchor,
                      color=color,bold=bold,italic=italic,font=font))

def image(x,y,w,h,path):
    items.append(dict(type="image",x=x,y=y,w=w,h=h,path=asset_dir+path))

def waveform(x,y,w,h,name):
    svg=ET.parse(HERE/asset_dir/"active_windows"/(name+".svg")).getroot()
    _,_,width,height=map(float,svg.attrib["viewBox"].split())
    trace=svg.find(".//*[@id='line2d_1']/{*}path").attrib["d"]
    samples=re.findall(r"[ML]\s+([\d.eE+-]+)\s+([\d.eE+-]+)",trace)
    line([(x+w*float(a)/width,y+h*float(b)/height) for a,b in samples],"#1a5276",.57)

def disk(x,y,r,fill,stroke="none",sw=1):
    poly([(x+r*math.cos(t*math.pi/32),y+r*math.sin(t*math.pi/32)) for t in range(64)],fill,stroke,sw)

def diff(x,y):
    disk(x,y,13,"#ffffff",ink,1.7)
    line([(x-6,y),(x+6,y)],ink,1.7)

def tag(x,y,letter,title):
    text(x,y,"("+letter+")",24,color=muted,font="Cambria",bold=True)
    text(x+39,y,title,25,font="Cambria",bold=True)

def module(x,y,w,h,labels,color):
    rect(x,y,w,h,color,ink,1.15)
    for i,label in enumerate(labels):
        text(x+w/2,y+h/2+6+(i-(len(labels)-1)/2)*20,label,19,"middle",bold=True)

def slab(x,y,w,h,d,pal,cols):
    face,top,side,grid=pal
    t=d*.58
    poly([(x,y),(x+d,y-t),(x+w+d,y-t),(x+w,y)],top,ink,1.1)
    poly([(x+w,y),(x+w+d,y-t),(x+w+d,y+h-t),(x+w,y+h)],side,ink,1.1)
    rect(x,y,w,h,face,ink,1.1)
    for i in range(1,cols):
        xp=x+i*w/cols
        line([(xp,y+h),(xp,y),(xp+d,y-t)],grid,.65)
    for i in range(1,3):
        yp=y+h*i/3
        line([(x,yp),(x+w,yp),(x+w+d,yp-t)],grid,.6)

def figure():
    rect(0,0,W,H,"#ffffff")
    # Separate overview and magnifications with a consistent two-column grid.
    for x,w in [(184,545),(746,535)]:
        rect(x,15,w,254,"#f7f9f8",edge,1)
    rect(20,309,435,409,"#f9fbfa",edge,1)
    rect(475,309,985,409,"#f9fbfc",edge,1)
    tag(201,48,"a","Structured estimation")
    tag(763,48,"b","Complex residual refinement")
    tag(37,347,"c","Local dictionary")
    tag(492,347,"d","Inside the complex U-Net")
    for x1,x2,y in [(202,711,60),(764,1263,60),(38,437,362),(493,1442,362)]:
        line([(x1,y),(x2,y)],edge,.8)

    # Main observation bypass and two subtractive updates.
    waveform(16,70,148,54,"received_overlap")
    text(91,143,"ISRRJ observation",18,"middle",bold=True)
    text(91,165,"Input  y",18,"middle",color=muted,italic=True)
    line([(165,97),(665,97)],arrow=True)
    line([(691,97),(1218,97)],arrow=True)
    line([(1244,97),(1300,97)],arrow=True)
    text(843,87,"Initial recovery  û₁",18,"middle",color=muted)
    diff(678,97)
    diff(1231,97)
    text(1366,103,"Output  û",21,"middle",bold=True)
    waveform(1304,121,150,52,"clean_lfm")
    text(1380,197,"Clean LFM",18,"middle",bold=True)
    text(1380,218,"reference template",17,"middle",color=muted)

    module(244,189,153,58,["MF-guided","dictionary"],teal)
    module(479,189,153,58,["Variable","projection"],gold)
    module(479,126,153,39,["Evidence gate"],rose)
    line([(218,97),(218,218),(244,218)],arrow=True)
    disk(218,97,3.4,ink)
    line([(397,218),(479,218)],arrow=True)
    line([(320,189),(320,145),(479,145)],green,1.6,True,True)
    line([(555,165),(555,189)],green,1.6,True,True)
    line([(632,218),(860,218)],arrow=True)
    line([(678,218),(678,110)],arrow=True)
    disk(678,218,3.4,ink)
    text(760,207,"ĵ₁",20,"middle",italic=True)
    line([(798,97),(798,185),(860,185)],arrow=True)
    disk(798,97,3.4,ink)
    module(860,168,150,66,["Complex","U-Net"],blue)
    module(1082,178,110,46,["Pulse gate"],rose)
    line([(1010,201),(1082,201)],arrow=True)
    text(1045,190,"δ̃",22,"middle",italic=True)
    line([(1192,201),(1231,201),(1231,110)],arrow=True)
    text(1247,165,"δ₂",20,"middle",italic=True)
    for x,y,w,h,target,c in [(239,184,163,68,215,green),(855,163,160,76,990,rust)]:
        line([(x,y),(x+w,y),(x+w,y+h),(x,y+h),(x,y)],c,1,True)
        mid=x+w/2
        line([(mid,y+h),(mid,278),(target,309)],c,1.2,True)

    # Insets use the actual cropped observation and its matching dictionary window.
    text(42,399,"16 dictionary atoms",20,bold=True)
    text(42,423,"Ordered by the time of half cumulative energy",16,color=muted)
    hx,hy,hw,hh=43,448,386,155
    image(hx,hy,hw,hh,"local_dictionary/dictionary_inset.png")
    for t in (4,6,8,10):
        xp=hx+(t-3.55)/(11.433333333-3.55)*hw
        line([(xp,hy+hh),(xp,hy+hh+5)],muted,1)
        text(xp,628,str(t),16,"middle",color=muted)
    text(237,651,"Fast time (μs)",17,"middle")
    from matplotlib.colors import LinearSegmentedColormap, to_hex
    cm=LinearSegmentedColormap.from_list("dictionary",[(0,"#112649"),(.28,"#19538a"),(.48,"#3989b2"),(.62,"#9dbfc0"),(.75,"#ebcf77"),(.88,"#ffdc54"),(1,"#fff4ae")])
    for i in range(46):
        rect(44+3*i,671,3.1,9,to_hex(cm(i/45)))
    text(43,701,"−32",15,color=muted)
    text(183,701,"0 dB",15,"end",color=muted)
    text(202,681,"RMS magnitude",17,bold=True)
    text(202,701,"relative to dictionary peak",15,color=muted)

    # Equal-resolution pairs share a baseline, their skips a horizontal route.
    enc=("#acd0c7","#d9ebe6","#76a69e","#6f9d94")
    dec=("#b2ccdd","#dae7ee","#81a3b8","#7797aa")
    neck=("#e9ca87","#f7e7be","#c3a469","#bfa269")
    text(726,400,"Encoder",21,"middle",bold=True,font="Cambria")
    text(1164,400,"Decoder",21,"middle",bold=True,font="Cambria")
    text(949,420,"Skip connections",17,"middle",color=muted)
    levels=[(665,440,124,23,12,1092,12),(710,508,88,23,17,1065,9),(756,575,62,23,22,1042,6)]
    labels=["T × 9","T/2 × 13","T/4 × 17"]
    for i,(x,y,w,h,d,dx,n) in enumerate(levels):
        line([(x+w+d,y+9),(dx,y+9)],"#7695a1",1.4,arrow=True)
        slab(x,y,w,h,d,enc,n)
        slab(dx,y,w,h,d,dec,n)
        text((x+w+d+dx)/2,y+1,labels[i],17,"middle",color=muted)
    text(500,438,"RMS-normalized",17,bold=True)
    text(500,462,"input pair",17,color=muted)
    line([(590,450),(665,450)],arrow=True)
    line([(1228,450),(1261,450)],arrow=True)
    module(1261,429,104,43,["Cplx conv"],blue)
    line([(1365,450),(1397,450)],arrow=True)
    text(1411,457,"δ̃",24,"middle",italic=True)
    text(1358,494,"Correction proposal",16,"middle",color=muted)
    for points in [[(728,463),(728,485),(754,485),(754,499)],
                   [(752,531),(752,551),(786,551),(786,562)],
                   [(790,598),(790,655),(861,655)],
                   [(1011,655),(1075,655),(1075,598)],
                   [(1076,562),(1076,551),(1108,551),(1108,531)],
                   [(1109,498),(1109,484),(1152,484),(1152,463)]]:
        line(points,ink,1.55,arrow=True)
    text(741,551,"Avg. pool  ↓2",17,"end",color=muted)
    text(1121,551,"↑2  Upsample",17,color=muted)
    slab(862,642,126,25,23,neck,9)
    text(933,620,"T/8 × 21",17,"middle",color=muted)
    text(936,696,"Dilated bottleneck",20,"middle",bold=True,font="Cambria")
    text(493,699,"Gridded blocks: complex features",16,color=muted)
    text(1438,699,"Length × complex channels",16,"end",color=muted)

def export():
    (HERE/"layout.json").write_text(json.dumps(dict(width=W,height=H,items=items),ensure_ascii=False),encoding="utf-8")
    fig=plt.figure(figsize=(W/100,H/100))
    ax=fig.add_axes([0,0,1,1],xlim=(0,W),ylim=(H,0))
    ax.axis("off")
    # One layout unit is one pixel at the 100-dpi design resolution.
    for z,el in enumerate(items):
        kind=el["type"]
        if kind=="text":
            ax.text(el["x"],el["y"],el["text"],fontsize=el["size"]*.72,
                    fontfamily=el["font"],fontweight="bold" if el["bold"] else "normal",
                    style="italic" if el["italic"] else "normal",color=el["color"],
                    ha={"start":"left","middle":"center","end":"right"}[el["anchor"]],
                    va="baseline",zorder=z)
        elif kind=="image":
            x,y,w,h=[el[k] for k in ("x","y","w","h")]
            with Image.open(HERE/el["path"]) as im:
                ax.imshow(im.copy(),extent=(x,x+w,y+h,y),aspect="auto",interpolation="antialiased",zorder=z)
        elif kind=="line":
            xs,ys=zip(*el["points"])
            ax.plot(xs,ys,color=el["stroke"],lw=el["sw"]*.72,ls=(0,(5,4)) if el["dash"] else "-",zorder=z)
        else:
            attrs=dict(facecolor=el["fill"],edgecolor=el["stroke"],lw=el["sw"]*.72,zorder=z)
            shape=Rectangle((el["x"],el["y"]),el["w"],el["h"],**attrs) if kind=="rect" else Polygon(el["points"],**attrs)
            ax.add_patch(shape)
    plt.rcParams.update({"svg.fonttype":"none","pdf.fonttype":42,"svg.hashsalt":"recovp-plate"})
    for ext in ("png","svg","pdf"):
        fig.savefig(STEM.with_suffix("."+ext),dpi=250,facecolor="white")
    plt.close(fig)

if __name__=="__main__":
    figure()
    export()
