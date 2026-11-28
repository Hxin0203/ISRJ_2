"""Compose editable Visio drawing primitives in a 1200 by 656 reference grid."""
from pathlib import Path
import json
import math

from lower import draw as training_band
from middle import draw as refinement_band

INK = "#263D4B"
TEAL = "#C8DEDA"
GOLD = "#EFD9A7"
SALMON = "#EDCDC0"
RULE = "#9AACAE"


class Geometry:
    def __init__(self):
        self.items = []

    def rect(self, x, y, w, h, fill, stroke, lw=0.8, dashed=False):
        self.items.append(dict(kind="rect", x=x, y=y, w=w, h=h, fill=fill,
                               stroke=stroke, lw=lw, dashed=dashed))

    def ellipse(self, x, y, w, h, fill, stroke, lw=0.8):
        self.items.append(dict(kind="ellipse",x=x,y=y,w=w,h=h,
                               fill=fill,stroke=stroke,lw=lw))

    def text(self, x, y, w, h, text, size=10, bold=False, align="center"):
        self.items.append(dict(kind="text",x=x,y=y,w=w,h=h,text=text,size=size,
                               bold=bold,align=align))

    def line(self, points, stroke=INK, lw=1, arrow=True, dashed=False):
        self.items.append(dict(kind="line",points=points,stroke=stroke,
                               lw=lw,arrow=arrow,dashed=dashed,fill="none"))

    def poly(self, points, fill, stroke, lw=0.8):
        self.items.append(dict(kind="poly",points=points,fill=fill,stroke=stroke,lw=lw))


def trace(g,x,y,w,h,fn,colour="#3E7579"):
    g.line([(x+i*w/120, y+h*fn(i/120)) for i in range(121)],colour,.78,False)


def projection_band(g):
    g.rect(8,8,1184,224,"#F4F8F7",RULE)
    g.text(20,12,1125,27,"(a)  Model-driven waveform projection and evidence gating",15,True,"left")
    g.line([(20,43),(1179,43)],RULE,.55,False)
    g.text(20,55,112,24,"Observed I/Q",11,True)
    g.rect(20,84,111,60,"#FFFFFF",RULE)
    trace(g,25,114,101,20,lambda t:math.exp(-11*(t-.5)**2)*math.sin(t*92+13*t*t))
    g.text(20,147,111,21,"Input  y",10)
    g.line([(131,115),(156,115)])
    g.rect(158,58,240,127,TEAL,INK)
    g.text(166,63,224,24,"Matched-filter evidence",12,True)
    for x in (170,285):
        g.rect(x,93,101,59,"#FFFFFF",RULE,.55)
    trace(g,174,124,93,19,lambda t:math.exp(-30*(t-.48)**2)*math.sin(130*t))
    trace(g,290,143,90,-43,lambda t:.12+.78*math.exp(-320*(t-.49)**2)+.28*math.exp(-260*(t-.76)**2))
    g.text(169,157,103,18,"I/Q correlation",9.5)
    g.text(282,157,107,18,"Candidate peaks",9.5)
    g.line([(398,116),(429,116)])

    g.rect(431,58,154,127,"#E3EEEB",INK)
    g.text(433,63,151,24,"Local dictionary",12,True)
    palette=["#ADCBC7","#D3DFD7","#9DBAC6","#EDD9A8","#B0C7BD"]
    for j in range(4):
        for i in range(8):
            g.rect(450+i*14.5,94+j*13.8,14.2,13.5,palette[(i*3+j*(i+1))%5],"#FFFFFF",.30)
    g.text(438,155,139,22,"Dictionary atoms  B",9.8)
    g.rect(620,60,312,124,"#FAF8F1","#BA9D7B",.78,True)
    g.line([(373,58),(373,50),(704,50),(704,74)],"#956755",.72,True,True)
    g.rect(639,74,131,32,SALMON,INK)
    g.text(640,74,129,32,"Evidence gate",11,True)
    g.rect(783,117,135,46,GOLD,INK)
    g.text(785,119,131,42,"Variable\nprojection",12,True)
    g.line([(585,139),(782,139)])
    g.line([(770,90),(849,90),(849,116)],"#956755",.85,True,True)
    g.line([(918,140),(950,140),(950,89),(987,89)])
    g.rect(988,62,184,52,"#E7EEEF",INK,.65)
    g.text(990,63,180,49,"Estimated interference\nĵ₁",11,True)
    g.line([(1080,114),(1080,142)])
    g.ellipse(1068,144,24,24,"#FFFFFF",INK,.85)
    g.text(1068,143,24,24,"−",14,True)
    g.line([(1080,168),(1080,179)])
    g.rect(987,181,185,37,TEAL,INK,.65)
    g.text(989,181,181,37,"Initial recovery   û₁",11,True)
    g.line([(73,169),(73,203),(957,203),(957,156),(1067,156)],INK,.82)
    g.text(410,205,182,18,"Observation bypass",9)


if __name__ == "__main__":
    geometry = Geometry()
    projection_band(geometry)
    refinement_band(geometry)
    training_band(geometry)
    Path(__file__).with_name("geometry.json").write_text(
        json.dumps(geometry.items, ensure_ascii=False), encoding="utf-8")
