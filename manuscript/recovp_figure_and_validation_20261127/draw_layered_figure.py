"""Draw the three-panel manuscript figure at a two-column print width."""
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import numpy as np

mpl.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix",
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
W, H = 720, 438
fig = plt.figure(figsize=(183/25.4, 183/25.4*H/W))
ax = fig.add_axes([0, 0, 1, 1], xlim=(0,W), ylim=(H,0))
ax.axis("off")
ink, muted, rule = "#203a48", "#496774", "#a2b6bc"
jade, ochre, blue, rose = "#c4deda", "#efdaa6", "#bed1df", "#e8c8b7"
scale = fig.get_figwidth()*72/W


def label(x,y,s,size=12,weight="normal",ha="center",color=ink):
    ax.text(x,y,s,ha=ha,va="center",fontsize=size*scale,weight=weight,
            color=color,zorder=8)


def rect(x,y,w,h,fill="white",edge=ink,lw=.7,ls="-"):
    ax.add_patch(Rectangle((x,y),w,h,fc=fill,ec=edge,lw=lw,ls=ls))


def path(points,color=ink,lw=.85,arrow=True,ls="-"):
    pts=np.array(points)
    ax.plot(*pts.T,color=color,lw=lw,ls=ls,zorder=4,solid_capstyle="butt")
    if arrow:
        d=pts[-1]-pts[-2]
        d=d/np.linalg.norm(d)
        side=np.array([-d[1],d[0]])*2
        end=pts[-1]
        ax.add_patch(Polygon([end,end-4.2*d+side,end-4.2*d-side],
                             fc=color,ec="none",zorder=5))


def panel(y,h,title,sub,fill):
    rect(9,y,702,h,fill,rule,.55)
    label(20,y+14,title,15,"bold","left")
    label(699,y+14,sub,10.5,ha="right",color=muted)
    path([(20,y+26),(699,y+26)],rule,.45,False)


def stack(x,y,w,h,face):
    for dx,dy in ((6,-4),(3,-2),(0,0)):
        rect(x+dx,y+dy,w,h,face,ink,.6)


panel(6,133,"(a)  From observation to initial recovery","Structured estimation","#f6f9f8")
panel(152,159,"(b)  From initial recovery to final output","Complex residual refinement","#f5f8fa")
panel(324,106,"(c)  What the training objective measures","Training only","#fcfaf5")
# A drawn waveform and dictionary are pictograms, not output from an experiment.
t=np.linspace(-1,1,190)
trace=(np.exp(-4*t*t)*np.sin(40*t+11*t*t)+.2*np.cos(30*t))
rect(22,69,85,32,"white",rule)
ax.plot(26+(t+1)*38,85-trace*10,color="#497d80",lw=.65)
label(64,57,"Observed I/Q",12,"bold")
label(53,115,r"$\mathbf{y}$",15)
path([(107,85),(139,85)])
stack(141,69,51,33,jade)
for i in range(3):
    ax.plot(146+(t+1)*20,77+8*i-2*np.sin((6+i*3)*t)*np.exp(-2*t*t),
            color="#497d80",lw=.6)
label(169,114,"Local dictionary",11.5)
path([(198,85),(252,85)])
rect(252,66,102,38,ochre)
label(303,85,"Variable projection",11.1,"bold")
rect(257,40,92,18,rose)
label(303,49,"Evidence gate",11)
path([(303,58),(303,66)],muted,.65)
path([(354,85),(397,85)])
rect(397,62,103,46,"white",rule)
label(448,75,r"$\widehat{\mathbf{j}}_1$",16)
label(448,95,"Estimated interference",10.4)
path([(500,85),(543,85)])
ax.add_patch(Circle((552,85),9,fc="white",ec=ink,lw=.8,zorder=5))
label(552,85,"−",18)
path([(65,101),(65,127),(552,127),(552,94)],muted,.65)
label(552,57,r"$\mathbf{y}-\widehat{\mathbf{j}}_1$",13)
path([(561,85),(592,85)])
rect(592,62,104,46,"#deebe8",rule)
label(644,75,r"$\widehat{\mathbf{u}}_1$",16)
label(644,95,"Initial recovery",11)

# The magnified network is contained within the refinement panel.
label(68,214,"Stage-one pair",12,"bold")
label(68,234,r"$(\widehat{\mathbf{u}}_1,\widehat{\mathbf{j}}_1)$",16)
path([(112,233),(151,233)])
rect(137,192,322,104,"none","#ac6655",.65,(0,(4,3)))
label(147,202,"Complex U-Net",12.8,"bold","left")
for x,y,w,h,c in [(151,220,16,32,blue),(184,225,18,27,jade),(223,230,20,22,ochre),
                     (344,230,20,22,ochre),(383,225,18,27,jade),(418,220,16,32,blue)]:
    stack(x,y,w,h,c)
for x1,x2 in [(173,184),(208,223),(370,383),(407,418)]:
    path([(x1,238),(x2,238)],lw=.7)
path([(249,239),(264,239),(264,253),(274,253)],lw=.7)
rect(274,243,49,20,rose)
path([(323,253),(332,253),(332,239),(344,239)],lw=.7)
label(299,275,"Dilated bottleneck",11)
label(186,275,"Encoder",11.5,"bold")
label(404,275,"Decoder",11.5,"bold")
for left,right,y in [(160,426,208),(192,391,213),(232,353,218)]:
    top={160:216,192:221,232:226}[left]
    path([(left,top),(left,y),(right,y),(right,top)],muted,.48)
path([(440,234),(499,234)])
label(477,217,r"$\widetilde{\delta}$",15)
label(478,251,"Proposal",9.8)
rect(499,219,55,30,rose)
label(526,234,"Gate",12,"bold")
path([(554,234),(598,234)])
label(576,220,r"$\delta_2$",13.5)
ax.add_patch(Circle((608,234),9,fc="white",ec=ink,lw=.8,zorder=5))
label(608,234,"−",18)
path([(608,202),(608,225)],lw=.75)
label(608,194,r"$\widehat{\mathbf{u}}_1$",14)
path([(617,234),(642,234)])
label(671,231,r"$\widehat{\mathbf{u}}$",20)
label(671,253,"Final output",11.5,"bold")

# Four complementary training criteria with a non-directional state key.
for x in (175,347,544):
    path([(x,365),(x,417)],rule,.52,False)
for x,head,sym in [(87,"Waveform",r"$\mathcal{L}_{\rm wav}$"),
                   (258,"Pulse compression",r"$\mathcal{L}_{\rm pc}$"),
                   (624,"Residual tail",r"$\mathcal{L}_{\rm tail}$")]:
    label(x,381,head,13,"bold")
    label(x,404,sym,16)
for i,(lab,c) in enumerate([(r"$y_{00}$","white"),(r"$y_{10}$",jade),
                            (r"$y_{01}$",ochre),(r"$y_{11}$",blue)]):
    x,y=359+(i%2)*26,371+(i//2)*19
    rect(x,y,24,17,c,rule,.55)
    label(x+12,y+8.5,lab,11)
label(473,381,"Target damage",13,"bold")
label(473,404,r"$\mathcal{L}_{\rm tar}$",16)
stem=Path(__file__).with_name("ReCoVP_Hybrid_layered")
for ext in (".svg",".pdf",".png"):
    fig.savefig(stem.with_suffix(ext),dpi=300,facecolor="white")
