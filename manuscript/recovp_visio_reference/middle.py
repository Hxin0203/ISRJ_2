"""Visio geometry for the complex residual refinement panel."""


def draw(g):
    ink = "#263D4B"
    grid = "#9FB2B5"
    guide = "#577783"
    teal = ("#A7CBCA", "#D0E4E1", "#7A9FA3")
    gold = ("#E9CF90", "#F9E8BD", "#C2A86E")
    blue = ("#96B4C5", "#C4D7E1", "#688CA3")
    rose = ("#DDAE99", "#F4CFBC", "#B28D80")
    red = "#AB6957"

    def arrow(points, color=ink, lw=0.85):
        g.line(points, color, lw, True)

    def segment(points, color=ink, lw=0.65, dashed=False):
        g.line(points, color, lw, False, dashed)

    def card(x, y, w, h, colors, depth=10):
        face, top, side = colors
        d = depth
        g.poly([(x,y),(x+d,y-d),(x+w+d,y-d),(x+w,y)],top,ink,.74)
        g.poly([(x+w,y),(x+w+d,y-d),(x+w+d,y+h-d),(x+w,y+h)],side,ink,.74)
        g.rect(x,y,w,h,face,ink,.74)

    def module(x, y, w, h, name, fill):
        g.rect(x,y,w,h,fill,ink,.7)
        g.text(x+2,y+1,w-4,h-2,name,10,True)

    g.rect(8,240,1184,260,"#F3F7F7",grid,.65)
    g.text(19,242,1035,27,"(b)  Deep complex-valued residual refinement",15,True,"left")
    segment([(20,272),(1180,272)],grid,.55)

    # Two input estimates converge at the normalizer.
    g.text(15,291,82,22,"Stage 1",11,True)
    for y, name in [(323,"û₁"),(374,"ĵ₁")]:
        g.rect(25,y,55,28,"#FFFFFF",grid,.75)
        g.text(25,y+1,55,26,name,14,True)
    arrow([(80,337),(96,337),(96,348),(113,348)])
    arrow([(80,388),(96,388),(96,369),(113,369)])
    module(113,334,96,49,"RMS\nnormalization","#DCE7EB")
    arrow([(209,359),(264,359)])
    g.text(100,389,125,26,"Normalized pair",10)

    # A compact, symmetric U-shaped feature hierarchy.
    g.rect(230,282,539,201,"#FCFDFB",grid,.65)
    g.text(246,284,175,26,"Complex U-Net",13,True,"left")
    g.text(441,286,134,22,"Skip connections",10)
    arrow([(230,359),(264,359)])
    for lx, rx, rung, end in [(279,726,312,334),(336,675,320,344),(390,623,329,354)]:
        arrow([(lx,end),(lx,rung),(rx,rung),(rx,end)],guide,.62)

    # The front-face widths and heights are illustrative rather than channel counts.
    specs=[(264,344,18,67,blue),(322,354,18,51,teal),(377,364,20,37,gold),
           (439,378,108,22,rose),(607,364,20,37,gold),(659,354,18,51,teal),
           (710,344,18,67,blue)]
    for x,y,w,h,c in specs:
        card(x,y,w,h,c)
    for xa, ya, xb, yb in [(292,373,322,373),(350,380,377,380),
                            (407,389,439,389),(557,389,607,389),
                            (637,380,659,380),(687,373,710,373)]:
        arrow([(xa,ya),(xb,yb)])
    g.text(432,379,128,19,"Bottleneck",10,True)
    g.text(431,407,132,23,"Dilated convolutions",10)

    for x,w,caption in [(260,142,"Encoder"),(599,142,"Decoder")]:
        segment([(x,429),(x,435),(x+w,435),(x+w,429)],guide,.6)
        g.text(x,440,w,23,caption,12,True)

    # Proposal and final correction are deliberately labelled at different locations.
    arrow([(738,361),(809,361)])
    g.text(752,330,63,26,"δ̃",14,False)
    g.text(770,306,62,19,"Proposal",9)
    module(809,339,130,44,"Pulse-level\nscalar gate","#E9CCC0")
    arrow([(939,361),(1011,361)])
    g.text(949,331,59,22,"δ₂",13)
    g.ellipse(1011,349,24,24,"#FFFFFF",ink,.85)
    g.text(1011,349,24,24,"−",14,True)
    g.text(984,285,78,23,"û₁",14,True)
    arrow([(1023,310),(1023,349)])
    arrow([(1035,361),(1080,361)])
    module(1080,340,94,42,"Final output\nû","#D6E6E1")

    # The inset is tied to a decoder block with a separate non-directional leader.
    segment([(724,414),(747,427),(798,427)],red,.62,True)
    g.rect(798,402,375,82,"#FCFAF5",red,.72,True)
    g.text(808,405,354,23,"Inside a complex residual block",11.5,True,"left")
    g.rect(813,435,106,27,"#D4E2E7",grid,.65)
    g.text(814,435,104,27,"Complex conv.",9.7,True)
    arrow([(919,449),(945,449)])
    g.rect(945,435,120,27,"#F2DEB4",grid,.65)
    g.text(946,435,118,27,"Magnitude gate",9.7,True)
    arrow([(1065,449),(1106,449)])
    g.ellipse(1106,439,20,20,"#FFFFFF",ink,.8)
    g.text(1106,439,20,20,"+",12,True)
    arrow([(1126,449),(1152,449)])
    arrow([(813,449),(805,449),(805,473),(1116,473),(1116,459)],guide,.65)
