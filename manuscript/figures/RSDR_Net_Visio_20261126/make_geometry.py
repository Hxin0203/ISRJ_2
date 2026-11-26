import json
from pathlib import Path


OUT = Path(__file__).parent
W, H = 1800, 1280
P = {
    "ink": "#243746", "edge": "#526775", "blue": "#5285A3",
    "teal": "#70B0B7", "coral": "#CE7569", "gold": "#E4C169",
    "blue_light": "#DFEAF1", "teal_light": "#DCECEE",
    "coral_light": "#F4E5E2", "gold_light": "#FBF6E8",
    "pale": "#F5F8FA", "gray": "#71838D", "rule": "#C2D0D9",
    "white": "#FFFFFF", "stop": "#3E7C83"
}
items = []


def box(name, group, x, y, w, h, label="", fill=None, line=None, size=15,
        bold=False, color=None, dash=False, weight=0.8, kind="rect", align=1):
    items.append({"kind": kind if not label else "rect", "name": name, "group": group,
                  "x": x, "y": y, "w": w, "h": h, "text": label,
                  "fill": fill or P["white"], "line": line or P["edge"],
                  "weight": weight, "size": size, "bold": bold,
                  "color": color or P["ink"], "align": align, "dash": dash})


def text(name, group, x, y, w, h, label, size=14, bold=False, color=None,
         align=1, italic=False):
    items.append({"kind": "text", "name": name, "group": group, "x": x,
                  "y": y, "w": w, "h": h, "text": label, "size": size,
                  "bold": bold, "italic": italic, "color": color or P["ink"],
                  "align": align})


def path(name, group, points, line=None, arrow=True, dash=False, weight=1.1,
         start=None, end=None):
    items.append({"kind": "path", "name": name, "group": group,
                  "points": points, "line": line or P["ink"], "arrow": arrow,
                  "dash": dash, "weight": weight, "start": start, "end": end})


def panel(group, x, y, w, h, title):
    box("panel_" + group, group, x, y, w, h, fill=P["pale"], line=P["rule"],
        weight=0.8)
    text("title_" + group, group, x + 18, y + 7, w - 36, 32, title,
         size=19, bold=True, align=0)


# (a) End-to-end inference chain.
panel("A_Overall", 20, 18, 1760, 455,
      "(a)  Overall inference architecture of RSDR-Net")
box("a_input", "A_Overall", 48, 202, 132, 74, "Observed I/Q\nY", P["white"],
    size=17, bold=True)
box("a_e1", "A_Overall", 245, 92, 184, 78, "Reliability estimator E₁\nR₁, b₁, T₁", P["teal_light"],
    line=P["teal"], size=14, bold=True)
box("a_e2", "A_Overall", 245, 292, 184, 78, "Reliability estimator E₂\nR₂, b₂, T₂", P["teal_light"],
    line=P["teal"], size=14, bold=True)
box("a_x1", "A_Overall", 486, 92, 236, 78,
    "Expert 1: residual U-Net\nproposal δ₁ + action gate g₁", P["blue_light"],
    line=P["blue"], size=14, bold=True)
box("a_x2", "A_Overall", 486, 292, 236, 78,
    "Expert 2: conditioned U-Net\nproposal δ₂ + bounded gate a₂", P["blue_light"],
    line=P["blue"], size=14, bold=True)
box("a_u1", "A_Overall", 772, 103, 128, 58, "U₁ = Y + g₁δ₁", P["white"],
    line=P["blue"], size=14, bold=True)
box("a_u2", "A_Overall", 772, 303, 128, 58, "U₂ = Y + a₂δ₂", P["white"],
    line=P["blue"], size=14, bold=True)
box("a_anchor", "A_Overall", 974, 165, 174, 86,
    "Fixed anchor\nA = Σ w̄ᵢ Cᵢ\nC₀=Y, C₁=U₁, C₂=U₂", P["gold_light"],
    line=P["gold"], size=13, bold=True)
box("a_router", "A_Overall", 974, 302, 174, 66,
    "3 local routers\nL = π̄₁U₁ + π̄₂U₂", P["coral_light"],
    line=P["coral"], size=14, bold=True)
box("a_gate", "A_Overall", 1215, 218, 168, 86,
    "Group gate\n10 statistics z(Y, A, L)\nρ = 0.25 σ(hψ(z))", P["teal_light"],
    line=P["teal"], size=13, bold=True)
box("a_blend", "A_Overall", 1432, 218, 160, 86,
    "Bounded blend\nF = A + ρ(L − A)", P["gold_light"],
    line=P["gold"], size=14, bold=True)
box("a_out", "A_Overall", 1630, 224, 112, 74, "Restored\noutput F", P["white"],
    line=P["edge"], size=17, bold=True)

# Observation fans out to both pulse-wise branches and remains the KEEP candidate.
path("a_y_to_e1", "A_Overall", [[180, 218], [210, 218], [210, 131], [245, 131]], weight=1.0)
path("a_y_to_e2", "A_Overall", [[180, 260], [210, 260], [210, 331], [245, 331]], weight=1.0)
path("a_y_to_x1", "A_Overall", [[180, 218], [225, 218], [225, 77], [466, 77], [466, 131], [486, 131]], weight=0.9)
path("a_y_to_x2", "A_Overall", [[180, 260], [225, 260], [225, 385], [466, 385], [466, 331], [486, 331]], weight=0.9)
path("a_e1_to_x1", "A_Overall", [[429, 131], [486, 131]], arrow=False,
     line=P["stop"], dash=True, weight=1.1)
path("a_e2_to_x2", "A_Overall", [[429, 331], [486, 331]], arrow=False,
     line=P["stop"], dash=True, weight=1.1)
path("a_x1_to_u1", "A_Overall", [[722, 131], [772, 131]], weight=1.0)
path("a_x2_to_u2", "A_Overall", [[722, 331], [772, 331]], weight=1.0)
# Candidates and KEEP signal into anchor, routed around branch lanes.
path("a_y_keep", "A_Overall", [[180, 238], [194, 238], [194, 438], [944, 438], [944, 223], [974, 223]],
     weight=0.9)
text("a_keep_label", "A_Overall", 708, 411, 92, 22, "KEEP: C₀ = Y", 12,
     color=P["gray"])
path("a_u1_anchor", "A_Overall", [[900, 132], [935, 132], [935, 185], [974, 185]], weight=1.0)
path("a_u2_anchor", "A_Overall", [[900, 332], [935, 332], [935, 232], [974, 232]], weight=1.0)
path("a_u1_router", "A_Overall", [[900, 145], [922, 145], [922, 329], [974, 329]],
     line=P["coral"], weight=0.9)
path("a_u2_router", "A_Overall", [[900, 345], [944, 345], [944, 346], [974, 346]],
     line=P["coral"], weight=0.9)
path("a_anchor_blend", "A_Overall", [[1148, 207], [1432, 239]], line=P["gold"], weight=1.2)
path("a_router_gate", "A_Overall", [[1148, 335], [1180, 335], [1180, 282], [1215, 282]],
     line=P["coral"], weight=1.1)
path("a_gate_blend", "A_Overall", [[1383, 261], [1432, 261]],
     line=P["teal"], weight=1.2)
path("a_blend_out", "A_Overall", [[1592, 261], [1630, 261]], weight=1.2)
path("a_stats", "A_Overall", [[180, 244], [195, 244], [195, 454], [1192, 454], [1192, 303], [1215, 303]],
     line=P["gray"], dash=True, weight=0.8)
text("a_stats_label", "A_Overall", 1065, 434, 160, 19, "frame-level statistics", 11,
     color=P["gray"])
text("a_stop_note", "A_Overall", 42, 425, 340, 19,
     "Dashed: stop-gradient reliability/features or group statistics", 11,
     color=P["stop"], align=0)
text("a_rho_note", "A_Overall", 1416, 314, 352, 30,
     "ρ ∈ [0, 0.25]; fixed-anchor coefficient 1−ρ ≥ 0.75", 12,
     color=P["gray"])

# (b) Expanded estimator and Expert 2, matching the manuscript's detailed panel.
panel("B_Detail", 20, 490, 1170, 770,
      "(b)  Reliability estimator E₂ and proposal network D₂")
text("b_subtitle", "B_Detail", 42, 535, 1115, 24,
     "Per-pulse fast-time processing; E₁ and E₂ have independent parameters", 13,
     color=P["gray"], align=0)
text("b_e_label", "B_Detail", 48, 574, 195, 26,
     "Reliability estimator E₂", 16, bold=True, color=P["teal"], align=0)
box("b_iqy", "B_Detail", 48, 622, 124, 62, "I(Y)\n2 channels", P["white"],
    size=14, bold=True)
box("b_enc0", "B_Detail", 217, 610, 132, 74, "Conv encoder\n32 ch · N=1800", P["teal_light"],
    line=P["teal"], size=13, bold=True)
box("b_enc1", "B_Detail", 382, 610, 132, 74, "Conv encoder\n64 ch · N=900", P["teal_light"],
    line=P["teal"], size=13, bold=True)
box("b_enc2", "B_Detail", 547, 610, 145, 74, "Conv encoder\n128 ch · N=450", P["teal_light"],
    line=P["teal"], size=13, bold=True)
box("b_trans", "B_Detail", 726, 600, 184, 94,
    "Pre-LN Transformer\n4 layers · 4 heads\nRoPE", P["gold_light"],
    line=P["gold"], size=13, bold=True)
box("b_dec", "B_Detail", 941, 610, 138, 74, "Skip decoder\n3 scales", P["teal_light"],
    line=P["teal"], size=13, bold=True)
box("b_eout", "B_Detail", 1095, 592, 78, 110, "R₂, b₂\nT₂,₀:₂", P["white"],
    line=P["teal"], size=13, bold=True)
for nm, pts, col in [
    ("b_e0", [[172,653],[217,653]], P["ink"]),
    ("b_e1", [[349,647],[382,647]], P["ink"]),
    ("b_e2", [[514,647],[547,647]], P["ink"]),
    ("b_e3", [[692,647],[726,647]], P["ink"]),
    ("b_e4", [[910,647],[941,647]], P["ink"]),
    ("b_e5", [[1079,647],[1095,647]], P["ink"]),
]: path(nm, "B_Detail", pts, line=col, weight=1.0)
text("b_skip_lbl", "B_Detail", 377, 716, 594, 18,
     "Encoder features connect to decoder at matching resolutions (skip connections)",
     11, color=P["gray"])
for nm, pts in [
    ("b_skip0", [[283,684],[283,692],[1002,692],[1002,684]]),
    ("b_skip1", [[448,684],[448,701],[1010,701],[1010,684]]),
    ("b_skip2", [[619,684],[619,710],[1018,710],[1018,684]])
]: path(nm, "B_Detail", pts, line=P["gray"], weight=0.75)

text("b_d_label", "B_Detail", 48, 752, 350, 26,
     "Expert 2 proposal network D₂", 16, bold=True, color=P["blue"], align=0)
box("b_di", "B_Detail", 48, 814, 124, 62, "I(Y)\n2 channels", P["white"],
    size=14, bold=True)
box("b_denc", "B_Detail", 217, 803, 176, 84,
    "Residual U-Net encoder\n32 → 64 → 128 ch\n↓2 at each scale", P["blue_light"],
    line=P["blue"], size=13, bold=True)
box("b_dmod", "B_Detail", 429, 791, 184, 108,
    "3 reliability modulation cells\nH̃c = Hc + (1−rc)γc vc(Hc)\nfeatures: 32 / 64 / 128", P["coral_light"],
    line=P["coral"], size=12, bold=True)
box("b_ddec", "B_Detail", 649, 803, 176, 84,
    "Skip decoder\n128 → 64 → 32 ch\n↑2 linear resize", P["blue_light"],
    line=P["blue"], size=13, bold=True)
box("b_delta", "B_Detail", 864, 814, 122, 62, "Complex proposal\nδ₂", P["white"],
    line=P["blue"], size=13, bold=True)
box("b_gate", "B_Detail", 1016, 799, 156, 94,
    "Action gate Gω₂\n[ I(Y), I(δ₂), R̃₂ ]\na₂=(1−R̃₂)g₂", P["gold_light"],
    line=P["gold"], size=12, bold=True)
path("b_d0", "B_Detail", [[172,845],[217,845]], weight=1.0)
path("b_d1", "B_Detail", [[393,845],[429,845]], weight=1.0)
path("b_d2", "B_Detail", [[613,845],[649,845]], weight=1.0)
path("b_d3", "B_Detail", [[825,845],[864,845]], weight=1.0)
path("b_d4", "B_Detail", [[986,845],[1016,845]], weight=1.0)
path("b_r_to_d", "B_Detail", [[1134,702],[1134,744],[522,744],[522,791]],
     line=P["stop"], dash=True, weight=1.1)
path("b_t_to_d", "B_Detail", [[1120,702],[1120,733],[610,733],[610,773],[522,773],[522,791]],
     line=P["stop"], dash=True, weight=1.0)
text("b_stop", "B_Detail", 758, 752, 210, 19,
     "R̃₂=sg(R₂), T̃₂=sg(T₂)", 11, color=P["stop"])
box("b_u2", "B_Detail", 1016, 934, 156, 62,
    "U₂ = Y + a₂δ₂", P["white"], line=P["blue"], size=14, bold=True)
path("b_gate_to_u", "B_Detail", [[1094,893],[1094,934]], weight=1.0)
path("b_y_to_u", "B_Detail", [[110,876],[110,965],[1016,965]],
     line=P["gray"], dash=True, weight=0.9)
text("b_d1_compare", "B_Detail", 48, 1030, 1100, 52,
     "Expert 1 uses the same residual U-Net backbone without reliability conditioning or modulation; its action gate uses R̃₁ only at the output.",
     13, color=P["gray"], align=0)
text("b_infer_note", "B_Detail", 48, 1102, 1100, 54,
     "Estimators, experts and local routers process each pulse independently. Only the group gate aggregates the complete P × N frame.",
     13, color=P["ink"], align=0)
text("b_train_only", "B_Detail", 48, 1183, 1090, 34,
     "Training-only auxiliary output: boundary map b₂; it is not an inference input.",
     12, italic=True, color=P["gray"], align=0)

# (c) Modulation cell and reliability-bounded action.
panel("C_Modulation", 1210, 490, 570, 770,
      "(c)  Reliability modulation cell")
text("c_intro", "C_Modulation", 1235, 540, 520, 52,
     "At each encoder–decoder scale c, resize the stopped reliability and map the estimator feature:",
     13, color=P["ink"], align=0)
box("c_h", "C_Modulation", 1250, 625, 118, 62, "Encoder feature\nHc", P["blue_light"],
    line=P["blue"], size=14, bold=True)
box("c_conv", "C_Modulation", 1451, 615, 190, 82,
    "Conv₁ [ Hc, rc, ηc(T̃₂,c) ]\nSigmoid", P["teal_light"],
    line=P["teal"], size=13, bold=True)
box("c_gamma", "C_Modulation", 1655, 625, 94, 62, "Gate\nγc", P["white"],
    line=P["teal"], size=14, bold=True)
path("c_h_to_conv", "C_Modulation", [[1368,656],[1451,656]], weight=1.0)
path("c_conv_to_g", "C_Modulation", [[1641,656],[1655,656]], weight=1.0)
box("c_rc", "C_Modulation", 1248, 744, 142, 64, "r_c = Resize(R̃₂)\n1 − r_c", P["coral_light"],
    line=P["coral"], size=13, bold=True)
box("c_v", "C_Modulation", 1450, 744, 190, 64, "v_c(Hc)\nConv₃ + GELU", P["blue_light"],
    line=P["blue"], size=13, bold=True)
box("c_sum", "C_Modulation", 1510, 864, 154, 66,
    "H̃c = Hc +\n(1−r_c) ⊙ γc ⊙ v_c(Hc)", P["gold_light"],
    line=P["gold"], size=12, bold=True)
path("c_rc_gate", "C_Modulation", [[1390,776],[1415,776],[1415,708],[1698,708],[1698,687]],
     line=P["coral"], weight=0.9)
path("c_rc_sum", "C_Modulation", [[1320,808],[1320,896],[1510,896]], line=P["coral"], weight=0.9)
path("c_v_sum", "C_Modulation", [[1545,808],[1545,864]], line=P["blue"], weight=0.9)
path("c_gamma_sum", "C_Modulation", [[1702,687],[1702,896],[1664,896]], line=P["teal"], weight=0.9)
path("c_h_sum", "C_Modulation", [[1309,687],[1309,842],[1485,842],[1485,896],[1510,896]],
     line=P["blue"], weight=0.9)
text("c_behaviour", "C_Modulation", 1240, 965, 510, 72,
     "R̃₂ ≈ 1  →  little feature change\nLower R̃₂  →  larger permitted modulation",
     14, bold=True, color=P["ink"], align=0)
box("c_adopt", "C_Modulation", 1240, 1070, 510, 108,
    "Output action gate\ng₂ = σ(Gω₂[ I(Y), I(δ₂), R̃₂ ])\na₂ = (1 − R̃₂) ⊙ g₂  ∈ [0, 1 − R̃₂]", P["gold_light"],
    line=P["gold"], size=14, bold=True)

cfg = {
    "width": W, "height": H, "page_width_in": 18, "page_height_in": 12.8,
    "font": "Times New Roman", "palette": P, "items": items,
    "grouping_order": ["A_Overall", "B_Detail", "C_Modulation"],
    "source_documents": [
        {"path": "G:/Files/Markdown/ISRJ/草稿/终稿1124/修订稿_CVEDNet_20261126_第二轮/论文终稿_英文版.md",
         "sha256": "80FAD871CA3257165A2E8A2873EC187B82834CB494E11E914B478A43909DCF3A"},
        {"path": "G:/Files/Markdown/ISRJ/草稿/终稿1124/修订稿_CVEDNet_20261126_第二轮/论文终稿_中文版 - 副本.md",
         "sha256": "EC76F6CA779B883F2FA7D98065CD1C327B53791A62C4973F1E4BAF9FAFD65CE6"}
    ]
}
(OUT / "figure_geometry.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(items)} native geometry objects to {OUT / 'figure_geometry.json'}")
