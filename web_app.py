"""
Đớp Tool — Tối ưu hoá kết cấu dàn/khung thép bằng Phép tính Biến phân
Chạy:  streamlit run web_app.py
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from fem_core import build_docx_bytes
import viz_results as vz
from opt_core import (
    BEAM_BCS, COLUMN_BCS, STEEL_E, STEEL_RHO, STEEL_SIGMA,
    BeamProblem, StructInput, make_family, optimize_beam, optimize_column, optimize_frame,
    ten_bar_truss_example, two_bar_kkt_solution, weight_to_mass_kg,
)
from section_3d import section_3d_comparison, section_3d_note
from dynamic_core import free_vibration, frame_story_stiffness, inelastic_mdof_response, mdof_response, newmark_response, rc_circular_column_properties, rc_section_properties, response_spectrum
from qcvn02_lookup import lookup as qcvn02_lookup, places as qcvn02_places, provinces as qcvn02_provinces
from seismic_methods import TYPE1_GROUND_PARAMETERS, equivalent_lateral_force, modal_properties, modal_response_spectrum, tcvn9386_type1_elastic_spectrum
from opensees_ui import render_earthquake_tab
try:
    from toi_uu_gian_10_thanh_GA_v2 import (
        ThamSoGA,
        khoi_luong as ga_khoi_luong,
        phan_tich_gian as ga_phan_tich_gian,
        thuat_toan_di_truyen,
        toi_uu_gradient,
    )
    GA_IMPORT_ERROR = None
except Exception:
    GA_IMPORT_ERROR = None
    try:
        raise
    except Exception as ex:
        GA_IMPORT_ERROR = ex
    ThamSoGA = None
    ga_khoi_luong = ga_phan_tich_gian = thuat_toan_di_truyen = toi_uu_gradient = None
st.set_page_config(page_title="Đớp Tool", page_icon="🦈", layout="wide")

import opt_core as _oc
if "user_nodal" not in getattr(_oc.StructInput, "__dataclass_fields__", {}):
    st.error("File `opt_core.py` trên máy bạn là phiên bản CŨ (thiếu trường `user_nodal`). "
             "Hãy giải nén lại toàn bộ `dop_tool.zip` và ghi đè TẤT CẢ file (opt_core.py, web_app.py, "
             "legacy_plots.py, viz_results.py, fem_core.py), xoá thư mục `__pycache__`, rồi chạy lại Streamlit.")
    st.stop()

HERE = os.path.dirname(os.path.abspath(__file__))
BLUE, ORANGE, GREEN, RED, GREY = "#1f77b4", "#e67300", "#2ca02c", "#d62728", "#7f7f7f"
KIP_TO_KN = 4.4482216152605
LB_TO_KN = KIP_TO_KN / 1000.0
KN_TO_LB = 1.0 / LB_TO_KN
TEN_BAR_GA_TO_APP_ORDER = np.array([0, 1, 2, 3, 4, 5, 7, 6, 9, 8], dtype=int)


# ══════════════════════════════════════════════════════
#  CSS & tiện ích
# ══════════════════════════════════════════════════════
def inject_css():
    """Theme-aware CSS for Streamlit light/dark themes."""
    try:
        theme_type = getattr(getattr(st.context, "theme", None), "type", "light")
    except Exception:
        theme_type = "light"

    dark = theme_type == "dark"
    if dark:
        paper, ink, muted = "#0e1117", "#f1f5f9", "#aab4c0"
        green, green_soft, rule = "#69c08a", "#173526", "#303844"
        card, input_bg = "#151922", "#11151c"
        good, bad = "#73d69a", "#ff8d86"
    else:
        paper, ink, muted = "#f4f6f3", "#202a25", "#66736c"
        green, green_soft, rule = "#285c45", "#e7efe9", "#cfd8d2"
        card, input_bg = "#ffffff", "#ffffff"
        good, bad = "#326746", "#b64237"

    st.markdown(f"""
    <style>
    #MainMenu, footer {{display:none !important;}}

    :root {{
        --course-paper:{paper}; --course-ink:{ink}; --course-muted:{muted};
        --course-green:{green}; --course-green-soft:{green_soft};
        --course-rule:{rule}; --course-card:{card}; --course-input:{input_bg};
        --course-good:{good}; --course-bad:{bad};
    }}

    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background:var(--course-paper) !important;
        color:var(--course-ink) !important;
    }}
    .block-container {{padding-top:1rem; max-width:1520px;}}

    h1,h2,h3,h4,h5,h6,.stMarkdown,p,label {{
        color:var(--course-ink);
    }}
    h1,h2,h3 {{font-family:Cambria,Georgia,serif; letter-spacing:0;}}
    h1 {{font-size:2.05rem;}}

    .course-kicker {{
        display:flex; align-items:center; gap:.65rem; margin:.25rem 0 .45rem;
        color:var(--course-muted); font-size:.76rem; font-weight:650;
    }}
    .course-code {{
        padding:.24rem .46rem; border-radius:3px;
        background:var(--course-green); color:#fff;
    }}

    .metric-strip {{
        display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
        gap:10px; margin:.35rem 0 .9rem;
    }}
    .metric-card {{
        border:1px solid var(--course-rule); border-radius:6px;
        padding:10px 12px; background:var(--course-card);
        color:var(--course-ink); box-shadow:none;
    }}
    .metric-label {{color:var(--course-muted); font-size:.74rem; margin-bottom:4px; text-transform:uppercase;}}
    .metric-value {{font-weight:700; font-size:1.12rem; color:var(--course-ink);}}
    .metric-good .metric-value {{color:var(--course-good);}}
    .metric-bad .metric-value {{color:var(--course-bad);}}

    .stTabs [data-baseweb="tab-list"] {{
        gap:.15rem; background:transparent; border-bottom:1px solid var(--course-rule);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius:5px 5px 0 0; padding:.62rem .85rem; color:var(--course-muted);
    }}
    .stTabs [data-baseweb="tab"] p {{margin:0; font-weight:600;}}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        background:var(--course-green-soft); color:var(--course-green);
    }}
    .stTabs [data-baseweb="tab-highlight"] {{background:var(--course-green);}}

    .report-shell {{
        border:1px solid var(--course-rule); border-radius:6px;
        padding:.5rem .65rem; background:var(--course-card);
        color:var(--course-ink);
    }}
    .section-callout {{
        border-left:3px solid var(--course-green);
        background:var(--course-green-soft); padding:.8rem 1rem;
        border-radius:4px; margin:.6rem 0 1rem; color:var(--course-ink);
    }}
    .theory-limit {{
        border-left:4px solid var(--course-green); border-radius:6px;
        background:var(--course-green-soft); color:var(--course-ink);
        padding:.85rem 1rem; margin:.7rem 0;
    }}
    .example-grid {{
        display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr));
        gap:.9rem; margin-top:.8rem;
    }}
    .example-card {{
        border:1px solid var(--course-rule); border-radius:6px;
        padding:.9rem 1rem; background:var(--course-card);
        color:var(--course-ink);
    }}
    .example-card h4 {{margin:0 0 .45rem; font-size:1rem; color:var(--course-ink);}}
    .example-card p {{margin:0; color:var(--course-muted); font-size:.9rem;}}
    .example-card .chip {{
        display:inline-block; margin-top:.6rem; background:var(--course-green-soft);
        color:var(--course-green); border-radius:3px; padding:.2rem .6rem;
        font-size:.74rem; font-weight:600;
    }}

    /* Streamlit/BaseWeb controls now follow the same palette. */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {{
        background:var(--course-input) !important;
        color:var(--course-ink) !important;
        border-color:var(--course-rule) !important;
    }}
    [data-baseweb="select"] span,
    [data-baseweb="select"] input,
    [data-baseweb="input"] input {{
        color:var(--course-ink) !important;
    }}
    [role="listbox"], [role="option"] {{
        background:var(--course-card) !important;
        color:var(--course-ink) !important;
    }}
    [role="option"]:hover {{background:var(--course-green-soft) !important;}}

    textarea {{
        font-family:Consolas,"Courier New",monospace !important;
        border-radius:4px !important;
    }}
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label {{
        color:var(--course-ink) !important;
    }}
    [data-testid="stCaptionContainer"] p {{
        color:var(--course-muted) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


def metrics(items):
    html = '<div class="metric-strip">'
    for it in items:
        label, value = it[0], it[1]
        cls = it[2] if len(it) > 2 else ""
        html += f'<div class="metric-card {cls}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)


def clean_rows(df, ncols, int_cols=()):
    out = []
    if df is None:
        return out
    for _, row in df.iterrows():
        vals = list(row.values[:ncols])
        if any(pd.isna(v) for v in vals):
            continue
        out.append(tuple(int(v) if k in int_cols else float(v) for k, v in enumerate(vals)))
    return out


def section_figure(fam, area: float, state_label: str = "", view_extent: float | None = None) -> go.Figure:
    dimensions = fam.cross_section_dimensions(area)
    fig = go.Figure()
    try:
        _dark = getattr(getattr(st.context, "theme", None), "type", "light") == "dark"
    except Exception:
        _dark = False
    fill = "#263b52" if _dark else "#dcecff"
    edge = "#7db7ff" if _dark else "#1f5f99"
    dim_color = "#b8c4d1" if _dark else "#485766"
    dim_text = "#f1f5f9" if _dark else "#202a35"
    hollow_fill = "#0e1117" if _dark else "#ffffff"

    def dimension_line(x1, y1, x2, y2, text):
        fig.add_annotation(x=x2, y=y2, ax=x1, ay=y1, xref="x", yref="y", axref="x", ayref="y",
                           text=text, showarrow=True, arrowhead=3, startarrowhead=3, arrowsize=1,
                           arrowwidth=1.2, arrowcolor="#485766", font=dict(size=11, color=dim_text),
                           bgcolor=("rgba(21,25,34,0.94)" if _dark else "rgba(255,255,255,0.9)"))

    if fam.shape_kind == "rect":
        width, height = dimensions["b"], dimensions["h"]
        fig.add_shape(type="rect", x0=-width / 2, x1=width / 2, y0=-height / 2, y1=height / 2,
                      fillcolor=fill, line=dict(color=edge, width=2))
        dimension_line(-width / 2, -height * 0.62, width / 2, -height * 0.62,
                       f"b = {width * 1000:.1f} mm")
        dimension_line(-width * 0.72, -height / 2, -width * 0.72, height / 2,
                       f"h = {height * 1000:.1f} mm")
        extent = max(width, height)
    elif fam.shape_kind in ("circle", "tube"):
        diameter = dimensions["D"]
        radius = diameter / 2
        fig.add_shape(type="circle", x0=-radius, x1=radius, y0=-radius, y1=radius,
                      fillcolor=fill, line=dict(color=edge, width=2))
        if fam.shape_kind == "tube":
            inner_radius = radius - dimensions["t"]
            fig.add_shape(type="circle", x0=-inner_radius, x1=inner_radius,
                          y0=-inner_radius, y1=inner_radius,
                          fillcolor=hollow_fill, line=dict(color=edge, width=1.5))
        dimension_line(-radius, -radius * 1.35, radius, -radius * 1.35,
                       f"D = {diameter * 1000:.1f} mm")
        extent = diameter
    elif fam.shape_kind == "ibeam":
        height, width, thickness = dimensions["h"], dimensions["b_f"], dimensions["t_f"]
        for center_y in (-height / 2 + thickness / 2, height / 2 - thickness / 2):
            fig.add_shape(type="rect", x0=-width / 2, x1=width / 2,
                          y0=center_y - thickness / 2, y1=center_y + thickness / 2,
                          fillcolor=fill, line=dict(color=edge, width=2))
        fig.add_shape(type="line", x0=0, x1=0, y0=-height / 2 + thickness, y1=height / 2 - thickness,
                      line=dict(color=dim_color, width=1.2, dash="dash"))
        dimension_line(-width / 2, height * 0.68, width / 2, height * 0.68,
                       f"b_f = {width * 1000:.1f} mm")
        dimension_line(-width * 0.68, -height / 2, -width * 0.68, height / 2,
                       f"h = {height * 1000:.1f} mm")
        extent = max(width, height)
    else:
        extent = 1.0
        fig.add_shape(type="rect", x0=-0.35, x1=0.35, y0=-0.35, y1=0.35,
                      fillcolor="rgba(220,236,255,0.35)", line=dict(color="#778899", width=2, dash="dash"))
        fig.add_annotation(x=0, y=0, text="Chưa khai báo hình học", showarrow=False,
                           font=dict(size=12, color=dim_color))

    padding = max(extent, view_extent or 0.0) * 0.85
    title = f"Mặt cắt ngang · A = {area * 1e4:.2f} cm²"
    if state_label:
        title = f"{state_label} · A = {area * 1e4:.2f} cm²"
    fig.update_layout(title=dict(text=title, x=0.5, xanchor="center",
                                 font=dict(size=14)),
                      height=280, margin=dict(l=18, r=18, t=42, b=18),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      showlegend=False)
    fig.update_xaxes(range=[-padding, padding], visible=False, fixedrange=True, constrain="domain")
    fig.update_yaxes(range=[-padding, padding], visible=False, fixedrange=True, scaleanchor="x", scaleratio=1)
    return fig


def show_section_preview(fam, area: float, key: str, state_label: str = "",
                         view_extent: float | None = None) -> None:
    dimensions = fam.cross_section_dimensions(area)
    st.plotly_chart(section_figure(fam, area, state_label, view_extent), width="stretch",
                    config={"displayModeBar": False}, key=key)
    if fam.shape_kind == "rect":
        ratio = fam.shape_parameters[0]
        shape_note = f"Chữ nhật đặc: b/h = {ratio:g}; b = {dimensions['b'] * 1000:.1f} mm, h = {dimensions['h'] * 1000:.1f} mm."
    elif fam.shape_kind == "circle":
        shape_note = f"Tròn đặc: D = {dimensions['D'] * 1000:.1f} mm."
    elif fam.shape_kind == "tube":
        shape_note = (f"Ống thành mỏng: D = {dimensions['D'] * 1000:.1f} mm, "
                      f"t = {dimensions['t'] * 1000:.1f} mm; A ≈ πDt.")
    elif fam.shape_kind == "ibeam":
        shape_note = (f"Chữ I lý tưởng hóa: h = {dimensions['h'] * 1000:.1f} mm cố định; "
                      "A là tổng diện tích hai cánh, bụng bỏ qua; bề rộng cánh chỉ minh họa.")
    else:
        shape_note = "Họ tuỳ chỉnh chưa xác định hình học vật lý của tiết diện."
    st.caption(f"{shape_note} Mô hình: I = αAⁿ (n = {fam.n:g}); W = βAⁿ⁻ᵞ (γ = {fam.gamma:g}).")


def _section_extent(fam, area: float) -> float:
    if fam.shape_kind == "custom":
        return 1.0
    return max(fam.cross_section_dimensions(area).values())


def show_section_comparison(fam, area_before: float, area_after: float, key: str) -> None:
    view_extent = max(_section_extent(fam, area_before), _section_extent(fam, area_after))
    before, after = st.columns(2)
    with before:
        show_section_preview(fam, area_before, f"{key}_before", "Trước", view_extent)
    with after:
        show_section_preview(fam, area_after, f"{key}_after", "Sau", view_extent)


def show_section_3d(fam, length: float, area_before, area_after, key: str,
                    orientation: str = "horizontal") -> None:
    st.plotly_chart(section_3d_comparison(fam, length, area_before, area_after, orientation), width="stretch",
                    config={"displayModeBar": False}, key=key)
    axis_note = "Chiều dài cột dựng theo phương đứng." if orientation == "vertical" else \
        "Chiều dài cấu kiện dựng theo phương ngang."
    st.caption(section_3d_note(fam) + f" {axis_note} Màu xám = trước, xanh = sau tối ưu.")


def report_block(text: str, key: str, title: str):
    with st.expander("📄 Thuyết minh tính toán", expanded=False):
        st.text_area("Nội dung", text, height=420, key=f"ta_{key}", label_visibility="collapsed")
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tải .txt", text.encode("utf-8"), file_name=f"{key}.txt", key=f"dl_txt_{key}")
        try:
            c2.download_button("⬇️ Tải .docx", build_docx_bytes(text, title), file_name=f"{key}.docx",
                               key=f"dl_docx_{key}")
        except Exception as ex:  # python-docx thiếu
            c2.caption(f"Không tạo được .docx: {ex}")


def family_selector(prefix: str, default="Ống tròn mỏng"):
    opts = ["Chữ nhật đặc", "Tròn đặc", "Ống tròn mỏng", "Chữ I (cánh mỏng, h cố định)", "Tuỳ chỉnh I = α·Aⁿ"]
    st.session_state.setdefault(f"{prefix}_fam", default)
    c1, c2 = st.columns([2, 1.2])
    kind = c1.selectbox("Họ tiết diện", opts, key=f"{prefix}_fam",
                        help="Biến thiết kế duy nhất là diện tích A(x); I(x) và W(x) suy ra từ họ tiết diện (I = α·Aⁿ).")
    if kind == "Chữ nhật đặc":
        st.session_state.setdefault(f"{prefix}_r", 0.5)
        r = c2.number_input("b/h", 0.1, 2.0, step=0.05, key=f"{prefix}_r")
        fam = make_family("rect", ratio=r)
    elif kind == "Tròn đặc":
        fam = make_family("circle")
    elif kind == "Ống tròn mỏng":
        st.session_state.setdefault(f"{prefix}_t", 0.05)
        r = c2.number_input("t/D", 0.01, 0.2, step=0.01, key=f"{prefix}_t")
        fam = make_family("tube", ratio=r)
    elif kind.startswith("Chữ I"):
        st.session_state.setdefault(f"{prefix}_h", 0.4)
        h = c2.number_input("h (m)", 0.05, 3.0, step=0.05, key=f"{prefix}_h")
        fam = make_family("ibeam", h=h)
    else:
        cc = st.columns(4)
        st.session_state.setdefault(f"{prefix}_n", 2.0)
        st.session_state.setdefault(f"{prefix}_al", 0.08)
        st.session_state.setdefault(f"{prefix}_ga", 0.5)
        st.session_state.setdefault(f"{prefix}_be", 0.17)
        n = cc[0].number_input("n", 0.5, 3.0, step=0.1, key=f"{prefix}_n")
        al = cc[1].number_input("α", 1e-4, 10.0, step=0.01, format="%.4f", key=f"{prefix}_al")
        ga = cc[2].number_input("γ", 0.0, 1.5, step=0.05, key=f"{prefix}_ga")
        be = cc[3].number_input("β", 1e-4, 10.0, step=0.01, format="%.4f", key=f"{prefix}_be")
        fam = make_family("custom", n=n, alpha=al, gamma=ga, beta=be)

    with st.expander("Xem hình dạng tiết diện", expanded=True):
        area_cm2 = st.slider("Diện tích minh hoạ A (cm²)", 5, 500, 100, 5, key=f"{prefix}_section_area")
        show_section_preview(fam, area_cm2 * 1e-4, f"{prefix}_section_preview")
    return fam


def fig_layout(fig, title="", h=380, xt="", yt=""):
    fig.update_layout(title=title, height=h, margin=dict(l=50, r=20, t=50, b=40), template="plotly_white",
                      legend=dict(orientation="h", y=1.12, x=0), xaxis_title=xt, yaxis_title=yt)
    return fig


# ══════════════════════════════════════════════════════
#  TRỰC QUAN HOÁ TRƯỚC / SAU TỐI ƯU
#  Nét đứt = phương án trước tối ưu (mốc), nét liền + tô màu = sau tối ưu.
#  Kích thước vẽ ∝ A^g, g = γ của họ tiết diện (γ = 0, vd chữ I cao cố định → vẽ ∝ A).
# ══════════════════════════════════════════════════════
SAVE_FILL, UP_FILL, BODY_FILL = "rgba(44,160,44,.22)", "rgba(230,115,0,.45)", "rgba(31,119,180,.40)"


def _shape_exp(fam) -> float:
    return float(fam.gamma) if fam.gamma > 1e-9 else 1.0


def _band_xy(x, lo, hi):
    """Hai dải đối xứng (+ và −) giữa lo và hi, ngăn bằng None để vẽ trong 1 trace."""
    xs = np.concatenate([x, x[::-1]])
    up = np.concatenate([hi, lo[::-1]])
    dn = np.concatenate([-hi, -lo[::-1]])
    return (list(xs) + [None] + list(xs)), (list(up) + [None] + list(dn))


def fig_before_after_member(x, A_ref, A_opt, fam, ref_label, title, xt="x (m)", h=330):
    x = np.asarray(x, float)
    A_opt = np.asarray(A_opt, float)
    A_ref = np.full_like(x, float(A_ref)) if np.isscalar(A_ref) else np.asarray(A_ref, float)
    g = _shape_exp(fam)
    s = max(float(np.max(A_opt)), float(np.max(A_ref)), 1e-30)
    ho, hr = (A_opt / s) ** g, (A_ref / s) ** g
    fig = go.Figure()
    bx, by = _band_xy(x, np.minimum(ho, hr), hr)
    fig.add_scatter(x=bx, y=by, fill="toself", mode="lines", line=dict(width=0), fillcolor=SAVE_FILL,
                    name="Vật liệu tiết kiệm được", hoverinfo="skip")
    # thân tối ưu: một polygon kín từ −h_o đến +h_o
    fig.add_scatter(x=list(x) + list(x[::-1]), y=list(ho) + list(-ho[::-1]), fill="toself", mode="lines",
                    line=dict(color=BLUE, width=2.5), fillcolor=BODY_FILL, name="Sau tối ưu (nét liền)", hoverinfo="skip")
    if np.any(ho > hr + 1e-9):
        bx, by = _band_xy(x, hr, np.maximum(ho, hr))
        fig.add_scatter(x=bx, y=by, fill="toself", mode="lines", line=dict(width=0), fillcolor=UP_FILL,
                        name="Phần phải tăng thêm", hoverinfo="skip")
    for sgn, show in ((1, True), (-1, False)):
        fig.add_scatter(x=x, y=sgn * hr, mode="lines", line=dict(color=vz.before_color(), width=2, dash="dash"),
                        name=ref_label, showlegend=show, hoverinfo="skip")
    fig.add_hline(y=0, line=dict(color=GREY, width=1, dash="dot"))
    fig.update_yaxes(range=[-1.18, 1.18], showticklabels=False, zeroline=False)
    fig.update_layout(legend=dict(orientation="h", y=1.2, x=0))
    return fig_layout(fig, title, h, xt, "")


def _poly_quad(p0, p1, n, h0, h1):
    return [p0 + n * h0, p1 + n * h1, p1 - n * h1, p0 - n * h0, p0 + n * h0]


def plot_structure_before_after(res, title="Tiết diện từng thanh — trước (nét đứt) so với sau tối ưu (nét liền)"):
    inp, model, opt = res.inp, res.model, res.opt
    fam = inp.fam
    g = _shape_exp(fam)
    A_ref = float(opt.A_uniform)
    Amax = max(float(np.max(opt.A)), A_ref)
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    wfull = 0.05 * span                                     # bề rộng vẽ của tiết diện lớn nhất
    half = lambda A: 0.5 * wfull * (float(A) / Amax) ** g

    def add_poly(store, pts):
        store[0] += [float(q[0]) for q in pts] + [None]
        store[1] += [float(q[1]) for q in pts] + [None]

    blue, orange, dash = [[], []], [[], []], [[], []]
    for k, (i, j) in enumerate(inp.members):               # mốc: 1 hình chữ nhật nét đứt cho mỗi thanh
        p0, p1 = np.array(inp.nodes[i], float), np.array(inp.nodes[j], float)
        d = p1 - p0; Lm = float(np.hypot(*d)); n = np.array([-d[1], d[0]]) / Lm
        add_poly(dash, _poly_quad(p0, p1, n, half(A_ref), half(A_ref)))
    for e in range(model.n_el):                            # sau tối ưu: từng phần tử (thấy được vút thon trong thanh)
        p0, p1 = model.xy[model.conn[e, 0]], model.xy[model.conn[e, 1]]
        d = p1 - p0; Lm = float(np.hypot(*d)); n = np.array([-d[1], d[0]]) / Lm
        h = half(opt.A[e])
        add_poly(orange if opt.A[e] > A_ref * 1.0005 else blue, _poly_quad(p0, p1, n, h, h))

    fig = go.Figure()
    fig.add_scatter(x=blue[0], y=blue[1], fill="toself", mode="lines", fillcolor=BODY_FILL,
                    line=dict(color=BLUE, width=1.6), name="Sau tối ưu — nhỏ hơn mốc", hoverinfo="skip")
    fig.add_scatter(x=orange[0], y=orange[1], fill="toself", mode="lines", fillcolor=UP_FILL,
                    line=dict(color=ORANGE, width=1.6), name="Sau tối ưu — lớn hơn mốc", hoverinfo="skip")
    fig.add_scatter(x=dash[0], y=dash[1], mode="lines", line=dict(color=vz.before_color(), width=1.6, dash="dash"),
                    name="Trước tối ưu (tiết diện đều)", hoverinfo="skip")
    # điểm hover tại giữa mỗi thanh
    mx, my, tx = [], [], []
    for k, (i, j) in enumerate(inp.members):
        sel = res.member_of_el == k
        mx.append((inp.nodes[i][0] + inp.nodes[j][0]) / 2); my.append((inp.nodes[i][1] + inp.nodes[j][1]) / 2)
        a_o = float(res.A_member[k])
        tx.append(f"Thanh {k}: trước {A_ref * 1e4:.2f} cm² → sau {a_o * 1e4:.2f} cm² ({(a_o / A_ref - 1) * 100:+.0f}%)")
    fig.add_scatter(x=mx, y=my, mode="markers", marker=dict(size=14, color="rgba(0,0,0,0)"), text=tx,
                    hoverinfo="text", showlegend=False)
    _add_nodes_supports_loads(fig, inp, span, xs, ys)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    pad = 0.12 * span
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad]); fig.update_yaxes(range=[min(ys) - pad, max(ys) + 2 * pad])
    fig = fig_layout(fig, title, 580)
    fig.update_layout(margin=dict(l=50, r=20, t=105, b=40), legend=dict(orientation="h", y=1.0, yanchor="bottom", x=0))
    return fig


# ══════════════════════════════════════════════════════
#  TAB 0 — LÝ THUYẾT
# ══════════════════════════════════════════════════════
@st.cache_resource
def _cached_lecture_examples():
    return _oc.lecture_examples()


def _lecture_example_figure(key: str, item: dict):
    result = item["result"]
    if key == "two_bar":
        B = item["geometry"]["B"]
        h = result["h"]
        nodes = [(0.0, 0.0), (2 * B, 0.0), (B, h)]
        fig = go.Figure()
        for member_id, (i, j) in enumerate(((0, 2), (1, 2)), start=1):
            xi, yi = nodes[i]
            xj, yj = nodes[j]
            fig.add_trace(go.Scatter(
                x=[xi, xj], y=[yi, yj], mode="lines",
                line=dict(color="#2676a8", width=5), showlegend=False,
                hovertemplate=f"Thanh {member_id}<br>N = −{result['axial_force']:.2f} kN (nén)<extra></extra>",
            ))
            fig.add_annotation(x=(xi + xj) / 2, y=(yi + yj) / 2,
                               text=f"Thanh {member_id}: N = −{result['axial_force']:.1f} kN",
                               showarrow=False, bgcolor="rgba(255,255,255,.88)", font=dict(size=11))
        fig.add_trace(go.Scatter(x=[point[0] for point in nodes], y=[point[1] for point in nodes],
                                 mode="markers", marker=dict(size=10, color="#263746"), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[0.0, 2 * B], y=[-0.06 * h, -0.06 * h], mode="markers",
                                 marker=dict(symbol="triangle-up", size=15, color="#c44536"),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=B, y=h, ax=B, ay=h + 0.45 * h, showarrow=True,
                           arrowhead=3, arrowsize=1.2, arrowwidth=2.5, arrowcolor="#d94841",
                           text=f"P = {item['geometry']['P_kN']:.0f} kN")
        fig.add_annotation(x=B, y=-0.17 * h, text=f"2B = {2 * B:.2f} m", showarrow=False,
                           font=dict(size=11, color="#4b5d70"))
        fig.update_layout(title="Sơ đồ hình học và cân bằng nút đỉnh", height=350,
                          margin=dict(l=20, r=20, t=55, b=25), template="plotly_white")
        fig.update_xaxes(title="x (m)", range=[-0.25 * B, 2.25 * B], zeroline=False)
        fig.update_yaxes(title="y (m)", range=[-0.35 * h, 1.75 * h], scaleanchor="x", scaleratio=1,
                         zeroline=False)
        return fig, "Đối xứng nên hai thanh có cùng lực nén; tăng h làm thanh đứng hơn, giảm lực dọc nhưng tăng chiều dài/vật liệu."

    if key == "column":
        fig = go.Figure()
        fig.add_scatter(x=result.x, y=result.A_opt * 1e4, mode="lines", name="A*(x) — tối ưu",
                        line=dict(color=BLUE, width=3),
                        hovertemplate="x=%{x:.2f} m<br>A=%{y:.2f} cm²<extra></extra>")
        fig.add_scatter(x=[0, result.L], y=[result.A_uniform * 1e4] * 2, mode="lines",
                        name="Cột đều — cùng V", line=dict(color=ORANGE, width=2, dash="dash"),
                        hovertemplate="A đều=%{y:.2f} cm²<extra></extra>")
        fig.update_layout(title="Cột khớp–khớp: dồn vật liệu theo dạng mất ổn định", height=350,
                          margin=dict(l=55, r=20, t=55, b=45), template="plotly_white",
                          legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"))
        fig.update_xaxes(title="x dọc cột (m)")
        fig.update_yaxes(title="Diện tích A (cm²)", rangemode="tozero")
        return fig, "V cố định: cột tối ưu tăng A ở vùng mode uốn lớn và giảm A gần gối; đường gạch là cột đều cùng thể tích."

    if key == "beam":
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.16,
                            subplot_titles=("Diện tích dọc dầm", "Đường võng (hướng xuống dương)"))
        fig.add_trace(go.Scatter(x=result.x_mid, y=result.A_opt * 1e4, mode="lines",
                                 name="A*(x) — tối ưu", line=dict(color=BLUE, width=3),
                                 hovertemplate="x=%{x:.2f} m<br>A=%{y:.2f} cm²<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0, result.prob.L], y=[result.A_uniform * 1e4] * 2, mode="lines",
                                 name="Dầm đều — cùng giới hạn", line=dict(color=ORANGE, width=2, dash="dash"),
                                 hovertemplate="A đều=%{y:.2f} cm²<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=result.xs, y=result.deflection_opt * 1e3, mode="lines",
                                 name="Võng tối ưu", line=dict(color=RED, width=2.5),
                                 hovertemplate="x=%{x:.2f} m<br>δ=%{y:.2f} mm<extra></extra>"), row=2, col=1)
        fig.add_trace(go.Scatter(x=result.xs, y=result.deflection_uni * 1e3, mode="lines",
                                 name="Võng dầm đều", line=dict(color="#697586", width=2, dash="dash"),
                                 hovertemplate="x=%{x:.2f} m<br>δ=%{y:.2f} mm<extra></extra>"), row=2, col=1)
        if result.prob.delta_max is not None:
            limit_mm = result.prob.delta_max * 1e3
            fig.add_hline(y=limit_mm, line=dict(color="#4b5563", dash="dot"), row=2, col=1)
            fig.add_hline(y=-limit_mm, line=dict(color="#4b5563", dash="dot"), row=2, col=1)
        fig.update_layout(title="Dầm tựa đơn: vật liệu theo mômen ảo và đáp ứng võng", height=540,
                          margin=dict(l=55, r=20, t=75, b=75), template="plotly_white",
                          legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"))
        fig.update_xaxes(title_text="x dọc dầm (m)", row=2, col=1)
        fig.update_yaxes(title_text="A (cm²)", rangemode="tozero", row=1, col=1)
        fig.update_yaxes(title_text="Độ võng (mm)", autorange="reversed", row=2, col=1)
        return fig, "Vật liệu được ưu tiên nơi |M·m̄| lớn; đường chấm trong đồ thị võng là ±δ giới hạn."

    raise ValueError(f"Không có hình minh họa cho ví dụ: {key}")


def _constitutive_model_figure(material_name):
    """Draw the representative uniaxial material law used by nonlinear RC THA."""
    fig = go.Figure()
    if material_name == "Concrete01":
        fc = 30.0  # MPa, representative B30 input shown in the OpenSees page
        eps_peak, eps_ult, residual_ratio = 0.002, 0.006, 0.20
        eps = np.r_[np.linspace(0.0, -eps_peak, 70),
                    np.linspace(-eps_peak, -eps_ult, 70),
                    np.linspace(-eps_ult, -0.008, 20)]
        sigma = np.empty_like(eps)
        first = eps >= -eps_peak
        r = np.divide(-eps[first], eps_peak, out=np.zeros_like(eps[first]), where=eps_peak > 0)
        sigma[first] = -fc * (2.0 * r - r**2)
        second = (eps < -eps_peak) & (eps >= -eps_ult)
        sigma[second] = -fc * (1.0 - (1.0-residual_ratio) *
                               ((-eps[second]-eps_peak)/(eps_ult-eps_peak)))
        sigma[eps < -eps_ult] = -fc * residual_ratio
        fig.add_trace(go.Scatter(x=eps, y=sigma, mode="lines", name="Concrete01 · nén",
                                 line=dict(color="#b65b43", width=3),
                                 hovertemplate="εc=%{x:.4f}<br>σc=%{y:.1f} MPa<extra></extra>"))
        fig.add_vline(x=-eps_peak, line_dash="dot", line_color="#6b7780",
                      annotation_text="εc0 = −0.002")
        fig.add_vline(x=-eps_ult, line_dash="dot", line_color="#6b7780",
                      annotation_text="εcu = −0.006")
        fig.update_layout(title="Concrete01 · đường bao nén đơn trục",
                          xaxis_title="Biến dạng bê tông εc (−)",
                          yaxis_title="Ứng suất bê tông σc (MPa)")
        fig.add_annotation(x=0.99,y=0.04,xref="paper",yref="paper",showarrow=False,
                           text="Không chịu kéo trong mô hình này",font=dict(color="#65717a"))
    else:
        fy, es, b, r0 = 350.0, 200000.0, 0.01, 18.0
        ey = fy / es
        eps = np.linspace(-0.035, 0.035, 500)
        xi = eps / ey
        sigma = fy * (b*xi + (1.0-b)*xi / (1.0 + np.abs(xi)**r0)**(1.0/r0))
        fig.add_trace(go.Scatter(x=eps, y=sigma, mode="lines", name="Steel02 · đường đơn điệu",
                                 line=dict(color="#286b8e", width=3),
                                 hovertemplate="εs=%{x:.4f}<br>σs=%{y:.1f} MPa<extra></extra>"))
        fig.add_trace(go.Scatter(x=[-ey, ey], y=[-fy, fy], mode="markers+text",
                                 text=["−fy", "fy"], textposition="top center",
                                 marker=dict(size=8,color="#c7463b"),name="Điểm chảy tham chiếu"))
        fig.add_vline(x=-ey,line_dash="dot",line_color="#87929a")
        fig.add_vline(x=ey,line_dash="dot",line_color="#87929a")
        fig.update_layout(title="Steel02 · Menegotto–Pinto, đường đơn điệu minh họa",
                          xaxis_title="Biến dạng thép εs (−)",
                          yaxis_title="Ứng suất thép σs (MPa)")
        fig.add_annotation(x=0.99,y=0.04,xref="paper",yref="paper",showarrow=False,
                           text="Đường cong chuyển tiếp trơn; vòng đảo chiều phụ thuộc lịch sử tải",
                           font=dict(color="#65717a"))
    fig.update_layout(height=390,margin=dict(l=20,r=20,t=65,b=30),
                      template="plotly_white",hovermode="x unified",
                      xaxis=dict(zeroline=True,zerolinecolor="#77828a",gridcolor="#e3e8eb"),
                      yaxis=dict(zeroline=True,zerolinecolor="#77828a",gridcolor="#e3e8eb"))
    return fig


def _render_dynamics_theory_library():
    """Theory lookup for concepts and assumptions exposed by the dynamics UI."""
    entries = [
        ("Mô hình", "SDOF · hệ một bậc tự do", "Hệ lý tưởng có một tọa độ độc lập, thường là khối lượng–lò xo–cản. Đây là mô hình cơ bản để hiểu dao động và phổ phản ứng.", r"m\ddot u+c\dot u+ku=p(t)", "Tab SDOF dùng để xem ảnh hưởng của khối lượng, độ cứng, cản và tải lên chu kỳ/đáp ứng.", "Không mô tả phân bố biến dạng hoặc nhiều mode của công trình nhiều tầng."),
        ("Mô hình", "MDOF · hệ nhiều bậc tự do", "Hệ có vector nhiều tọa độ độc lập; mỗi tầng thường có chuyển vị ngang riêng.", r"\mathbf M\ddot{\mathbf u}+\mathbf C\dot{\mathbf u}+\mathbf K\mathbf u=\mathbf p(t)", "Mô hình nhiều tầng dùng khối lượng, ma trận độ cứng và cản để tính mode cùng đáp ứng.", "Số bậc tự do và các liên kết phải phản ánh đúng mô hình kết cấu."),
        ("Mô hình", "Shear building · nhà cắt", "Mô hình lý tưởng hóa mỗi tầng thành một khối lượng tập trung; biến dạng ngang tập trung ở các lò xo tầng, sàn được coi cứng trong mặt phẳng.", r"V_i=k_i(u_i-u_{i-1})", "Dùng để giải thích và tính nhanh đáp ứng ngang nhiều tầng; FEM 3D có thể được ngưng tụ thành độ cứng tầng tương đương.", "Bỏ qua biến dạng dầm/cột phân bố, xoắn sàn và các dạng biến dạng ngoài giả thiết shear-building."),
        ("Mô hình", "Shear wall tương đương", "Trong bài mẫu, tường cứng được giả thiết EI→∞ để gần như không biến dạng; độ mềm ngang được quy về các lò xo tầng.", r"EI\to\infty\quad\Rightarrow\quad \text{biến dạng tập trung tại lò xo tầng}", "Đây là giả thiết tương đương để minh họa shear-building, không phải phần tử tường được phân tích.", "Không dùng giả thiết này để kết luận nội lực, nứt, cắt hoặc cấu tạo của vách thực."),
        ("Mô hình", "Diaphragm cứng · sàn cứng trong mặt phẳng", "Các nút sàn cùng chuyển động ngang theo một mặt phẳng cứng; trong không gian vẫn có thể có chuyển vị và xoay tổng thể.", r"u_x^{(a)}=u_x^{(b)}\quad\text{trên cùng tầng}", "OpenSees 3D dùng ràng buộc diaphragm để liên kết nút sàn.", "Không có nghĩa sàn tuyệt đối cứng ngoài mặt phẳng hoặc bỏ qua bất quy tắc xoắn."),
        ("Mô hình", "OpenSees / OpenSeesPy", "OpenSees là nền tảng phân tích kết cấu bằng phần tử hữu hạn; OpenSeesPy cung cấp giao diện Python để dựng mô hình, vật liệu, tải, phân tích và truy xuất kết quả.", r"\text{model} + \text{elements} + \text{materials} + \text{analysis}", "Tab Động đất không gian gọi solver OpenSeesPy cho ELF, MRSA và THA; tab bài mẫu trình bày shear-building lý tưởng hóa.", "Kết quả phụ thuộc mô hình, đơn vị, thuật toán và dữ liệu đầu vào; tên phần mềm không tự bảo đảm thiết kế đúng."),
        ("Vật liệu", "Đàn hồi tuyến tính", "Ứng suất tỷ lệ với biến dạng và dỡ tải trở về trạng thái ban đầu; độ cứng không phụ thuộc lịch sử tải.", r"\sigma=E\varepsilon;\qquad \mathbf f_s=\mathbf K\mathbf u", "Dùng trong bài toán dao động đàn hồi, ELF/MRSA và mô hình khung không gian THA hiện tại.", "Không mô tả chảy dẻo, nứt, suy giảm độ cứng/sức kháng hoặc năng lượng dẻo."),
        ("Vật liệu", "Phi tuyến vật liệu", "Quan hệ lực–chuyển vị hoặc ứng suất–biến dạng thay đổi theo mức tải và có thể phụ thuộc lịch sử.", r"\mathbf f_s=\mathbf f_s(\mathbf u,\mathbf z)", "Mô hình giảm bậc THA có luật song tuyến tính; vật liệu OpenSees phi tuyến chỉ được dùng khi mô hình và phần tử khai báo vật liệu đó.", "Phi tuyến vật liệu không đồng nghĩa đã mô hình hóa phi tuyến hình học hoặc mọi cơ chế phá hoại."),
        ("Vật liệu", "HystereticSM", "Uniaxial material của OpenSees mô tả đường bao đa tuyến và đáp ứng chu kỳ, có các tham số pinching, suy giảm sức kháng/độ cứng và năng lượng.", r"F=F(\Delta,\text{lịch sử};\,\text{envelope, pinch, damage})", "Dùng làm ví dụ vật liệu lò xo trong bài shear-building OpenSees; tham số phải lấy/hiệu chuẩn từ thí nghiệm hoặc nguồn có căn cứ.", "Không tự đại diện cho bê tông cốt thép hay khớp dẻo dầm/cột; kết quả phụ thuộc mạnh vào tham số."),
        ("Vật liệu", "Fiber section · tiết diện thớ", "Tiết diện được chia thành các thớ bê tông và thép; mỗi thớ dùng luật ứng suất–biến dạng riêng. OpenSees tích phân ứng suất thớ để thu lực dọc và mô men tiết diện.", r"N=\int_A\sigma\,dA;\qquad M=\int_A\sigma z\,dA", "THA phi tuyến không gian dùng tọa độ và diện tích thanh thép đã nhập để dựng Fiber section cột/dầm.", "Kết quả phụ thuộc mật độ chia thớ, vật liệu, vị trí thanh và giả thiết tương thích biến dạng; mô hình hiện chưa xét confinement, tuột neo và cắt."),
        ("Vật liệu", "Concrete01", "Vật liệu bê tông đơn trục Kent–Scott–Park. fpc là ứng suất nén đỉnh (nhập âm theo quy ước OpenSees), epsc0 là biến dạng tại đỉnh; fpcu và epscu là ứng suất và biến dạng cuối nhánh suy giảm. Không có sức chịu kéo.", r"(f_{pc},\epsilon_{c0})\to(f_{pcu},\epsilon_{cu})", "Dùng cho các thớ bê tông trong Fiber section; hình bên dưới minh họa nhánh nén với tham số đại diện mà giao diện THA đang sử dụng.", "Tool hiện dùng đường bao giản lược, bê tông không kéo và chưa tách bê tông lõi được đai kiềm chế khỏi lớp bảo vệ."),
        ("Vật liệu", "Steel02 · thép Menegotto–Pinto", "Mô hình thép đơn trục có chuyển tiếp trơn qua chảy và hóa bền. fy là giới hạn chảy, E0 là mô đun đàn hồi, b là tỷ lệ độ cứng sau chảy; R0, cR1, cR2 điều khiển độ cong và chuyển tiếp khi đảo chiều tải.", r"\sigma_s=\sigma_s(\epsilon_s; f_y,E_0,b,R_0,cR_1,cR_2)", "Mỗi thanh dọc nhập ở mặt cắt trở thành thớ thép; số thanh, đường kính và vị trí ảnh hưởng trực tiếp đến sức kháng uốn/độ cứng sau chảy. Đồ thị minh họa bên dưới là nhánh đơn điệu; THA dùng luật chu kỳ của vật liệu.", "Mô hình hiện không tự hiệu chuẩn theo thí nghiệm thép, không có tuột neo hay suy giảm mỏi chu kỳ thấp."),
        ("Mô hình", "NonlinearBeamColumn · phần tử dầm-cột phi tuyến phân bố", "Phần tử dầm-cột tích phân đáp ứng của tiết diện tại một số điểm dọc cấu kiện để mô tả chảy dẻo phân bố thay vì đặt một khớp tập trung duy nhất.", r"\mathbf f_e=\int_0^L\mathbf B^T\mathbf s(\epsilon)\,dx", "THA phi tuyến dùng 5 điểm tích phân Lobatto cùng Fiber section; phần tử đàn hồi vẫn dùng cho ELF/MRSA và đối chứng.", "Giả thiết biến dạng cắt nhỏ theo formulation hiện dùng; không tự mô hình hóa phá hoại cắt, nút, neo hoặc hiệu ứng P–Delta."),
        ("Vật liệu", "Backbone / đường bao", "Đường đơn điệu nối các điểm đặc trưng của vật liệu, ví dụ gốc, chảy, cực hạn và suy giảm.", r"(0,0)\to(\Delta_y,F_y)\to(\Delta_u,F_u)", "Giải thích sức kháng tham chiếu của lò xo trước khi xem vòng tải chu kỳ.", "Đường bao không phải lịch sử tải và không tự cho biết năng lượng tiêu tán."),
        ("Vật liệu", "Pinching · thắt vòng trễ", "Hiện tượng lực tái huy động nhỏ khi đảo chiều tải, làm vòng lực–chuyển vị co vào gần gốc.", r"F_{reload}<F_{envelope}\quad\text{trên nhánh tái tải}", "HystereticSM có tham số điều khiển pinching; cần hiệu chuẩn phù hợp thí nghiệm cấu kiện/liên kết.", "Mô hình hiện tại không được mặc định xem như đã hiệu chuẩn pinching nếu chưa khai báo và xác minh tham số."),
        ("Vật liệu", "Vòng trễ và năng lượng tiêu tán", "Quỹ đạo lực–chuyển vị khi tải lặp; diện tích khép kín của chu kỳ là công cơ học tiêu tán trong chu kỳ đó.", r"E_D=\oint F\,d\Delta", "THA và phân tích tải chu kỳ dùng biểu đồ V–u để quan sát đáp ứng theo lịch sử.", "Vòng hẹp/đàn hồi có năng lượng dẻo nhỏ; phải phân biệt với bao lồi hình học hoặc đường minh họa."),
        ("Dao động", "Chu kỳ riêng T₁ và mode", "Mode là dạng dao động riêng; T₁ là chu kỳ của mode có tần số thấp nhất trong hệ ổn định.", r"\mathbf K\boldsymbol\phi_r=\omega_r^2\mathbf M\boldsymbol\phi_r;\quad T_r=2\pi/\omega_r", "Tool giải bài toán trị riêng từ M và K FEM để lấy chu kỳ, mode và hiệu chỉnh cản.", "T₁ phụ thuộc độ cứng, khối lượng, điều kiện biên và giả thiết mô hình."),
        ("Dao động", "Cản nhớt và Rayleigh damping", "Cản làm giảm biên độ dao động; cản Rayleigh xấp xỉ bằng tổ hợp ma trận khối lượng và độ cứng.", r"\mathbf C=\alpha_M\mathbf M+\beta_K\mathbf K;\quad \xi_r=\tfrac12(\alpha_M/\omega_r+\beta_K\omega_r)", "THA OpenSees dùng cản Rayleigh; hệ số được chọn theo tỷ số cản và mode mục tiêu.", "Cản Rayleigh phụ thuộc tần số; lựa chọn hệ số không phù hợp có thể tạo cản giả lớn ở mode cao."),
        ("Dao động", "Newmark-β", "Thuật toán tích phân số theo thời gian, cập nhật chuyển vị, vận tốc và gia tốc ở từng bước.", r"u_{n+1}=u_n+\Delta t\,v_n+\Delta t^2[(\tfrac12-\beta)a_n+\beta a_{n+1}]", "Dùng để giải phương trình chuyển động trong THA; OpenSees chạy Newmark với tham số đã khai báo.", "Độ chính xác phụ thuộc bước thời gian, hội tụ và dạng tải; ổn định số không bảo đảm mô hình vật lý đúng."),
        ("Động đất", "Tải gia tốc nền", "Chuyển động nền tạo lực quán tính tương đương trên khối lượng kết cấu.", r"\mathbf M\ddot{\mathbf u}+\mathbf C\dot{\mathbf u}+\mathbf f_s=-\mathbf M\mathbf r a_g(t)", "THA nhận accelerogram; có thể nhập bản ghi hoặc dùng tín hiệu tổng hợp chỉ để minh họa.", "Tín hiệu tổng hợp chưa xử lý/hiệu chỉnh phổ không phải bản ghi thiết kế hay động đất thực."),
        ("Động đất", "ELF · Equivalent Lateral Force", "Phương pháp tĩnh lực ngang tương đương: thay hiệu ứng động đất bằng lực ngang thiết kế phân bố theo tầng.", r"F_b=S_a(T_1)\sum_i m_i\,\lambda;\qquad F_i\propto m_i z_i", "Tool dùng để ước tính đáp ứng đàn hồi khi kết cấu thỏa điều kiện áp dụng của tiêu chuẩn.", "Không tạo lịch sử theo thời gian; tham số phổ, hệ số và điều kiện áp dụng phải theo tiêu chuẩn/dự án."),
        ("Động đất", "Phổ phản ứng", "Đường biểu diễn đáp ứng cực đại của hệ SDOF theo chu kỳ riêng, với tỷ số cản và phổ gia tốc/chuyển vị quy định.", r"S_a=S_a(T,\xi);\qquad S_d=S_a/\omega^2", "ELF lấy Sa(T₁); MRSA lấy phổ tại chu kỳ từng mode.", "Phổ cho cực đại, không giữ dấu thời gian/pha và không tự tạo lịch sử đáp ứng."),
        ("Động đất", "MRSA · Modal Response Spectrum Analysis", "Phân tích phổ phản ứng dạng dao động: phân tích đáp ứng từng mode rồi tổ hợp các cực đại modal.", r"R_r=\Gamma_r\phi_r S_d(T_r)", "Tool tính mode, hệ số tham gia, đáp ứng modal và tổ hợp SRSS hoặc CQC.", "Kết quả thường đàn hồi; cần đủ mode và phổ phù hợp; tổ hợp mode không khôi phục lịch sử thời gian."),
        ("Động đất", "THA · Time-History Analysis", "Phân tích lịch sử thời gian: tích phân phương trình chuyển động theo chuỗi gia tốc nền ở từng bước.", r"\mathbf M\ddot{\mathbf u}(t)+\mathbf C\dot{\mathbf u}(t)+\mathbf f_s(t)=-\mathbf M\mathbf r a_g(t)", "Tool vẽ chuyển vị, lệch tầng, lực cắt, hoạt ảnh và vòng V–u theo thời gian.", "Kết quả nhạy với bản ghi, scale, bước thời gian, cản, mô hình vật liệu và tiêu chí hội tụ."),
        ("Động đất", "SRSS · căn tổng bình phương", "Cách tổ hợp cực đại modal giả định các đáp ứng mode gần như độc lập.", r"R=\sqrt{\sum_r R_r^2}", "Lựa chọn tổ hợp trong MRSA khi tương tác giữa các mode không đáng kể.", "Có thể không phù hợp với các mode có tần số gần nhau."),
        ("Động đất", "CQC · tổ hợp toàn phương đầy đủ", "Tổ hợp đáp ứng modal có xét tương quan giữa các mode, phụ thuộc tỷ số cản và tỷ lệ tần số.", r"R=\sqrt{\sum_r\sum_s\rho_{rs}R_rR_s}", "Tool cho chọn CQC trong MRSA; thường phù hợp hơn SRSS khi mode gần nhau.", "CQC vẫn là tổ hợp thống kê cực đại, không phải phép cộng đại số và không tạo lịch sử thời gian."),
        ("Động đất", "Lệch tầng / drift", "Chuyển vị tương đối giữa hai sàn liên tiếp; drift ratio là lệch tầng chia chiều cao tầng.", r"\Delta_i=u_i-u_{i-1};\qquad \theta_i=\Delta_i/h_i", "Dùng để theo dõi biến dạng tầng và so với giới hạn dự án/tiêu chuẩn.", "Phải dùng đúng định nghĩa chuyển vị và tổ hợp theo tiêu chuẩn; một giá trị đơn lẻ không xác nhận an toàn."),
        ("Động đất", "Độ dẻo μ", "Tỷ số biến dạng cực đại với biến dạng chảy tham chiếu.", r"\mu=|\Delta_{max}|/\Delta_y", "THA phi tuyến giảm bậc dùng để theo dõi mức vượt ngưỡng chảy của lò xo tương đương.", "Không đồng nhất độ dẻo cấu kiện nếu Δy chưa được xác định từ mô hình tiết diện/cấu kiện có căn cứ."),
        ("FEM & thiết kế", "FEM và ngưng tụ độ cứng tầng", "FEM chia kết cấu thành phần tử/nút, lắp ma trận độ cứng; ngưng tụ loại bỏ bậc tự do phụ để giữ đáp ứng ở các tọa độ quan tâm.", r"\mathbf K_c=\mathbf K_{xx}-\mathbf K_{xr}\mathbf K_{rr}^{-1}\mathbf K_{rx}", "Tool tính độ cứng khung 3D đàn hồi rồi quy đổi về shear-building tương đương.", "Độ cứng tương đương không cung cấp tự động toàn bộ nội lực/ứng suất cấu kiện của FEM gốc."),
        ("FEM & thiết kế", "Ứng suất, η demand/capacity và dải màu", "η là tỷ số nhu cầu tính toán trên sức kháng tương ứng. η càng gần 1 thì cấu kiện càng dùng nhiều sức kháng; η>1 báo vượt mức sàng lọc của mô hình.", r"\sigma=N/A\pm Mc/I;\qquad \eta=E_d/R_d", "Trên bản đồ màu của tool: xanh đậm η<0,50 (mức dùng thấp); xanh nhạt 0,50–0,75; vàng 0,75–0,90; cam 0,90–1,00 (gần giới hạn); đỏ η>1,00 (vượt sức kháng tham chiếu).", "Các dải màu là ngưỡng hiển thị do giao diện chọn, không phải giới hạn độc lập của tiêu chuẩn. Trong mô hình giảm bậc, η là ước tính chia lực tầng cho cột; chưa phải contour nội lực FEM phần tử. Không thay thế tương tác N–M, cắt, ổn định, cấu tạo và tổ hợp tải theo tiêu chuẩn."),
        ("FEM & thiết kế", "Ngàm chân cột và móng", "Ngàm lý tưởng khóa chuyển vị và xoay tại chân cột; liên kết móng thực có thể có độ mềm và sức kháng hữu hạn.", r"u_x=u_y=u_z=\theta_x=\theta_y=\theta_z=0\quad\text{(ngàm lý tưởng)}", "Mô hình khung hiện dùng chân ngàm để phân tích; hình móng/cổ cột minh họa đường truyền lực.", "Chưa tự động kiểm tra trượt, lật, sức chịu tải đất, neo thép chờ và phá hoại đài."),
        ("FEM & thiết kế", "Kiểm chứng và xác nhận mô hình", "Verification kiểm tra cách giải số có đúng phương trình/mã dự kiến; validation so mô hình với dữ liệu thực nghiệm hoặc quan sát phù hợp.", r"\text{Mô hình số}\xrightarrow{\text{verification}}\text{nghiệm đúng}\xrightarrow{\text{validation}}\text{mô tả thực tế}", "Cần đối chiếu đơn vị, cân bằng lực, hội tụ, nghiệm chuẩn và dữ liệu thử nghiệm trước khi dùng kết quả thiết kế.", "Chạy được phần mềm hoặc có biểu đồ đẹp không đồng nghĩa mô hình đã được xác nhận."),
    ]
    st.subheader("Cơ sở lý thuyết động lực học và động đất")
    st.markdown("Tra cứu các khái niệm, phương trình nền tảng, vị trí sử dụng trong tool và giả thiết cần nhớ. Nội dung được nhóm theo chủ đề để tiện đối chiếu với từng bài toán.")
    all_categories=sorted({item[0] for item in entries})
    filter_col, search_col=st.columns([0.34,0.66])
    with filter_col:
        category=st.selectbox("Nhóm kiến thức",["Tất cả"]+all_categories,key="theory_dyn_category")
    with search_col:
        query=st.text_input("Tìm khái niệm",placeholder="Ví dụ: CQC, chu kỳ T1, pinching, độ dẻo…",key="theory_dyn_search").strip().casefold()
    filtered=[item for item in entries if (category=="Tất cả" or item[0]==category) and (not query or query in " ".join(item).casefold())]
    if not filtered:
        st.info("Không tìm thấy thuật ngữ phù hợp. Thử từ khóa ngắn hơn hoặc chọn ‘Tất cả’.")
    else:
        labels=[f"{item[1]} · {item[0]}" for item in filtered]
        chosen=st.selectbox("Chọn khái niệm để xem thuyết minh",labels,key="theory_dyn_term")
        item=filtered[labels.index(chosen)]
        st.markdown(f"### {item[1]}")
        st.markdown(item[2])
        st.latex(item[3])
        if item[1] in {"Concrete01", "Steel02 · thép Menegotto–Pinto"}:
            material_name = "Concrete01" if item[1] == "Concrete01" else "Steel02"
            st.plotly_chart(_constitutive_model_figure(material_name),
                            use_container_width=True,
                            key=f"theory_material_{material_name.lower()}")
            if material_name == "Concrete01":
                st.caption("Hình là đường bao nén minh họa theo bộ tham số đang dùng trong THA: f′c = 30 MPa, εc0 = −0.002, fcu = −0.20f′c, εcu = −0.006. Giá trị này cần được thay/hiệu chuẩn theo vật liệu thực tế.")
            else:
                st.caption("Hình đường đơn điệu minh họa dùng fy = 350 MPa, Es = 200 GPa và b = 0.01. Steel02 có đáp ứng chu kỳ theo Menegotto–Pinto; đồ thị này không phải kết quả thử nghiệm hay vòng trễ đã hiệu chuẩn.")
        st.markdown(f"**Tool sử dụng:** {item[4]}")
        st.markdown(
            f"<div class='theory-limit'><strong>Giả thiết / giới hạn:</strong> {item[5]}</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"{len(filtered)} khái niệm trong nhóm đang hiển thị · danh mục sẽ được mở rộng khi thêm thuật toán hoặc mô hình mới.")


def tab_theory():
    st.header("Cơ sở lý thuyết")
    _render_dynamics_theory_library()
    st.divider()
    st.subheader("Cơ sở toán học · phép tính biến phân và tối ưu kết cấu")
    st.header("Cơ sở toán học")
    st.markdown("""
**Ý tưởng.** Khi biến thiết kế là *hàm liên tục* $A(x)$ (dầm vút thon, cột biến tiết diện) ta không tìm một vài con số mà tìm *cả một hàm*.
Đó là bài toán cực trị phiếm hàm. Đớp Tool dùng phép tính biến phân để ra **dạng đóng** của $A^*(x)$, rồi dùng FEM + tiêu chuẩn tối ưu để
(i) kiểm chứng và (ii) mở rộng cho các sơ đồ siêu tĩnh, khung và dàn, nơi không còn nghiệm đóng.
""")
    st.subheader("1. Biến phân bậc nhất và phương trình Euler–Lagrange")
    st.latex(r"J[y]=\int_{0}^{L}F(x,y,y')\,\mathrm dx\ \to\ \text{stationary}\quad\Rightarrow\quad F_y-\frac{\mathrm d}{\mathrm dx}\big(F_{y'}\big)=0")
    st.markdown(r"""
Chọn hàm thử $y_\varepsilon=y+\varepsilon\eta$, trong đó $\eta$ là biến phân khả dĩ. Với hai đầu cố định, $\eta(a)=\eta(b)=0$.
""")
    st.latex(r"\delta J=\left.\frac{\mathrm dJ[y+\varepsilon\eta]}{\mathrm d\varepsilon}\right|_{\varepsilon=0}=\int_a^b(F_y\eta+F_{y'}\eta')\,\mathrm dx")
    st.latex(r"\delta J=\int_a^b\left(F_y-\frac{\mathrm d}{\mathrm dx}F_{y'}\right)\eta\,\mathrm dx+\left[F_{y'}\eta\right]_a^b")
    st.markdown(r"""
Tích phân từng phần đưa đạo hàm khỏi $\eta$. Theo **bổ đề cơ bản của phép tính biến phân**, nếu $\delta J=0$ với mọi $\eta$ khả dĩ thì
$F_y-\frac{\mathrm d}{\mathrm dx}F_{y'}=0$. Ở đầu tự do, $\eta$ không bị ấn định nên số hạng biên cho điều kiện tự nhiên $F_{y'}=0$;
điều kiện biên cơ bản thì được áp đặt trước lên lớp hàm.
""")
    st.subheader("1.1. Đưa ràng buộc vào bài toán biến phân")
    st.markdown(r"Với ràng buộc toàn cục $G[y]=\int_a^b g(x,y,y')\,\mathrm dx-G_0=0$, lập phiếm hàm Lagrange $H=F+\lambda g$ rồi áp dụng Euler–Lagrange cho $H$. Khi $F$ không chứa $y'$, điều kiện dừng suy biến thành phương trình đại số.")
    st.latex(r"H_y-\frac{\mathrm d}{\mathrm dx}H_{y'}=0,\qquad A_{\min}\leq A(x)\leq A_{\max}")
    st.markdown("Với cận cục bộ, chia miền thành vùng tự do và vùng chạm cận; tại nghiệm phải kiểm tra khả thi, dấu nhân tử và bù trừ, không chỉ phương trình dừng.")
    with st.expander("Ví dụ trên lớp: tối ưu giàn hai thanh và kiểm tra KKT", expanded=False):
        c1, c2, c3 = st.columns(3)
        load_kN = c1.number_input("P (kN)", min_value=0.1, value=150.0, step=10.0, key="tb_P")
        half_span_mm = c2.number_input("B (mm)", min_value=1.0, value=1000.0, step=100.0, key="tb_B")
        stress_MPa = c3.number_input("[σ] (MPa)", min_value=0.1, value=180.0, step=10.0, key="tb_sigma")
        density = st.number_input("Khối lượng riêng ρ (kg/m³)", min_value=1.0, value=7850.0, step=100.0, key="tb_rho")
        example = two_bar_kkt_solution(load_kN, half_span_mm / 1000, stress_MPa * 1000, density)
        metrics([
            ("Chiều cao tối ưu h*", f"{example['h'] * 1000:.1f} mm"),
            ("Góc thanh α*", "45°"),
            ("Diện tích mỗi thanh A*", f"{example['area'] * 1e6:.1f} mm²"),
            ("Khối lượng giàn m*", f"{example['mass']:.2f} kg"),
        ])
        report_block(example["report"], "gian_hai_thanh_kkt", "Thuyết minh — Giàn hai thanh và KKT")
    lecture_examples = _cached_lecture_examples()
    st.markdown('<div class="section-callout"><strong>Bộ ví dụ về áp dụng lý thuyết chương 1-4.</strong> Mỗi ví dụ minh họa một nội dung đã học: giàn hai thanh, ổn định cột và tối ưu dầm vút thon.</div>', unsafe_allow_html=True)
    example_cols = st.columns(len(lecture_examples))
    for idx, (key, item) in enumerate(lecture_examples.items()):
        with example_cols[idx]:
            st.markdown(f"<div class='example-card'><h4>{item['title']}</h4><p>{item['summary']}</p><span class='chip'>{key}</span></div>", unsafe_allow_html=True)
            if st.button(f"Xem {item['title']}", key=f"see_{key}"):
                st.session_state[f"example_report_{key}"] = item["report"]
    for key, item in lecture_examples.items():
        if st.session_state.get(f"example_report_{key}"):
            figure, geometric_note = _lecture_example_figure(key, item)
            st.plotly_chart(figure, width="stretch", config={"displayModeBar": False},
                            key=f"lecture_figure_{key}")
            st.caption(geometric_note)
            report_block(st.session_state[f"example_report_{key}"], f"lecture_{key}", item["title"])

    st.subheader("2. Dầm vút thon — cực tiểu thể tích với ràng buộc độ võng")
    st.markdown(r"Họ tiết diện: $I=\alpha A^{n}$ (chữ nhật/tròn/ống đồng dạng $n=2$; chữ I cánh mỏng với $h$ cố định $n=1$). "
                r"Độ võng tại điểm khống chế $x_p$ theo nguyên lý công ảo:")
    st.latex(r"\delta(x_p)=\int_0^L\frac{M(x)\,\bar m(x)}{E\,\alpha\,A^{n}(x)}\,\mathrm dx=\delta_{\max},\qquad V=\int_0^L A\,\mathrm dx\to\min")
    st.latex(r"\tilde F=A+\lambda\,\frac{M\bar m}{E\alpha A^{n}}\ \ (\text{independent of }A')\ \Rightarrow\ 1-\frac{\lambda n M\bar m}{E\alpha A^{n+1}}=0")
    st.latex(r"\boxed{A^{*}(x)=\Big[\tfrac{\lambda n}{E\alpha}\,M(x)\,\bar m(x)\Big]^{\frac1{n+1}}}")
    st.markdown(r"""
- Tải tập trung tại điểm khống chế: $\bar m\propto M\Rightarrow A^*\propto|M|^{\frac{2}{n+1}}$ (chữ nhật/tròn: $|M|^{2/3}$; chữ I cánh mỏng: $|M|$).
- Kết hợp điều kiện bền $A\ge A_{\mathrm{strength}}(x)=A_W(|M|/[\sigma])$ và $A\ge A_{\min}$: nghiệm KKT là $A=\max(A_{\mathrm{strength}},A_{\min},c\,g^{1/(n+1)})$, $g=\max(M\bar m,0)$; hằng số $c$ giải từ $\delta(c)=\delta_{\max}$.
- Dầm siêu tĩnh: $M$ phụ thuộc $A$ nên không có dạng đóng → dùng FEM + OC (mục 4).
""")
    st.subheader("3. Cột — cực đại tải tới hạn với thể tích cho trước (Lagrange – Keller – Tadjbakhsh)")
    st.latex(r"P_{cr}=\min_w\frac{\int E\alpha A^{n}w''^2\,\mathrm dx}{\int w'^2\,\mathrm dx},\quad \int A\,\mathrm dx=V")
    st.latex(r"\delta_A:\ \ n\,E\alpha A^{n-1}w''^{2}=\lambda\ \ \ (\text{algebraic}),\qquad M=-Pw\ \Rightarrow\ A\propto|w|^{\frac2{n+1}}")
    st.latex(r"w''=-\kappa\,|w|^{\frac{1-n}{n+1}}\,\mathrm{sgn}\,w\ \xrightarrow{\ u=A/A_{\max}\ }\ \xi=I_u\!\Big(\tfrac{n+1}2,\tfrac12\Big)\ \Rightarrow\ u(\xi)=I^{-1}_{\xi}\!\Big(\tfrac{n+1}2,\tfrac12\Big)")
    st.latex(r"\frac{\bar A}{A_{\max}}=\frac{n+1}{n+2},\qquad P_{cr}^*=\frac{(n+1)\,B^{2}\!\big(\tfrac{n+1}2,\tfrac12\big)\,E\alpha\,A_{\max}^{\,n}}{L_{ef}^{2}}")
    st.markdown(r"Với $n=2$, hai đầu khớp: $P^*_{cr}/P_{cr,\mathrm{uniform}}=4/3$ — **kết quả cổ điển của Keller (1960)**, được dùng làm test kiểm chứng. "
                r"Nghiệm E–L cho $A\to0$ tại điểm mômen bằng 0; thực tế phải áp cận dưới $A_{\min}$.")
    st.subheader("4. Phần số: tiêu chuẩn tối ưu (OC) và KKT trên lưới FEM")
    st.latex(r"\min_{A}\ W=\rho\sum A_eL_e\quad\text{s.t.}\quad \frac{|N_e|}{A_e\sigma_N}+\frac{|M_e|}{W_e[\sigma]}\le1,\ \ |\delta_k|\le\delta_k^{\lim}")
    st.latex(r"\nabla_A\mathcal L=\nabla_A W+\sum_j\mu_j\nabla_A g_j+\sum_k\lambda_k\nabla_A h_k=\mathbf 0,\quad g_j\leq0,\quad\mu_j\geq0,\quad\mu_jg_j=0")
    st.latex(r"\frac{\partial\delta_k}{\partial A_e}=-\,\mathbf u_{unit}^{T}\frac{\partial\mathbf K}{\partial A_e}\mathbf u")
    st.markdown("""
Điều kiện KKT gồm: (1) dừng; (2) khả thi nguyên thủy; (3) khả thi đối ngẫu; (4) bù trừ. KKT là điều kiện cần khi thỏa điều kiện chính quy; chỉ kết luận cực tiểu toàn cục khi có thêm giả thiết thích hợp, chẳng hạn tính lồi.

Trong ứng dụng: FEM phân tích; OC/FSD tạo nghiệm; SLSQP tinh chỉnh các ràng buộc chuyển vị với độ nhạy liên hợp. Báo cáo kiểm tra khả thi và một chẩn đoán dừng KKT xấp xỉ trên phần tử tự do; chẩn đoán đó **không thay thế kiểm tra đầy đủ mọi nhân tử và điều kiện bù trừ**. Đối chiếu khung với `fem_core.py` là kiểm tra chéo FEM, không phải kiểm tra KKT.
""")
    st.info("**Giới hạn:** Euler–Bernoulli, đàn hồi tuyến tính, bỏ qua biến dạng cắt, ổn định cục bộ, P-Δ, mỏi, tải bản thân và chi tiết liên kết. "
            "Tiết diện thực phải làm tròn theo thép hình rồi kiểm tra lại theo TCVN/AISC. Nghiệm với kết cấu siêu tĩnh là cực trị địa phương.")


# ══════════════════════════════════════════════════════
#  TAB 1 — CỘT
# ══════════════════════════════════════════════════════
def tab_column():
    st.header("CHƯƠNG 06 · Cột chịu nén biến tiết diện")
    left, right = st.columns([1, 2.2])
    with left:
        bc = st.selectbox("Sơ đồ liên kết", list(COLUMN_BCS), format_func=COLUMN_BCS.get, key="c_bc")
        L = st.number_input("Chiều dài L (m)", 0.5, 100.0, 5.0, 0.5, key="c_L")
        E = st.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="c_E") * 1e6
        fam = family_selector("c")
        mode = st.radio("Bài toán", ["Cho thể tích → cực đại P_cr", "Cho P yêu cầu → cực tiểu thể tích"], key="c_mode")
        if mode.startswith("Cho thể"):
            A0 = st.number_input("Diện tích trung bình A₀ = V/L (cm²)", 1.0, 5000.0, 30.0, 1.0, key="c_A0")
            P_req = None
        else:
            P_req = st.number_input("P yêu cầu (kN)", 1.0, 1e7, 500.0, 10.0, key="c_P")
        floor = st.slider("Cận dưới A_min (% của A_max)", 0.0, 20.0, 1.0, 0.5, key="c_floor") / 100.0
        run_num = st.checkbox("Chạy thêm tối ưu số OC (kiểm chứng nghiệm đóng)", True, key="c_num")
        go_btn = st.button("🚀 Tối ưu cột", type="primary", key="c_go")
    if go_btn:
        try:
            with st.spinner("Đang tính..."):
                if P_req is None:
                    st.session_state["c_res"] = optimize_column(L, fam, E, bc, V=A0 * 1e-4 * L, floor_frac=floor,
                                                                run_numeric=run_num)
                else:
                    st.session_state["c_res"] = optimize_column(L, fam, E, bc, P_req=P_req, floor_frac=floor,
                                                                run_numeric=run_num)
        except Exception as ex:
            st.error(f"Lỗi: {ex}")
    r = st.session_state.get("c_res")
    with right:
        if r is None:
            st.info("Nhập thông số rồi bấm **Tối ưu cột**.")
            return
        metrics([
            ("P_cr tối ưu (FEM)", f"{r.P_fem_opt:,.1f} kN"),
            ("P_cr cột đều", f"{r.P_uniform:,.1f} kN"),
            ("Hệ số tăng tải (FEM)", f"×{r.ratio_fem:.3f}", "metric-good"),
            ("Hệ số tăng tải (đóng, không cận)", f"×{r.ratio_closed:.3f}"),
            ("V = ∫A dx", f"{r.V * 1e6:,.0f} cm²·m→ {r.V:.5f} m³"),
            ("A_max", f"{np.max(r.A_opt) * 1e4:.2f} cm²"),
        ])
        if getattr(r, "bimodal", False):
            st.info("**Cột hai đầu ngàm:** nghiệm đóng đơn-mode (A = 0 tại L/4 và 3L/4) biến cột thành cơ cấu nên tải FEM thực "
                    "thấp hơn giá trị đóng (thậm chí thấp hơn cột đều). Profile hiển thị là nghiệm **bimodal** (λ₁ = λ₂) "
                    "giải bằng SLSQP; hệ số \"đóng\" chỉ là cận tham khảo.")
        fig = go.Figure()
        fig.add_scatter(x=r.x / r.L, y=r.A_opt * 1e4, name=("Bimodal SLSQP (FEM kiểm)" if getattr(r, "bimodal", False) else "E–L nghiệm đóng (FEM kiểm)"), line=dict(color=BLUE, width=3))
        if r.A_numeric is not None:
            fig.add_scatter(x=r.x_numeric / r.L, y=r.A_numeric * 1e4, name="OC số", mode="lines",
                            line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=[0, 1], y=[r.A_uniform * 1e4] * 2, name="Cột đều", line=dict(color=GREY, dash="dot"))
        st.plotly_chart(fig_layout(fig, "Diện tích tối ưu A(x)", 360, "x/L", "A (cm²)"), width="stretch")
        sec_pos = st.columns([0.8, 1.2])[0]
        with sec_pos:
            section_x_ratio = st.slider("Vị trí mặt cắt x/L", 0.0, 1.0, 0.5, 0.01, key="c_section_x")
            section_area = float(np.interp(section_x_ratio, r.x / r.L, r.A_opt))
            st.metric("A tại vị trí chọn", f"{section_area * 1e4:.2f} cm²")
        ref_mode = st.radio("Mốc so sánh (nét đứt)", ["Cột đều cùng P_cr (vật liệu cần để chịu cùng tải)",
                                                     "Cột đều cùng thể tích V"], horizontal=True, key="c_ref")
        if ref_mode.startswith("Cột đều cùng P_cr"):
            A_ref = r.A_uniform * (r.P_fem_opt / r.P_uniform) ** (1.0 / r.fam.n)   # P_cr ∝ A^n
            ref_lab = "Trước tối ưu: cột đều, cùng P_cr"
            note = (f"Cột đều chịu cùng P_cr = {r.P_fem_opt:,.0f} kN cần V = {A_ref * r.L * 1e6:,.0f} cm²·m; "
                    f"cột tối ưu chỉ cần {r.V * 1e6:,.0f} cm²·m → **tiết kiệm {(1 - r.V / (A_ref * r.L)) * 100:.1f}% vật liệu**.")
        else:
            A_ref = r.A_uniform
            ref_lab = "Trước tối ưu: cột đều, cùng thể tích"
            note = (f"Cùng thể tích, cột tối ưu chịu P_cr = {r.P_fem_opt:,.0f} kN so với {r.P_uniform:,.0f} kN của cột đều "
                    f"(**×{r.ratio_fem:.3f}**).")
        show_section_comparison(r.fam, A_ref, section_area, "c_section_comparison")
        show_section_3d(r.fam, r.L, A_ref, r.A_opt, "c_section_3d", orientation="vertical")
        st.plotly_chart(fig_before_after_member(r.x, A_ref, r.A_opt, r.fam, ref_lab,
                                                "Hình dạng cột — trước (nét đứt) so với sau tối ưu (nét liền)", "x (m)", 340),
                        width="stretch")
        st.caption(note + " Vùng **xanh lá** = vật liệu bỏ được; vùng **cam** = chỗ phải dày hơn mốc (vật liệu được dồn về "
                          "nơi mômen lớn, nên cột tối ưu luôn phình ra ngoài mốc ở giữa nhịp). "
                          f"Bề rộng vẽ ∝ A^{_shape_exp(r.fam):g}.")
        if r.history:
            cv = go.Figure()
            cv.add_scatter(y=np.array(r.history) / r.P_closed, name="P_cr(OC)/P_closed", line=dict(color=ORANGE))
            cv.add_hline(y=1.0, line_dash="dot")
            st.plotly_chart(fig_layout(cv, "Hội tụ OC so với nghiệm đóng", 300, "vòng lặp", "P/P*"), width="stretch")
        report_block(r.report, "cot_toi_uu", "Thuyết minh — Tối ưu cột")


# ══════════════════════════════════════════════════════
#  TAB 2 — DẦM
# ══════════════════════════════════════════════════════
def tab_beam():
    st.header("CHƯƠNG 07 · Dầm vút thon A(x)")
    left, right = st.columns([1.1, 2.2])
    with left:
        bc = st.selectbox("Sơ đồ", list(BEAM_BCS), format_func=BEAM_BCS.get, key="b_bc")
        L = st.number_input("Chiều dài L (m)", 1.0, 100.0, 8.0, 0.5, key="b_L")
        c1, c2 = st.columns(2)
        E = c1.number_input("E (GPa)", 1.0, 400.0, STEEL_E / 1e6, 5.0, key="b_E") * 1e6
        sig = c2.number_input("[σ] (MPa)", 10.0, 1000.0, STEEL_SIGMA / 1e3, 10.0, key="b_sig") * 1e3
        fam = family_selector("b", default="Chữ I (cánh mỏng, h cố định)")
        st.caption("Tải trọng (hướng xuống > 0)")
        pl_df = st.data_editor(pd.DataFrame({"P (kN)": [100.0], "x (m)": [L / 2]}), num_rows="dynamic",
                               key="b_pl", hide_index=True, width="stretch")
        udl_df = st.data_editor(pd.DataFrame({"q (kN/m)": [10.0], "x1 (m)": [0.0], "x2 (m)": [L]}), num_rows="dynamic",
                                key="b_udl", hide_index=True, width="stretch")
        use_d = st.checkbox("Ràng buộc độ võng", True, key="b_usd")
        d1, d2 = st.columns(2)
        dmax = d1.number_input("|δ| max (mm)", 0.1, 1000.0, round(L * 1000 / 400, 1), 1.0, key="b_dmax") / 1e3
        xc = d2.number_input("Vị trí khống chế x (m)", 0.0, L, L / 2 if bc != "cantilever" else L, 0.5, key="b_xc")
        amin = st.number_input("A_min (cm²)", 0.01, 1000.0, 5.0, 0.5, key="b_amin") * 1e-4
        nel = st.slider("Số phần tử lưới", 40, 200, 100, 10, key="b_nel")
        go_btn = st.button("🚀 Tối ưu dầm", type="primary", key="b_go")
    if go_btn:
        try:
            pls = clean_rows(pl_df, 2)
            udls = clean_rows(udl_df, 3)
            for P, x in pls:
                if not 0 <= x <= L:
                    raise ValueError(f"Tải tập trung tại x={x} nằm ngoài [0, {L}].")
            for q, a, b in udls:
                if not (0 <= a < b <= L):
                    raise ValueError(f"UDL cần 0 ≤ x1 < x2 ≤ {L}.")
            if not pls and not udls:
                raise ValueError("Chưa có tải trọng.")
            prob = BeamProblem(L=L, bc=bc, udls=udls, point_loads=pls, fam=fam, E=E, sigma_a=sig,
                               delta_max=dmax if use_d else None, x_ctrl=xc, A_min=amin, n_elem=nel)
            with st.spinner("Đang tối ưu (FEM + KKT)..."):
                st.session_state["b_res"] = optimize_beam(prob)
        except Exception as ex:
            st.error(f"Lỗi: {ex}")
    r = st.session_state.get("b_res")
    with right:
        if r is None:
            st.info("Nhập thông số rồi bấm **Tối ưu dầm**.")
            return
        sav = (1 - r.vol_opt / r.vol_uniform) * 100
        items = [("Thể tích tối ưu", f"{r.vol_opt:.4f} m³"), ("Thể tích dầm đều", f"{r.vol_uniform:.4f} m³"),
                 ("Tiết kiệm vật liệu", f"{sav:.1f}%", "metric-good")]
        if r.vol_closed is not None:
            items.append(("Nghiệm đóng E–L", f"{r.vol_closed:.4f} m³"))
        if r.delta_ctrl_opt is not None:
            items.append(("δ tại điểm khống chế", f"{r.delta_ctrl_opt * 1e3:.2f} / {r.prob.delta_max * 1e3:.2f} mm"))
        items.append(("Ràng buộc", "Thoả" if r.opt.feasible else "KHÔNG thoả", "metric-good" if r.opt.feasible else "metric-bad"))
        metrics(items)
        fig = go.Figure()
        xs_step = np.repeat(r.xs, 2)[1:-1]
        fig.add_scatter(x=xs_step, y=np.repeat(r.A_opt, 2) * 1e4, name="FEM + KKT", line=dict(color=BLUE, width=3))
        if r.closed is not None:
            fig.add_scatter(x=r.closed["x"], y=r.closed["A"] * 1e4, name="Nghiệm đóng E–L", line=dict(color=ORANGE, dash="dash"))
        fig.add_scatter(x=[0, r.prob.L], y=[r.A_uniform * 1e4] * 2, name="Dầm đều", line=dict(color=GREY, dash="dot"))
        st.plotly_chart(fig_layout(fig, "Diện tích tối ưu A(x)", 360, "x (m)", "A (cm²)"), width="stretch")
        sec_pos = st.columns([0.8, 1.2])[0]
        with sec_pos:
            section_x = st.slider("Vị trí mặt cắt x (m)", 0.0, r.prob.L, r.prob.L / 2,
                                  max(r.prob.L / 100, 0.01), key="b_section_x")
            section_area = float(np.interp(section_x, r.x_mid, r.A_opt))
            st.metric("A tại vị trí chọn", f"{section_area * 1e4:.2f} cm²")
        show_section_comparison(r.prob.fam, r.A_uniform, section_area, "b_section_comparison")
        show_section_3d(r.prob.fam, r.prob.L, r.A_uniform, r.A_opt, "b_section_3d")
        xs_step = np.repeat(r.xs, 2)[1:-1]
        st.plotly_chart(fig_before_after_member(xs_step, r.A_uniform, np.repeat(r.A_opt, 2), r.prob.fam,
                                                "Trước tối ưu: dầm đều (cùng ràng buộc)",
                                                "Hình dạng dầm — trước (nét đứt) so với sau tối ưu (nét liền)", "x (m)", 340),
                        width="stretch")
        st.caption(f"Dầm đều thoả cùng ràng buộc cần A = {r.A_uniform * 1e4:.2f} cm² (V = {r.vol_uniform:.4f} m³); "
                   f"dầm tối ưu V = {r.vol_opt:.4f} m³ → **tiết kiệm {sav:.1f}%**. Xanh lá = vật liệu bỏ được, "
                   f"cam = chỗ dày hơn mốc. Chiều cao vẽ ∝ A^{_shape_exp(r.prob.fam):g}.")
        st.subheader("Sơ đồ tải trọng · lực cắt · mômen · chuyển vị")
        fld = vz.beam_fields(r)
        rc = vz.beam_reactions(r, fld)
        if rc:
            metrics([(k, v) for k, v in rc])
        c1, c2 = st.columns(2)
        c1.plotly_chart(vz.plot_beam_loads(r.prob), width="stretch")
        c2.plotly_chart(vz.plot_beam_sfd(fld), width="stretch")
        c3, c4 = st.columns(2)
        c3.plotly_chart(vz.plot_beam_bmd(fld), width="stretch")
        c4.plotly_chart(vz.plot_beam_elastic(r.prob, fld), width="stretch")
        st.caption("Nét đứt trắng = dầm đều (trước tối ưu, cùng ràng buộc); nét liền = dầm vút thon sau tối ưu. "
                   "Quy ước: tải xuống > 0; M > 0 căng thớ dưới (vẽ về phía dưới); V > 0 khi phần bên trái đẩy lên; "
                   "độ võng hướng xuống > 0. Với dầm tĩnh định, V và M của hai phương án trùng nhau (chỉ độ võng khác).")
        report_block(r.report, "dam_toi_uu", "Thuyết minh — Tối ưu dầm vút thon")


# ══════════════════════════════════════════════════════
#  TAB 3 — KHUNG / DÀN
# ══════════════════════════════════════════════════════
@st.cache_resource
def _canvas_component():
    return components.declare_component("dop_canvas", path=os.path.join(HERE, "canvas"))


def _sup(node, kind):
    t = {"pin": (True, True, False), "rollx": (False, True, False), "fixed": (True, True, True)}[kind]
    return dict(node=node, ux=t[0], uy=t[1], rz=t[2])


def _truss_warren():
    nodes = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (1.5, 2.5), (4.5, 2.5), (7.5, 2.5), (10.5, 2.5)]
    mem = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8),
           (0, 5), (5, 1), (1, 6), (6, 2), (2, 7), (7, 3), (3, 8), (8, 4)]
    return dict(mode="Dàn (nút khớp)", nodes=nodes, members=mem, supports=[_sup(0, "pin"), _sup(4, "rollx")],
                nodal=[(5, 0, -50), (6, 0, -50), (7, 0, -50), (8, 0, -50)], udl=[], limits=[(2, "uy", 30.0)], seg=1)


def _truss_ten_bar():
    inp = ten_bar_truss_example()
    return dict(
        mode="Dàn (nút khớp)",
        nodes=inp.nodes,
        members=inp.members,
        supports=[_sup(node, "pin") for node, _, _, _ in inp.supports],
        nodal=[(node, fx, fy) for node, fx, fy, _ in inp.loads],
        udl=[],
        limits=[(node, direction, limit * 1e3) for node, direction, limit in inp.disp_limits],
        seg=1,
        material=dict(
            E=inp.E / 1e6,
            sigma=inp.sigma_a / 1e3,
            rho=inp.rho,
            amin=inp.A_min * 1e4,
            amax=inp.A_max * 1e4,
            buck=False,
            k=inp.k_eff,
            fs=inp.FS_buckling,
            family="Ống tròn mỏng",
            tube_ratio=0.05,
        ),
        note="Benchmark slide: nhịp 360 in, tải 100 kip tại hai nút đáy; giới hạn chuyển vị 2 in.",
    )


def _frame_portal():
    nodes = [(0, 0), (0, 4), (3, 5), (6, 4), (6, 0)]
    mem = [(0, 1), (1, 2), (2, 3), (3, 4)]
    return dict(mode="Khung phẳng (nút cứng)", nodes=nodes, members=mem,
                supports=[_sup(0, "fixed"), _sup(4, "fixed")], nodal=[(1, 15, 0)],
                udl=[(1, 20.0), (2, 20.0)], limits=[(2, "uy", 20.0)], seg=4)


def _frame_two_story():
    nodes = [(0, 0), (6, 0), (0, 3.5), (6, 3.5), (0, 7), (6, 7)]
    mem = [(0, 2), (2, 4), (1, 3), (3, 5), (2, 3), (4, 5)]
    return dict(mode="Khung phẳng (nút cứng)", nodes=nodes, members=mem,
                supports=[_sup(0, "fixed"), _sup(1, "fixed")], nodal=[(2, 20, 0), (4, 40, 0)],
                udl=[(4, 30.0), (5, 30.0)], limits=[(4, "ux", 35.0)], seg=4)


PRESETS = {"Dàn Warren 12 m": _truss_warren, "Dàn 10 thanh chuẩn (slide)": _truss_ten_bar,
           "Khung cổng 1 nhịp (mái dốc)": _frame_portal, "Khung 2 tầng 1 nhịp": _frame_two_story}


def _load_preset(name):
    p = PRESETS[name]()
    st.session_state["geo"] = dict(nodes=[dict(x=float(x), y=float(y)) for x, y in p["nodes"]],
                                   elements=[dict(i=i, j=j) for i, j in p["members"]], supports=p["supports"])
    st.session_state["s_mode"] = p["mode"]
    st.session_state["s_seg"] = p["seg"]
    if "material" in p:
        material = p["material"]
        st.session_state["s_E"] = material["E"]
        st.session_state["s_sig"] = material["sigma"]
        st.session_state["s_rho"] = material["rho"]
        st.session_state["s_amin"] = material["amin"]
        st.session_state["s_amax"] = material["amax"]
        st.session_state["s_buck"] = material["buck"]
        st.session_state["s_k"] = material["k"]
        st.session_state["s_fs"] = material["fs"]
        st.session_state["s_fam"] = material["family"]
        st.session_state["s_t"] = material["tube_ratio"]
    st.session_state["nodal0"] = pd.DataFrame(p["nodal"], columns=["Nút", "Fx (kN)", "Fy (kN)"]) if p["nodal"] else \
        pd.DataFrame({"Nút": [0], "Fx (kN)": [0.0], "Fy (kN)": [0.0]})
    st.session_state["udl0"] = pd.DataFrame(p["udl"], columns=["Thanh", "w (kN/m, hướng xuống)"]) if p["udl"] else \
        pd.DataFrame({"Thanh": [0], "w (kN/m, hướng xuống)": [0.0]})
    st.session_state["lim0"] = pd.DataFrame(p["limits"], columns=["Nút", "Phương", "|δ| max (mm)"]) if p["limits"] else \
        pd.DataFrame({"Nút": [0], "Phương": ["uy"], "|δ| max (mm)": [30.0]})
    st.session_state["s_token"] = st.session_state.get("s_token", 0) + 1
    st.session_state.pop("s_res", None)


def _init_struct_state():
    if "geo" not in st.session_state:
        _load_preset("Dàn Warren 12 m")


def build_struct_input(geo, mode_label, fam, E, sig, rho, amin, amax, buck, keff, fs, seg, nodal_df, udl_df, lim_df):
    nodes = [(n["x"], n["y"]) for n in geo["nodes"]]
    members = [(e["i"], e["j"]) for e in geo["elements"]]
    if len(nodes) < 2 or not members:
        raise ValueError("Cần ít nhất 2 nút và 1 thanh — hãy vẽ trên canvas hoặc chọn mẫu.")
    is_frame = mode_label.startswith("Khung")
    sup = []
    for s in geo["supports"]:
        sup.append((int(s["node"]), bool(s["ux"]), bool(s["uy"]), bool(s["rz"]) if is_frame else False))
    if not sup:
        raise ValueError("Chưa có gối tựa (chế độ 🔒 Gối tựa: bấm vào nút để đổi loại gối).")
    loads = {}
    for k, fx, fy in clean_rows(nodal_df, 3, int_cols=(0,)):
        if not 0 <= k < len(nodes):
            raise ValueError(f"Tải nút: nút {k} không tồn tại.")
        a = loads.setdefault(k, [0.0, 0.0, 0.0]); a[0] += fx; a[1] += fy
    member_q = [0.0] * len(members)
    for m, w in clean_rows(udl_df, 2, int_cols=(0,)):
        if not 0 <= m < len(members):
            raise ValueError(f"Tải phân bố: thanh {m} không tồn tại.")
        if w == 0:
            continue
        i, j = members[m]
        dx, dy = nodes[j][0] - nodes[i][0], nodes[j][1] - nodes[i][1]
        Lm = math.hypot(dx, dy); c, s = dx / Lm, dy / Lm
        if is_frame:
            member_q[m] += -w * c                     # thành phần vuông góc trục thanh (trục y cục bộ)
            fax = -w * s * Lm / 2.0                   # thành phần dọc trục → gộp 1/2 vào mỗi đầu thanh
            for nd in (i, j):
                a = loads.setdefault(nd, [0.0, 0.0, 0.0]); a[0] += fax * c; a[1] += fax * s
        else:
            for nd in (i, j):
                a = loads.setdefault(nd, [0.0, 0.0, 0.0]); a[1] += -w * Lm / 2.0
    if not any(abs(v) > 0 for a in loads.values() for v in a) and not any(member_q):
        raise ValueError("Chưa có tải trọng.")
    lims = []
    for k, d, v in [(int(r.iloc[0]), str(r.iloc[1]), float(r.iloc[2])) for _, r in lim_df.iterrows()
                    if not (pd.isna(r.iloc[0]) or pd.isna(r.iloc[1]) or pd.isna(r.iloc[2]))]:
        if not 0 <= k < len(nodes):
            raise ValueError(f"Giới hạn chuyển vị: nút {k} không tồn tại.")
        if d not in ("ux", "uy"):
            raise ValueError("Phương chuyển vị phải là ux hoặc uy.")
        if v > 0:
            lims.append((k, d, v / 1e3))
    return StructInput(nodes=nodes, members=members, supports=sup,
                       loads=[(k, a[0], a[1], a[2]) for k, a in loads.items()], member_q=member_q,
                       mode="frame" if is_frame else "truss", fam=fam, E=E, sigma_a=sig, rho=rho,
                       FS_buckling=fs, k_eff=keff, check_buckling=buck, A_min=amin, A_max=amax,
                       disp_limits=lims, segments=seg,
                       user_nodal=[(int(k), float(fx), float(fy), 0.0) for k, fx, fy in clean_rows(nodal_df, 3, int_cols=(0,))],
                       user_udl=[(int(m), float(wv)) for m, wv in clean_rows(udl_df, 2, int_cols=(0,)) if wv != 0])


def plot_structure(inp: StructInput, area_member=None, util_member=None, title="Hình học kết cấu"):
    fig = go.Figure()
    xs = [p[0] for p in inp.nodes]; ys = [p[1] for p in inp.nodes]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    vals = area_member
    vmin, vmax = (float(np.min(vals)), float(np.max(vals))) if vals is not None else (0, 1)
    for k, (i, j) in enumerate(inp.members):
        if vals is None:
            col, w = BLUE, 4
        else:
            t = 0.0 if vmax - vmin < 1e-12 else (vals[k] - vmin) / (vmax - vmin)
            col = pc.sample_colorscale("Turbo", 0.1 + 0.8 * t)[0]
            w = 2 + 12 * vals[k] / vmax
        fig.add_scatter(x=[inp.nodes[i][0], inp.nodes[j][0]], y=[inp.nodes[i][1], inp.nodes[j][1]], mode="lines",
                        line=dict(color=col, width=w), hoverinfo="text", showlegend=False,
                        text=f"Thanh {k}: " + (f"A = {vals[k] * 1e4:.2f} cm²" if vals is not None else ""))
        mx, my = (inp.nodes[i][0] + inp.nodes[j][0]) / 2, (inp.nodes[i][1] + inp.nodes[j][1]) / 2
        fig.add_annotation(x=mx, y=my, text=f"{k}" if vals is None else f"{vals[k] * 1e4:.1f}", showarrow=False,
                           font=dict(size=10, color="#333"), bgcolor="rgba(255,255,255,.75)")
    _add_nodes_supports_loads(fig, inp, span, xs, ys)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    pad = 0.12 * span
    fig.update_xaxes(range=[min(xs) - pad, max(xs) + pad]); fig.update_yaxes(range=[min(ys) - pad, max(ys) + 2 * pad])
    return fig_layout(fig, title, 520)


def _add_nodes_supports_loads(fig, inp: StructInput, span, xs, ys):
    fig.add_scatter(x=xs, y=ys, mode="markers+text", text=[str(i) for i in range(len(xs))], textposition="top center",
                    marker=dict(size=8, color="white", line=dict(color=BLUE, width=2)), showlegend=False, name="Nút")
    sx = [inp.nodes[s[0]][0] for s in inp.supports]; sy = [inp.nodes[s[0]][1] for s in inp.supports]
    fig.add_scatter(x=sx, y=sy, mode="markers", marker=dict(symbol="triangle-up", size=14, color=RED),
                    showlegend=False, name="Gối")
    for node, fx, fy, mz in inp.loads:
        px, py = inp.nodes[node]
        mag = math.hypot(fx, fy)
        if mag > 0:
            L0 = 0.12 * span
            fig.add_annotation(x=px, y=py, ax=px - fx / mag * L0, ay=py - fy / mag * L0, xref="x", yref="y",
                               axref="x", ayref="y", showarrow=True, arrowhead=3, arrowcolor=GREEN, arrowwidth=2,
                               text=f"{mag:.0f} kN", font=dict(color=GREEN, size=10))


def tab_struct():
    _init_struct_state()
    st.header("CHƯƠNG 03 · Tối ưu giàn / khung")
    tok = st.session_state["s_token"]

    top = st.columns([2, 1.3, 3])
    pre = top[0].selectbox("Mẫu có sẵn", list(PRESETS), key="s_preset")
    top[1].write(""); top[1].write("")
    top[1].button("📥 Nạp mẫu", on_click=lambda: _load_preset(st.session_state["s_preset"]), key="s_loadbtn")
    mode_label = top[2].radio("Mô hình liên kết nút", ["Dàn (nút khớp)", "Khung phẳng (nút cứng)"], horizontal=True,
                              key="s_mode")
    is_frame = mode_label.startswith("Khung")

    st.markdown("**Vẽ kết cấu** — kéo chuột để vẽ thanh; chế độ 🔒 bấm vào nút để đổi gối (ngàm → khớp → con lăn → bỏ).")
    geo = st.session_state["geo"]
    val = _canvas_component()(nodes=geo["nodes"], elements=geo["elements"], supports=geo["supports"], snap=0.5,
                              reset_token=tok, height=460, key=f"dop_canvas_{tok}", default=None)
    if val:
        geo = dict(nodes=val["nodes"], elements=val["elements"], supports=val["supports"])
        st.session_state["geo"] = geo
    st.caption(f"{len(geo['nodes'])} nút · {len(geo['elements'])} thanh · {len(geo['supports'])} gối")
    if st.session_state.get("s_preset") == "Dàn 10 thanh chuẩn (slide)":
        st.image(
            os.path.join(HERE, "assets", "ten_bar_toklu_ozbasaran.png"),
            caption="Nguồn: An efficient decoder for size optimization of trusses with automatic member grouping by Y. C. Toklu, H. Ozbasaran.",
            use_container_width=True,
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Tải tập trung tại nút (kN)")
        nodal_df = st.data_editor(st.session_state["nodal0"], num_rows="dynamic", key=f"s_nodal_{tok}", hide_index=True,
                                  width="stretch")
    with c2:
        st.caption("Tải phân bố đứng trên thanh (kN/m theo chiều dài thanh)")
        udl_df = st.data_editor(st.session_state["udl0"], num_rows="dynamic", key=f"s_udl_{tok}", hide_index=True,
                                width="stretch")
    with c3:
        st.caption("Ràng buộc chuyển vị (mm)")
        lim_df = st.data_editor(st.session_state["lim0"], num_rows="dynamic", key=f"s_lim_{tok}", hide_index=True,
                                width="stretch",
                                column_config={"Phương": st.column_config.SelectboxColumn(options=["ux", "uy"], required=True)})

    with st.expander("⚙️ Vật liệu · họ tiết diện · ổn định", expanded=False):
        a, b, c = st.columns(3)
        st.session_state.setdefault("s_E", STEEL_E / 1e6)
        st.session_state.setdefault("s_sig", STEEL_SIGMA / 1e3)
        st.session_state.setdefault("s_rho", STEEL_RHO)
        E = a.number_input("E (GPa)", 1.0, 400.0, step=5.0, key="s_E") * 1e6
        sig = b.number_input("[σ] (MPa)", 10.0, 1000.0, step=10.0, key="s_sig") * 1e3
        rho = c.number_input("ρg (kN/m³)", 1.0, 500.0, step=0.5, key="s_rho")
        fam = family_selector("s", default="Ống tròn mỏng")
        a, b, c, d = st.columns(4)
        st.session_state.setdefault("s_amin", 2.0)
        st.session_state.setdefault("s_amax", 500.0)
        st.session_state.setdefault("s_buck", True)
        st.session_state.setdefault("s_k", 1.0)
        st.session_state.setdefault("s_fs", 1.0)
        amin = a.number_input("A_min (cm²)", 0.01, 500.0, step=0.5, key="s_amin") * 1e-4
        amax = b.number_input("A_max (cm²)", 1.0, 20000.0, step=10.0, key="s_amax") * 1e-4
        buck = c.checkbox("Ổn định Euler khi nén", key="s_buck")
        keff = d.number_input("Hệ số chiều dài k", 0.5, 2.5, step=0.1, key="s_k")
        fs = a.number_input("FS ổn định", 1.0, 3.0, step=0.1, key="s_fs")
        st.session_state.setdefault("s_seg", 4)      # giá trị gán qua session_state, không truyền value= (tránh cảnh báo)
        seg = b.number_input("Số đoạn chia / thanh (khung)", 1, 8, step=1, key="s_seg",
                             help="Chia nhỏ để A thay đổi dọc thanh (vút thon trong khung).")

    is_ten_bar = st.session_state.get("s_preset") == "Dàn 10 thanh chuẩn (slide)" and not is_frame
    with st.expander("🧬 Thuật toán cho dàn 10 thanh", expanded=is_ten_bar):
        algo = st.radio(
            "Chế độ chạy",
            ["OC/FSD + SLSQP", "GA 10 thanh", "So sánh cả hai"],
            horizontal=True,
            key="s_algorithm",
            disabled=not is_ten_bar,
            help="GA trong file toi_uu_gian_10_thanh_GA_v2.py chỉ áp dụng cho benchmark dàn 10 thanh chuẩn.",
        )
        ga_cols = st.columns(4)
        st.session_state.setdefault("ga_pop", 100)
        st.session_state.setdefault("ga_gen", 500)
        st.session_state.setdefault("ga_runs", 5)
        st.session_state.setdefault("ga_seed", 0)
        ga_pop = ga_cols[0].number_input("Quần thể", 20, 500, step=10, key="ga_pop", disabled=not is_ten_bar)
        ga_gen = ga_cols[1].number_input("Số thế hệ", 50, 2000, step=50, key="ga_gen", disabled=not is_ten_bar)
        ga_runs = ga_cols[2].number_input("Số lần chạy", 1, 20, step=1, key="ga_runs", disabled=not is_ten_bar)
        ga_seed = ga_cols[3].number_input("Seed đầu", 0, 100000, step=1, key="ga_seed", disabled=not is_ten_bar)
        ga_polish = st.checkbox("Kiểm chứng lại nghiệm GA bằng SLSQP", value=True, key="ga_polish", disabled=not is_ten_bar)
        if not is_ten_bar:
            st.caption("Chọn preset “Dàn 10 thanh chuẩn (slide)” và mô hình dàn để bật GA.")

    if st.button("🚀 Tối ưu toàn kết cấu", type="primary", key="s_go"):
        try:
            inp = build_struct_input(geo, mode_label, fam, E, sig, rho, amin, amax, buck, keff, fs, int(seg),
                                     nodal_df, udl_df, lim_df)
            with st.spinner("Đang tối ưu (FEM + OC + KKT)..."):
                st.session_state["s_res"] = optimize_frame(inp)
            st.session_state.pop("s_ga_res", None)
            if is_ten_bar and algo in ("GA 10 thanh", "So sánh cả hai"):
                with st.spinner("Đang chạy thuật toán di truyền GA cho dàn 10 thanh..."):
                    st.session_state["s_ga_res"] = _run_ten_bar_ga(ga_pop, ga_gen, ga_runs, ga_seed, ga_polish)
        except Exception as ex:
            st.session_state.pop("s_res", None)
            st.session_state.pop("s_ga_res", None)
            st.error(f"Lỗi: {ex}")

    res = st.session_state.get("s_res")
    if res is None:
        st.plotly_chart(_preview_fig(geo, mode_label), width="stretch")
        return
    o = res.opt
    volume_opt = float(np.sum(o.A * res.model.L))
    volume_uniform = float(o.A_uniform * np.sum(res.model.L))
    metrics([
        ("Thể tích tối ưu", f"{volume_opt:.4f} m³"),
        ("Thể tích tiết diện đều", f"{volume_uniform:.4f} m³"),
        ("Khối lượng tối ưu", f"{weight_to_mass_kg(res.weight):,.1f} kg"),
        ("Khối lượng tiết diện đều (mốc)", f"{weight_to_mass_kg(res.weight_uniform):,.1f} kg"),
        ("Tiết kiệm", f"{res.saving * 100:.1f}%", "metric-good"),
        ("U lớn nhất", f"{float(np.max(o.util)):.3f}"),
        ("Hội tụ / thoả", f"{'✔' if o.converged else '…'} / {'✔' if o.feasible else '✘'}",
         "metric-good" if o.feasible else "metric-bad"),
        ("Dừng KKT (xấp xỉ)", f"{o.EL_cv:.1%}" if o.EL_cv is not None else "—"),
    ])
    result_view = st.selectbox("Xem kết quả", ["Tổng quan trước / sau", "Chuyển vị", "Nội lực",
                                               "Sơ đồ tải", "Tiết diện tối ưu", "Bảng thanh",
                                               "Hội tụ", "Mặt cắt"], key="s_result_view")
    if result_view == "Tổng quan trước / sau":
        st.plotly_chart(plot_structure_before_after(res), width="stretch")
        st.caption(f"Nét đứt: tiết diện đều A = {o.A_uniform * 1e4:.2f} cm² thoả cùng ràng buộc "
               f"(m = {weight_to_mass_kg(res.weight_uniform):,.1f} kg). "
               f"Nét liền: sau tối ưu (m = {weight_to_mass_kg(res.weight):,.1f} kg, "
               f"**tiết kiệm {res.saving * 100:.1f}%**). Xanh = nhỏ hơn mốc, "
                   f"cam = lớn hơn mốc. Bề rộng vẽ ∝ A^{_shape_exp(res.inp.fam):g}. Rê chuột lên giữa thanh để xem số liệu.")
    elif result_view == "Tiết diện tối ưu":
        st.plotly_chart(plot_structure(res.inp, area_member=res.A_member, title="Diện tích A (cm²) — độ dày/màu ∝ A"),
                        width="stretch")
    elif result_view == "Sơ đồ tải":
        st.plotly_chart(vz.plot_struct_loads(res), width="stretch")
        st.caption("Tải nút theo bảng nhập; với khung, tải phân bố hiển thị là thành phần vuông góc trục thanh (q⊥) mà solver dùng.")
    elif result_view == "Nội lực":
        kinds = [("axial", "N — lực dọc")] if res.inp.mode == "truss" else \
            [("moment", "M — mômen"), ("shear", "V — lực cắt"), ("axial", "N — lực dọc")]
        names = [n for _, n in kinds]
        pick = st.radio("Biểu đồ nội lực", names, horizontal=True, key="s_diag") if len(kinds) > 1 else names[0]
        st.plotly_chart(vz.plot_struct_diagram(res, dict((n, k) for k, n in kinds)[pick]), width="stretch")
        if res.inp.mode == "truss":
            st.caption("Mỗi thanh có một lực dọc N gần như không đổi: đỏ = kéo (+), xanh = nén (−), "
                       "độ dày biểu thị |N|; trị số đặt giữa thanh.")
        else:
            st.caption("Biểu đồ nội lực vẽ lệch vuông góc trục thanh, có trị số tại đầu thanh và cực trị; "
                       "dùng chung tỷ lệ cho toàn kết cấu. Quy ước M dương là kéo thớ dưới theo trục cục bộ.")
    elif result_view == "Chuyển vị":
        displacement_mode = st.segmented_control(
            "Chế độ chuyển vị",
            ["Chuyển động liên tục", "Độ võng lớn nhất"],
            default="Chuyển động liên tục",
            key="s_displacement_mode",
            label_visibility="collapsed",
            width="stretch",
        )
        if displacement_mode == "Độ võng lớn nhất":
            st.plotly_chart(vz.plot_struct_deformed(res, show_max=True), width="stretch")
        else:
            animation_html = vz.deformation_animation_html(vz.plot_struct_deformed(res))
            components.html(animation_html, height=560, scrolling=False)
        dd = res.model.node_disp(o.U)
        rows_d = [{"Nút": i, "ux (mm)": dd[i, 0] * 1e3, "uy (mm)": dd[i, 1] * 1e3,
                   **({"θz (mrad)": dd[i, 2] * 1e3} if dd.shape[1] == 3 else {})} for i in range(len(res.inp.nodes))]
        st.dataframe(pd.DataFrame(rows_d).round(4), hide_index=True, width="stretch")
    elif result_view == "Bảng thanh":
        rows = []
        for m_, (i, j) in enumerate(res.inp.members):
            sel = res.member_of_el == m_
            Nk = float(o.N[sel][np.argmax(np.abs(o.N[sel]))])
            rows.append({"Thanh": m_, "i-j": f"{i}-{j}", "L (m)": float(np.sum(res.model.L[sel])),
                         "A (cm²)": res.A_member[m_] * 1e4, "N (kN)": Nk, "Mmax (kNm)": float(np.max(o.M[sel])),
                         "U": float(np.max(o.util[sel]))})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        if res.crosscheck is not None:
            st.caption(f"Đối chiếu fem_core.solve_plane_frame: sai khác chuyển vị tương đối = {res.crosscheck['rel_diff']:.1e}")
    elif result_view == "Hội tụ":
        h = pd.DataFrame(o.history)
        mass_history = h["weight"].map(weight_to_mass_kg)
        ratio_max = max(float(h["max_util"].max()), float(h["max_disp_ratio"].max()), 1.0)
        fg = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Khối lượng thiết kế", "Mức sử dụng ràng buộc"),
            horizontal_spacing=0.14,
        )
        fg.add_scatter(
            x=np.arange(1, len(h) + 1), y=mass_history, row=1, col=1,
            name="Khối lượng (kg)", mode="lines+markers",
            marker=dict(size=5), line=dict(color=BLUE, width=2.5),
            hovertemplate="Vòng %{x}<br>m = %{y:.1f} kg<extra></extra>",
        )
        fg.add_scatter(
            x=np.arange(1, len(h) + 1), y=h["max_util"], row=1, col=2,
            name="Ứng suất/ổn định", mode="lines",
            line=dict(color=RED, width=2.5),
            hovertemplate="Vòng %{x}<br>U_max = %{y:.3f}<extra></extra>",
        )
        fg.add_scatter(
            x=np.arange(1, len(h) + 1), y=h["max_disp_ratio"], row=1, col=2,
            name="Chuyển vị/giới hạn", mode="lines",
            line=dict(color=ORANGE, width=2.5),
            hovertemplate="Vòng %{x}<br>δ/δ_lim = %{y:.3f}<extra></extra>",
        )
        fg.add_hline(y=1.0, row=1, col=2, line=dict(color="#455468", dash="dash", width=1.5),
                     annotation_text="Giới hạn = 1", annotation_position="top left",
                     annotation_font=dict(size=10, color="#455468"))
        fg.update_xaxes(title_text="Vòng lặp", row=1, col=1)
        fg.update_xaxes(title_text="Vòng lặp", row=1, col=2)
        fg.update_yaxes(title_text="Khối lượng (kg)", rangemode="tozero", row=1, col=1)
        fg.update_yaxes(title_text="Tỷ số (≤ 1 là đạt)", range=[0, ratio_max * 1.08], row=1, col=2)
        fg.update_layout(
            height=430,
            margin=dict(l=55, r=25, t=75, b=95),
            template="plotly_white",
            legend=dict(orientation="h", y=-0.28, yanchor="top", x=0.5, xanchor="center"),
            hovermode="x unified",
        )
        st.plotly_chart(fg, width="stretch", config={"displayModeBar": False})
        st.caption("Đọc đồ thị: bên trái là khối lượng qua từng vòng tối ưu; bên phải là mức sử dụng lớn nhất. "
                   "Đường đỏ kiểm tra bền/ổn định, đường cam kiểm tra chuyển vị; cả hai phải ≤ 1. "
                   "Đường gạch ngang là ngưỡng giới hạn. Đây là lịch sử thuật toán, không phải biểu đồ nội lực.")
    elif result_view == "Mặt cắt":
        member_index = st.selectbox(
            "Chọn thanh để xem mặt cắt",
            range(len(res.inp.members)),
            format_func=lambda index: f"Thanh {index}: nút {res.inp.members[index][0]}–{res.inp.members[index][1]}",
            key="s_section_member",
        )
        section_area = float(res.A_member[member_index])
        show_section_comparison(res.inp.fam, float(o.A_uniform), section_area, "s_section_comparison")
        st.caption("Diện tích hiển thị là giá trị trung bình theo chiều dài của thanh đã chọn.")
        selected_elements = np.flatnonzero(res.member_of_el == member_index)
        member_length = float(np.sum(res.model.L[selected_elements]))
        node_i, node_j = res.inp.members[member_index]
        dx = res.inp.nodes[node_j][0] - res.inp.nodes[node_i][0]
        dy = res.inp.nodes[node_j][1] - res.inp.nodes[node_i][1]
        orientation = "vertical" if abs(dy) > abs(dx) else "horizontal"
        show_section_3d(res.inp.fam, member_length, float(o.A_uniform), o.A[selected_elements],
                "s_section_3d", orientation=orientation)
    ga_res = st.session_state.get("s_ga_res")
    if ga_res is not None and is_ten_bar:
        st.subheader("So sánh hai thuật toán")
        _ten_bar_algorithm_compare(res, ga_res)
    report_block(res.report, "khung_toi_uu", "Thuyết minh — Tối ưu kết cấu khung/dàn")


def _preview_fig(geo, mode_label):
    inp = StructInput(nodes=[(n["x"], n["y"]) for n in geo["nodes"]], members=[(e["i"], e["j"]) for e in geo["elements"]],
                      supports=[(s["node"], s["ux"], s["uy"], s["rz"]) for s in geo["supports"]], loads=[], member_q=[],
                      mode="frame", fam=make_family("circle"))
    return plot_structure(inp, title="Xem trước hình học (đánh số nút/thanh)")


@st.cache_data(show_spinner=False)
def _run_ten_bar_ga(pop_size: int, generations: int, runs: int, seed0: int, use_slsqp_polish: bool):
    if any(x is None for x in (ThamSoGA, thuat_toan_di_truyen, ga_khoi_luong, ga_phan_tich_gian)):
        detail = f" Chi tiết: {GA_IMPORT_ERROR!r}" if GA_IMPORT_ERROR is not None else ""
        raise RuntimeError(
            "Không import được file toi_uu_gian_10_thanh_GA_v2.py hoặc thiếu thư viện numpy/scipy/matplotlib."
            + detail
        )
    p = ThamSoGA(kich_thuoc_quan_the=int(pop_size), so_the_he=int(generations))
    designs, histories = [], []
    for k in range(int(runs)):
        A, hist = thuat_toan_di_truyen(seed=int(seed0) + k, in_tien_trinh=False, tham_so=p)
        designs.append(A)
        histories.append(hist)
    weights_lb = np.array([ga_khoi_luong(A) for A in designs], dtype=float)
    best_idx = int(np.argmin(weights_lb))
    A_ga = np.asarray(designs[best_idx], dtype=float)
    A_sl = None
    W_sl_lb = None
    if use_slsqp_polish and toi_uu_gradient is not None:
        A_sl = np.asarray(toi_uu_gradient(A_ga), dtype=float)
        W_sl_lb = float(ga_khoi_luong(A_sl))
    u_ga, sigma_ga = ga_phan_tich_gian(A_ga)
    return {
        "A_ga": A_ga,
        "A_app": A_ga[TEN_BAR_GA_TO_APP_ORDER],
        "u_ga": np.asarray(u_ga, dtype=float),
        "sigma_ga": np.asarray(sigma_ga, dtype=float),
        "weights_lb": weights_lb,
        "histories": [np.asarray(h, dtype=float) for h in histories],
        "best_idx": best_idx,
        "W_ga_lb": float(weights_lb[best_idx]),
        "A_sl": A_sl,
        "W_sl_lb": W_sl_lb,
        "pop_size": int(pop_size),
        "generations": int(generations),
        "runs": int(runs),
        "seed0": int(seed0),
    }


def _ten_bar_algorithm_compare(res, ga_res):
    oc_lb = float(res.weight) * KN_TO_LB
    ga_lb = float(ga_res["W_ga_lb"])
    sl_lb = ga_res.get("W_sl_lb")
    diff_lb = ga_lb - oc_lb
    diff_pct = diff_lb / oc_lb * 100.0 if oc_lb else float("nan")
    better = "GA nhẹ hơn" if diff_lb < 0 else ("OC/FSD + SLSQP nhẹ hơn" if diff_lb > 0 else "Hai thuật toán bằng nhau")

    metrics([
        ("OC/FSD + SLSQP", f"{oc_lb:,.2f} lb"),
        ("GA tốt nhất", f"{ga_lb:,.2f} lb"),
        ("Chênh lệch GA - OC", f"{diff_lb:+,.2f} lb"),
        ("Chênh lệch", f"{diff_pct:+.3f}%"),
        ("Kết luận", better, "metric-good" if diff_lb <= 0 else "metric-bad"),
    ])
    if sl_lb is not None:
        st.caption(f"SLSQP kiểm chứng từ nghiệm GA: {sl_lb:,.2f} lb; lệch so với GA {sl_lb - ga_lb:+,.2f} lb "
                   f"({(sl_lb / ga_lb - 1) * 100:+.3f}%).")

    compare_view = st.selectbox("Nội dung so sánh GA", ["Tổng hợp", "Hội tụ", "Thiết kế GA", "Thuyết minh"],
                                key="ten_bar_compare_view")

    if compare_view == "Tổng hợp":
        rows = []
        A_oc_in2 = np.asarray(res.A_member, dtype=float) / (0.0254 ** 2)
        sigma_ga_app = np.asarray(ga_res["sigma_ga"], dtype=float)[TEN_BAR_GA_TO_APP_ORDER]
        for i, (a_oc, a_ga, s_ga) in enumerate(zip(A_oc_in2, ga_res["A_app"], sigma_ga_app), start=1):
            rows.append({
                "Thanh": i,
                "A_OC/SLSQP (in²)": a_oc,
                "A_GA (in²)": float(a_ga),
                "Chênh A (in²)": float(a_ga - a_oc),
                "Chênh A (%)": float((a_ga / a_oc - 1.0) * 100.0) if a_oc else np.nan,
                "σ_GA (ksi)": float(s_ga),
            })
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
        st.code(
            "\n".join([
                f"W_oc_lb = {oc_lb:.6f}",
                f"W_ga_lb = {ga_lb:.6f}",
                "chenh_lech_lb = W_ga_lb - W_oc_lb",
                "chenh_lech_pct = chenh_lech_lb / W_oc_lb * 100",
                f"chenh_lech_lb = {diff_lb:+.6f}",
                f"chenh_lech_pct = {diff_pct:+.6f}",
            ]),
            language="python",
        )

    elif compare_view == "Hội tụ":
        histories = ga_res["histories"]
        fig = go.Figure()
        for k, hist in enumerate(histories):
            y = pd.Series(hist).ffill()
            fig.add_scatter(
                x=np.arange(1, len(hist) + 1),
                y=y,
                mode="lines",
                name=f"Lần {k + 1}",
                line=dict(width=3 if k == ga_res["best_idx"] else 1.4),
                opacity=1.0 if k == ga_res["best_idx"] else 0.45,
                hovertemplate="Thế hệ %{x}<br>W = %{y:.2f} lb<extra></extra>",
            )
        fig.add_hline(y=oc_lb, line=dict(color=RED, dash="dash", width=1.5),
                      annotation_text=f"OC/FSD + SLSQP {oc_lb:.1f} lb",
                      annotation_position="top left")
        fig.add_hline(y=5060.85, line=dict(color="#111827", dash="dot", width=1.2),
                      annotation_text="Tài liệu 5060.85 lb",
                      annotation_position="bottom left")
        fig.update_layout(
            height=430,
            template="plotly_white",
            title="Quá trình hội tụ của thuật toán di truyền",
            xaxis_title="Thế hệ",
            yaxis_title="Khối lượng cá thể hợp lệ tốt nhất (lb)",
            hovermode="x unified",
            legend=dict(orientation="h", y=-0.22, x=0.5, xanchor="center"),
            margin=dict(l=55, r=25, t=65, b=90),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        run_rows = [{
            "Lần chạy": k + 1,
            "Seed": ga_res["seed0"] + k,
            "Khối lượng cuối (lb)": float(w),
            "Vai trò": "Tốt nhất" if k == ga_res["best_idx"] else "",
        } for k, w in enumerate(ga_res["weights_lb"])]
        st.dataframe(pd.DataFrame(run_rows).round(3), hide_index=True, width="stretch")

    elif compare_view == "Thiết kế GA":
        util_ga = np.abs(np.asarray(ga_res["sigma_ga"], dtype=float)[TEN_BAR_GA_TO_APP_ORDER]) / 25.0
        ga_inp = ten_bar_truss_example()
        ga_inp.fam = res.inp.fam
        st.plotly_chart(
            plot_structure(ga_inp, area_member=np.asarray(ga_res["A_app"], dtype=float) * (0.0254 ** 2),
                           util_member=util_ga,
                           title=f"Thiết kế GA — W = {ga_lb:,.2f} lb"),
            width="stretch",
        )
        st.caption("Độ dày/màu theo diện tích tiết diện GA; mức sử dụng lấy theo |σ|/[σ] của bài benchmark 10 thanh.")

    elif compare_view == "Thuyết minh":
        st.markdown(_ga_explanation_text(ga_res, oc_lb, ga_lb, diff_lb, diff_pct))


def _ga_explanation_text(ga_res, oc_lb, ga_lb, diff_lb, diff_pct):
    return f"""
**Thuật toán di truyền GA trong file `toi_uu_gian_10_thanh_GA_v2.py`.**

Bài toán dùng 10 biến thiết kế `A1..A10`, mỗi biến là diện tích tiết diện một thanh trong khoảng `0.1..35 in²`.
Mỗi cá thể của quần thể là một phương án thiết kế đầy đủ cho giàn 10 thanh.

**Chu trình tính toán.**

1. Khởi tạo `{ga_res["pop_size"]}` cá thể ngẫu nhiên.
2. Với từng cá thể, giải FEM để lấy chuyển vị và ứng suất.
3. Tính khối lượng `W = rho * sum(A_i * L_i)`.
4. Nếu vi phạm ứng suất hoặc chuyển vị, cộng phạt vào điểm mục tiêu.
5. Chọn lọc kiểu đấu, lai ghép SBX và đột biến đa thức để tạo thế hệ mới.
6. Giữ lại cá thể tinh hoa, lặp `{ga_res["generations"]}` thế hệ cho mỗi lần chạy.

**Cách đọc kết quả.**

- `OC/FSD + SLSQP`: thuật toán gốc của Đớp Tool, chạy tổng quát cho khung/dàn.
- `GA`: thuật toán di truyền riêng cho benchmark dàn 10 thanh.
- Công thức so sánh đang dùng:

```python
chenh_lech_lb = W_ga_lb - W_oc_lb
chenh_lech_pct = chenh_lech_lb / W_oc_lb * 100
```

Với lần chạy hiện tại: `W_oc = {oc_lb:,.2f} lb`, `W_ga = {ga_lb:,.2f} lb`,
chênh lệch `GA - OC = {diff_lb:+,.2f} lb`, tức `{diff_pct:+.3f}%`.
"""


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def tab_dynamics():
    st.header("Động lực học kết cấu · SDOF / nhà nhiều tầng")
    st.caption("Dao động tự do, đáp ứng theo thời gian và phổ đàn hồi; mô hình nhà cắt nhiều tầng dùng một bậc tự do ngang mỗi tầng. Đơn vị SI.")
    system = st.radio("Mô hình", ["SDOF", "Nhà nhiều tầng"], horizontal=True, key="dyn_system")
    if system == "Nhà nhiều tầng":
        _tab_mdof_dynamics()
        return
    left, right = st.columns([0.32, 0.68])
    with left:
        mode = st.radio("Phân tích", ["Dao động tự do", "Đáp ứng theo thời gian", "Phổ đáp ứng"], key="dyn_mode")
        mass = st.number_input("Khối lượng m (kg)", min_value=0.001, value=1000.0, step=100.0)
        stiffness = st.number_input("Độ cứng k (N/m)", min_value=0.001, value=1_000_000.0, step=100_000.0, format="%.3f")
        damping = st.number_input("Tỷ số cản ξ", min_value=0.0, max_value=2.0, value=0.05, step=0.01)
        natural_period = 2 * np.pi * np.sqrt(mass / stiffness)
        st.metric("Chu kỳ riêng Tₙ", f"{natural_period:.4f} s")
        if mode == "Dao động tự do":
            u0 = st.number_input("Chuyển vị ban đầu u₀ (m)", value=0.05, step=0.01, format="%.4f")
            v0 = st.number_input("Vận tốc ban đầu v₀ (m/s)", value=0.0, step=0.1, format="%.4f")
            duration = st.number_input("Thời lượng (s)", min_value=0.01, value=max(5.0, natural_period * 5), step=1.0)
            dt = st.number_input("Bước thời gian (s)", min_value=0.0001, value=min(0.01, max(0.0001, natural_period / 100)), format="%.4f")
            if st.button("Tính dao động tự do", type="primary"):
                try:
                    result = free_vibration(mass, stiffness, damping, duration, dt, u0, v0)
                    st.session_state["dyn_result"] = ("free", result)
                except ValueError as exc:
                    st.error(str(exc))
        else:
            upload = st.file_uploader("Gia tốc nền (m/s²), mỗi dòng một giá trị", type=["txt", "csv"])
            dt = st.number_input("Bước thời gian bản ghi (s)", min_value=0.0001, value=0.01, format="%.4f", key="dyn_dt")
            if upload is not None:
                try:
                    ag = np.loadtxt(upload, delimiter="," if upload.name.lower().endswith(".csv") else None, dtype=float)
                    ag = np.asarray(ag).reshape(-1)
                except Exception:
                    st.error("Không đọc được bản ghi. Hãy dùng file chỉ chứa một cột gia tốc nền (m/s²).")
                    ag = None
            else:
                duration = st.number_input("Thời lượng sóng sin (s)", min_value=0.1, value=10.0, step=1.0)
                ag_peak = st.number_input("Gia tốc đỉnh (m/s²)", min_value=0.0, value=1.0, step=0.1)
                excitation_period = st.number_input("Chu kỳ sóng sin (s)", min_value=0.01, value=max(0.1, natural_period), step=0.1)
                n = max(2, int(duration / dt) + 1)
                tt = np.arange(n) * dt
                ag = ag_peak * np.sin(2 * np.pi * tt / excitation_period)
            if mode == "Phổ đáp ứng":
                tmin = st.number_input("Chu kỳ nhỏ nhất (s)", min_value=0.01, value=0.1, step=0.05, key="spec_tmin")
                tmax = st.number_input("Chu kỳ lớn nhất (s)", min_value=0.02, value=4.0, step=0.5, key="spec_tmax")
                nperiod = st.number_input("Số chu kỳ khảo sát", min_value=5, max_value=300, value=80, step=5)
            if st.button("Tính đáp ứng", type="primary"):
                try:
                    if ag is None or len(ag) < 2:
                        raise ValueError("Bản ghi gia tốc phải có ít nhất hai mẫu.")
                    if mode == "Phổ đáp ứng":
                        if tmax <= tmin:
                            raise ValueError("Chu kỳ lớn nhất phải lớn hơn chu kỳ nhỏ nhất.")
                        result = response_spectrum(np.linspace(tmin, tmax, int(nperiod)), ag, dt, damping)
                        st.session_state["dyn_result"] = ("spectrum", result)
                    else:
                        result = newmark_response(mass, stiffness, damping, ag, dt)
                        st.session_state["dyn_result"] = ("history", result)
                except ValueError as exc:
                    st.error(str(exc))
    with right:
        saved = st.session_state.get("dyn_result")
        expected = {"Dao động tự do": "free", "Đáp ứng theo thời gian": "history", "Phổ đáp ứng": "spectrum"}[mode]
        if not saved or saved[0] != expected:
            st.info("Nhập thông số rồi bấm nút tính để xem kết quả.")
            return
        kind, result = saved
        if kind == "free":
            st.metric("Tần số riêng", f"{result['omega_n'] / (2*np.pi):.4f} Hz")
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Chuyển vị (m)", "Vận tốc (m/s)", "Gia tốc (m/s²)"))
            for row, key, name in [(1, "displacement", "u"), (2, "velocity", "v"), (3, "acceleration", "a")]:
                fig.add_trace(go.Scatter(x=result["time"], y=result[key], name=name), row=row, col=1)
            fig.update_xaxes(title_text="Thời gian (s)", row=3, col=1)
        elif kind == "history":
            c1, c2, c3 = st.columns(3)
            c1.metric("|u|max", f"{np.max(np.abs(result['displacement'])):.5g} m")
            c2.metric("|a tuyệt đối|max", f"{np.max(np.abs(result['absolute_acceleration'])):.5g} m/s²")
            c3.metric("|V đáy|max", f"{np.max(np.abs(result['base_shear']))/1000:.5g} kN")
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Gia tốc nền (m/s²)", "Chuyển vị tương đối (m)", "Gia tốc tuyệt đối (m/s²)"))
            for row, key in [(1, "ground_acceleration"), (2, "displacement"), (3, "absolute_acceleration")]:
                fig.add_trace(go.Scatter(x=result["time"], y=result[key], name=key), row=row, col=1)
            fig.update_xaxes(title_text="Thời gian (s)", row=3, col=1)
        else:
            c1, c2 = st.columns(2)
            c1.metric("Sa,max", f"{np.max(result['Sa'])/9.80665:.4g} g")
            c2.metric("Sd,max", f"{np.max(result['Sd'])*1000:.4g} mm")
            fig = make_subplots(rows=1, cols=2, subplot_titles=("Phổ chuyển vị Sd", "Phổ gia tốc giả PSa"))
            fig.add_trace(go.Scatter(x=result["period"], y=result["Sd"], name="Sd"), row=1, col=1)
            fig.add_trace(go.Scatter(x=result["period"], y=result["PSa_g"], name="PSa (g)"), row=1, col=2)
            fig.update_xaxes(title_text="Chu kỳ T (s)", row=1, col=1); fig.update_xaxes(title_text="Chu kỳ T (s)", row=1, col=2)
        fig.update_layout(height=650, margin=dict(l=15, r=15, t=55, b=20), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.download_button("Tải kết quả CSV", pd.DataFrame(result).to_csv(index=False).encode("utf-8-sig"), file_name="dop_tool_dynamic_results.csv", mime="text/csv")


def tab_opensees_10_story():
    """Educational view of the supplied OpenSeesPy ten-story shear-building model."""
    st.header("Mô hình 10 tầng OpenSees · cơ sở lý thuyết")
    st.markdown(
        "<div style='padding:12px 16px;border-left:4px solid #d77835;background:rgba(215,120,53,.08);'>"
        "Shear-building 10 tầng · 10 bậc tự do ngang · bốn trục lò xo/cản · tường cắt có EI → ∞"
        "</div>", unsafe_allow_html=True,
    )

    # Model properties in SI units.
    # Yield/backbone reference values are retained from the source model.
    fy = np.array([85_000.0, 95_000.0, 75_000.0, 65_000.0])
    fu = fy * np.array([1.18, 1.12, 1.20, 1.10])
    dsu = np.array([0.32, 0.34, 0.30, 0.36])
    with st.expander("Mô hình FEM dùng để xác định độ cứng lò xo", expanded=True):
        f1, f2, f3 = st.columns(3)
        op10_nbx = int(f1.number_input("Số nhịp X", min_value=1, max_value=2, value=1, step=1, key="op10_nbx"))
        op10_nby = int(f2.number_input("Số nhịp Y", min_value=1, max_value=2, value=1, step=1, key="op10_nby"))
        op10_h = f3.number_input("Chiều cao tầng (m)", min_value=2.0, max_value=6.0, value=3.3, step=0.1, key="op10_story_h")
        f4, f5, f6 = st.columns(3)
        op10_bx = f4.number_input("Nhịp X (m)", min_value=2.0, max_value=12.0, value=5.0, step=0.5, key="op10_bay_x")
        op10_by = f5.number_input("Nhịp Y (m)", min_value=2.0, max_value=12.0, value=5.0, step=0.5, key="op10_bay_y")
        op10_E = f6.number_input("Mô đun đàn hồi E (GPa)", min_value=10.0, max_value=45.0, value=30.0, step=1.0, key="op10_E") * 1e9
        f7, f8 = st.columns(2)
        op10_col_b = f7.number_input("Cột vuông · cạnh (m)", min_value=0.2, max_value=1.2, value=0.4, step=0.05, key="op10_col_b")
        op10_beam_b = f8.number_input("Dầm · b × h (m)", min_value=0.15, max_value=0.8, value=0.3, step=0.05, key="op10_beam_b")
        op10_beam_h = st.number_input("Dầm · chiều cao h (m)", min_value=0.25, max_value=1.2, value=0.5, step=0.05, key="op10_beam_h")
        st.caption("Mô hình khung chữ nhật 10 tầng; cột vuông; sàn diaphragm cứng. FEM đàn hồi 3D có 6 DOF/nút; kết quả tab này là độ cứng lắc ngang phương X sau khi ngưng tụ các DOF còn lại.")
    op10_Icol = op10_col_b**4 / 12
    op10_Ibeam = op10_beam_b * op10_beam_h**3 / 12
    try:
        fem_story_k = frame_story_stiffness(10, op10_nbx, op10_nby, op10_bx, op10_by, op10_h, op10_E, op10_Icol, op10_Ibeam, "rect")
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(f"Không lập được ma trận FEM cho mô hình 10 tầng: {exc}")
        fem_story_k = np.full(10, 1.0)
    # Four parallel equivalent springs share each condensed floor stiffness.
    spring_k_floor = fem_story_k / 4.0
    ke = np.full(4, float(np.mean(spring_k_floor)))
    damping = np.array([0.05, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
    dy = fy / ke
    ksec = 4 * ke
    m_total_tons = st.number_input(
        "Tổng khối lượng 10 tầng (tấn) · phân đều 10% mỗi tầng",
        min_value=10.0, value=1000.0, step=100.0, key="op10_total_mass_tons",
    )
    mass_each = m_total_tons * 1000 / 10
    # Shear-building stiffness assembly for fixed base and floor DOFs.
    story_k = fem_story_k
    floor_mass_ratio = np.full(10, 0.1)
    K = np.zeros((10, 10), dtype=float)
    for i, ki in enumerate(story_k):
        # Story i spring connects floor i to the floor below (ground for i=0).
        K[i, i] += ki
        if i:
            K[i - 1, i - 1] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
    eigvals, modes = np.linalg.eigh(K / mass_each)
    omega = np.sqrt(np.maximum(eigvals, 0.0))
    periods = 2 * np.pi / omega
    # Normalize each eigenvector to unit roof displacement, with positive roof ordinate.
    modes = modes / modes[-1, :]
    mode_shapes = modes[:, :3]
    first_period = float(periods[0])
    first_freq = 1 / first_period
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số tầng / bậc tự do", "10 / 10")
    c2.metric("Phần tử mỗi tầng", "4 lò xo + 4 cản")
    c3.metric("Chu kỳ riêng T₁ (đàn hồi)", f"{first_period:.3f} s")
    c4.metric("Tần số f₁", f"{first_freq:.3f} Hz")

    op10_view = st.selectbox("Nội dung bài mẫu", ["Mô hình & lò xo", "Cơ sở OpenSees", "Dao động", "Phi tuyến"],
                             key="op10_view")
    if op10_view == "Mô hình & lò xo":
        st.markdown("#### Sơ đồ shear-building tương đương theo bài toán gốc")
        fig = go.Figure()
        xcols = [0.0, 1.0, 2.0, 3.0]
        for level in range(11):
            fig.add_trace(go.Scatter(x=[-0.25, 3.25], y=[level, level], mode="lines", line=dict(color="#20252a", width=2.6), showlegend=False, hoverinfo="skip"))
            if level < 10:
                for col, x in enumerate(xcols, start=1):
                    mid = level + .5
                    # Spring and dashpot symbols on each of four parallel axes.
                    fig.add_trace(go.Scatter(x=[x, x, x-.12, x+.12, x-.12, x+.12, x, x], y=[level+.06, mid-.17, mid-.11, mid-.05, mid+.01, mid+.07, mid+.13, level+.94], mode="lines", line=dict(color="#22282e", width=1.6), showlegend=False, hoverinfo="skip"))
                    fig.add_trace(go.Scatter(x=[x+.18, x+.18, x+.29, x+.29, x+.18, x+.18], y=[level+.13, mid-.08, mid-.08, mid+.08, mid+.08, level+.87], mode="lines", line=dict(color="#a43737", width=1.5), showlegend=False, hoverinfo="skip"))
                    fig.add_annotation(x=x, y=mid, text=f"K{col}<br>C{col}", showarrow=False, font=dict(size=8, color="#963b3b"), bgcolor="white")
                fig.add_trace(go.Scatter(x=[1.5], y=[level+1], mode="markers+text", text=[f"m{level+1}"], textposition="middle center", marker=dict(size=27, color="#15191d"), textfont=dict(size=9, color="white"), showlegend=False, hovertemplate=f"Tầng {level+1}: mᵢ = M/10<extra></extra>"))
                fig.add_trace(go.Scatter(x=[3.5, 4.2, 4.2, 3.5, 3.5], y=[level, level, level+1, level+1, level], mode="lines", fill="toself", fillcolor="rgba(150,155,160,.55)", line=dict(color="#565b60", width=1), showlegend=False, hoverinfo="skip"))
                fig.add_annotation(x=3.85, y=mid, text="TƯỜNG<br>CỨNG", showarrow=False, font=dict(size=7, color="#353a3e"))
        for x in xcols:
            fig.add_trace(go.Scatter(x=[x-.16, x+.16], y=[-.03, -.03], mode="lines", line=dict(color="#20252a", width=4), showlegend=False, hoverinfo="skip"))
            for hatch in np.linspace(x-.13, x+.13, 4):
                fig.add_trace(go.Scatter(x=[hatch, hatch-.09], y=[-.05, -.17], mode="lines", line=dict(color="#20252a", width=1), showlegend=False, hoverinfo="skip"))
        for x in xcols[:3]:
            fig.add_annotation(x=x, y=10.16, text="EI = ∞", showarrow=False, font=dict(size=10, color="#2d3338"))
        fig.add_trace(go.Scatter(x=[5.15, 6.0, 6.0, 5.15, 5.15], y=[0, 0, 2.2, 2.2, 0], mode="lines", line=dict(color="#22282e", width=3), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[5.58], y=[2.5], mode="markers+text", text=["mₑff"], textposition="top center", marker=dict(size=26, color="#15191d"), textfont=dict(size=10, color="white"), showlegend=False))
        fig.add_annotation(x=6.42, y=1.1, text=f"Kₑff<br>Tₑff ≈ {first_period:.2f} s", showarrow=False, font=dict(size=11, color="#c43b48"))
        fig.add_annotation(x=4.72, y=5, text="≈", showarrow=False, font=dict(size=24, color="#30363b"))
        fig.add_annotation(x=-.52, y=5, text="u_g(t)", textangle=-90, showarrow=False, font=dict(size=10, color="#347db6"))
        fig.add_trace(go.Scatter(x=[-.37, -.37], y=[0, 10], mode="lines", line=dict(color="#347db6", width=2, dash="dash"), showlegend=False, hoverinfo="skip"))
        fig.update_layout(height=690, margin=dict(l=24, r=20, t=10, b=15), paper_bgcolor="white", plot_bgcolor="white", xaxis=dict(visible=False, range=[-.8, 6.95], fixedrange=True), yaxis=dict(title="Tầng", range=[-.25, 10.35], dtick=1, gridcolor="#edf0f2", fixedrange=True), showlegend=False, font=dict(family="Arial", color="#35414a"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("""
        **Cách đọc sơ đồ.** Mỗi chấm đen `mᵢ` là khối lượng quy tụ tại sàn tầng `i`; sàn được giả thiết đủ cứng trong mặt phẳng nên các nút cùng tầng chuyển vị ngang như nhau. Bốn nhánh lò xo `K₁…K₄` (đen) và bộ cản `C₁…C₄` (đỏ) nối hai sàn liên tiếp, làm việc song song; tổng độ cứng tầng là tổng các nhánh. Đường nét đứt `u_g(t)` biểu diễn chuyển động nền do động đất, còn chân cột gạch chéo là liên kết ngàm.

        Thanh xám “tường cứng” và nhãn `EI → ∞` chỉ là **lý tưởng hóa trong hình bài toán gốc**, nghĩa là tường được coi cứng tuyệt đối để minh họa. Mô hình FEM đang tính độ cứng khung dầm–cột với tiết diện và vật liệu đã nhập; hiện chưa có phần tử shell cho tường, nên không được hiểu rằng độ cứng tường đã được cộng vào kết quả FEM. Các lò xo trong sơ đồ là mô hình tương đương của độ cứng tầng FEM, được chia đều thành bốn nhánh theo giả thiết đối xứng.

        Khối bên phải biểu diễn hệ một bậc tự do tương đương theo mode dao động thứ nhất (SDOF): `K_eff` là độ cứng hiệu dụng và `T_eff` là chu kỳ hiệu dụng. Nó dùng để giải thích phép quy đổi, không phải một khung riêng trong công trình. Các giá trị độ cứng tầng FEM và cách tính được trình bày ngay bên dưới.
        """)
        st.markdown("#### Độ cứng tầng ngưng tụ từ FEM")
        st.dataframe(pd.DataFrame({"Tầng": np.arange(1, 11), "K tầng từ FEM (MN/m)": fem_story_k / 1e6, "k mỗi lò xo song song (MN/m)": spring_k_floor / 1e6}), hide_index=True, use_container_width=True)
        st.caption("Bốn lò xo được giả thiết giống nhau theo hình học đối xứng; mỗi lò xo nhận một phần tư độ cứng tầng FEM.")
        st.info("Độ cứng đàn hồi FEM dùng tiết diện nguyên, E đồng nhất, liên kết nút cứng, chân cột ngàm và chưa giảm độ cứng do nứt bê tông. Hình shear-wall là sơ đồ lý tưởng hóa; lưới FEM hiện tính khung dầm-cột, chưa có shell/wall element. Độ cứng thu được là tương đương phương X, không phải sáu lò xo độc lập theo sáu DOF.")
        st.markdown("#### Đặc tính bốn nhóm lò xo")
        st.dataframe(pd.DataFrame({"Nhóm": np.arange(1, 5), "kₑ FEM TB (MN/m)": ke / 1e6, "Fᵧ (kN)": fy / 1000, "Fᵤ (kN)": fu / 1000, "Δᵧ = Fᵧ/kₑ (mm)": dy * 1000, "Δᵤ (mm)": dsu * 1000, "4kₑ TB (MN/m)": ksec / 1e6}), hide_index=True, use_container_width=True)
        with st.expander("Thuyết minh phép tính và bảng kiểm từng lò xo", expanded=True):
            st.markdown("**Từ FEM đến độ cứng lò xo.** Từ các đầu vào `E`, nhịp, chiều cao tầng, kích thước dầm/cột, chương trình lắp ma trận độ cứng khung không gian đàn hồi 3D; chân cột ngàm và sàn làm việc như diaphragm cứng. Các bậc tự do quay và chuyển vị không phải phương X được ngưng tụ tĩnh để thu ma trận `K_cond`. Ma trận này được quy đổi về shear-building tương đương, thu độ cứng tầng `Kᵢ,FEM`; bốn nhánh song song chia đều `kₑ,ij = Kᵢ,FEM/4`. Bảng hiển thị kết quả riêng từng tầng; bảng nhóm hiển thị giá trị trung bình.")
            st.latex(r"\mathbf K_{cond}=\mathbf K_{xx}-\mathbf K_{xr}\mathbf K_{rr}^{-1}\mathbf K_{rx};\qquad k_{e,ij}=K_{i,FEM}/4")
            st.markdown("Sau khi có `kₑ,FEM`, chương trình tự tính `Δᵧ = Fᵧ/kₑ`, `k_h = (Fᵤ − Fᵧ)/(Δᵤ − Δᵧ)` và kiểm tra ngược `Fᵧ + k_h(Δᵤ − Δᵧ) = Fᵤ`. `Fᵧ`, tỷ lệ `Fᵤ/Fᵧ` và `Δᵤ` vẫn là backbone tham chiếu của bài toán gốc; chúng chưa được suy ra từ phân tích tiết diện phi tuyến.")
            st.markdown("`kₑ` lấy từ FEM; sau đổi đơn vị SI, tính `Δᵧ = Fᵧ/kₑ`, rồi `k_h = (Fᵤ − Fᵧ)/(Δᵤ − Δᵧ)`. Với bốn lò xo song song bằng nhau, tổng độ cứng tầng là `Kᵢ = 4kₑ`.")
            ultimate_ratio = np.array([1.18, 1.12, 1.20, 1.10])
            kh = np.divide(fu - fy, dsu - dy, out=np.full(4, np.nan), where=(dsu > dy))
            dy_calc = fy / ke
            fu_calc = ultimate_ratio * fy
            fu_back = fy + kh * (dsu - dy)
            validation = pd.DataFrame({
                "Nhóm": np.arange(1, 5),
                "rᵤ = Fᵤ/Fᵧ": ultimate_ratio,
                "Fᵤ tự tính (kN)": fu_calc / 1000,
                "Δᵧ tính = Fᵧ/kₑ (mm)": dy_calc * 1000,
                "k_h sau chảy (MN/m)": kh / 1e6,
                "Fᵤ kiểm ngược (kN)": fu_back / 1000,
                "Điều kiện hợp lệ": np.where((dy < dsu) & (fu > fy) & np.isclose(fu_calc, fu, rtol=1e-10) & np.isclose(fu_back, fu, rtol=1e-8, atol=1e-6), "ĐẠT", "KIỂM TRA"),
            })
            st.dataframe(validation.round(5), hide_index=True, use_container_width=True)
            group = st.selectbox("Xem phép tính chi tiết nhóm lò xo", [1, 2, 3, 4], key="op10_spring_check_group") - 1
            st.latex(rf"\Delta_{{y,{group+1}}}=\frac{{F_{{y,{group+1}}}}}{{k_{{e,{group+1}}}}}=\frac{{{fy[group]/1000:.1f}\;\mathrm{{kN}}}}{{{ke[group]/1e6:.2f}\;\mathrm{{MN/m}}}}={dy[group]*1000:.4f}\;\mathrm{{mm}}")
            st.latex(rf"F_{{u,{group+1}}}=r_uF_y={ultimate_ratio[group]:.2f}\times{fy[group]/1000:.1f}={fu_calc[group]/1000:.2f}\;\mathrm{{kN}}")
            st.latex(rf"k_{{h,{group+1}}}=\frac{{F_u-F_y}}{{\Delta_u-\Delta_y}}=\frac{{{(fu[group]-fy[group])/1000:.2f}\;\mathrm{{kN}}}}{{{(dsu[group]-dy[group])*1000:.4f}\;\mathrm{{mm}}}}={kh[group]/1e6:.4f}\;\mathrm{{MN/m}}")
            st.latex(rf"k_{{tầng,{group+1}}}=4k_e=4\times{ke[group]/1e6:.2f}={ksec[group]/1e6:.2f}\;\mathrm{{MN/m}}")
            if np.all(validation["Điều kiện hợp lệ"] == "ĐẠT"):
                st.success("Kiểm tra đạt: các đại lượng tự tính nhất quán với bộ thông số chuẩn và backbone lò xo.")
            else:
                st.error("Có thông số không nhất quán hoặc điều kiện hình học đường cong lò xo không hợp lệ.")
        st.markdown("**Diễn giải lực–chuyển vị.** Độ dốc ban đầu của mỗi lò xo là `kₑ,FEM`. Khi đạt lực chảy `Fᵧ`, chuyển vị chảy bằng `Δᵧ = Fᵧ/kₑ`. Sau chảy, lực tăng đến `Fᵤ` tại `Δᵤ` theo nhánh hóa bền; các nhánh sau giảm còn `0,2Fᵤ` tại `1,1Δᵤ` và `0,1Fᵤ` tại `1,25Δᵤ` để mô tả suy giảm sức kháng. Bốn nhánh song song cùng biến dạng nên `Kᵢ,FEM = Σkₑ = 4kₑ`, còn lực tầng bằng tổng lực của bốn nhánh.")
        st.markdown("**Cản nhớt.** Mỗi tầng có nhánh cản song song; `F_c = C_i |v|^α sign(v)`. Với `α = 1` là cản nhớt tuyến tính. Tỷ số cản từng tầng xác định `C_i = 2ξ_iω_i m_i`; cản Rayleigh có dạng `C = α_M M + β_K K`, được hiệu chỉnh theo hai mode mục tiêu.")
        with st.expander("Cách tính chu kỳ cơ bản T₁", expanded=True):
            st.markdown("1. Chọn 10 chuyển vị tầng `u = [u₁,…,u₁₀]ᵀ`; mỗi tầng có khối lượng `mᵢ = M_total/10`.\n2. Từ lò xo bốn nhánh song song, `kᵢ = 4kₑ`; các độ cứng tầng lặp theo bốn nhóm trong mô hình.\n3. Lắp ma trận tam đường chéo `K`: đường chéo gồm độ cứng tầng hiện tại và phía trên; hai đường kề chéo bằng `−kᵢ`. Ma trận khối lượng là `M = diag(m₁,…,m₁₀)`.\n4. Giải `Kφ = ω²Mφ`. Trị riêng nhỏ nhất dương là `λ₁ = ω₁²`; do đó `T₁ = 2π/√λ₁` và `f₁ = 1/T₁`.\n5. Tăng khối lượng làm chu kỳ tăng; tăng độ cứng lò xo làm chu kỳ giảm. Dạng mode 1 mô tả chuyển động đồng pha tổng thể của các tầng.")
            st.latex(r"\det(\mathbf K-\omega^2\mathbf M)=0;\qquad T_1=\frac{2\pi}{\sqrt{\lambda_1}};\qquad k_{i,story}=\sum_{j=1}^{4}k_{ij}")
        p1, p2 = st.columns(2)
        with p1:
            masses_fig = go.Figure(go.Bar(x=np.arange(1, 11), y=np.full(10, mass_each / 1000), marker_color="#a8df18", marker_line_color="#3d5b24", marker_line_width=1))
            masses_fig.update_layout(height=230, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Tầng", yaxis_title="mᵢ (tấn)", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(masses_fig, use_container_width=True)
        with p2:
            stiff_fig = go.Figure(go.Bar(x=np.arange(1, 11), y=story_k / 1e6, marker_color="#3d6b55"))
            stiff_fig.update_layout(height=230, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Tầng", yaxis_title="kᵢ (MN/m)", paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(stiff_fig, use_container_width=True)

    if op10_view == "Mô hình & lò xo":
        st.markdown("#### Hệ kết cấu và mức độ lý tưởng hóa")
        st.markdown("Bài toán được quy đổi thành **shear-building 2D** với 10 bậc tự do ngang. Mỗi tầng là một khối lượng tập trung; bốn nhánh lò xo/cản ngang cùng nối hai tầng kề nhau. Các cấu kiện đứng/tường shear-wall được lý tưởng hóa cứng (`EI → ∞`) để biến dạng ngang tập trung vào các lò xo tầng.")
        typ1, typ2, typ3 = st.columns(3)
        typ1.markdown("**Shear-building**  \nĐại diện độ cứng tầng bằng lò xo; không mô tả riêng biến dạng dầm, cột hay nút khung.")
        typ2.markdown("**Shear-wall tương đương**  \nTường được xem cứng tuyệt đối trong sơ đồ; không có phần tử shell/wall hữu hạn độ cứng.")
        typ3.markdown("**Khung dầm-cột thực**  \nCần mô hình beam-column, nút, tiết diện và khớp dẻo riêng; đó không phải idealization của bài toán 10 tầng này.")
        st.markdown("#### Hai lựa chọn vật liệu cho lò xo ngang")
        mat_choice = st.radio("Chế độ minh họa", ["Phi tuyến · HystereticSM", "Đàn hồi · Elastic"], horizontal=True, key="op10_material_choice")
        mat_col, plot_col = st.columns([0.45, 0.55])
        with mat_col:
            if mat_choice.startswith("Phi tuyến"):
                st.markdown("**HystereticSM đa tuyến**")
                st.markdown("Mô hình dùng vật liệu `HystereticSM` với envelope kéo và nén đối xứng. Bốn điểm sau gốc cho mỗi nhánh biểu diễn chảy, hóa bền đến sức kháng cực hạn, rồi suy giảm sức kháng. Tùy chọn pinching của mô hình nguồn đặt `pinch = (1, 1)`. Khi tải đảo chiều, lực phụ thuộc lịch sử biến dạng nên sinh vòng trễ và tiêu tán năng lượng.")
                st.markdown("- Điểm chảy: `(Δᵧ, Fᵧ)`\n- Đỉnh sức kháng: `(Δᵤ, Fᵤ)`\n- Suy giảm: `(1,1Δᵤ; 0,2Fᵤ)` và `(1,25Δᵤ; 0,1Fᵤ)`\n- Thông số đổi theo 4 nhóm lò xo trong mục **Mô hình & lò xo**.")
            else:
                st.markdown("**Elastic tuyến tính**")
                st.markdown("Chế độ đàn hồi dùng `uniaxialMaterial('Elastic', …)` với độ cứng ban đầu `kₑ`. Lực luôn tỷ lệ với chuyển vị, không có chảy dẻo, suy giảm sức kháng hay năng lượng trễ. Dùng để kiểm tra chu kỳ riêng và đối chiếu đáp ứng đàn hồi.")
        with plot_col:
            matfig = go.Figure()
            dd = np.array([0.0, dy[0], dsu[0], 1.1 * dsu[0], 1.25 * dsu[0]])
            ff = np.array([0.0, fy[0], fu[0], .2 * fu[0], .1 * fu[0]]) / 1000
            matfig.add_trace(go.Scatter(x=np.r_[-dd[:0:-1], dd], y=np.r_[-ff[:0:-1], ff], mode="lines+markers", line=dict(color="#bd3344", width=3), name="Envelope HystereticSM"))
            matfig.add_trace(go.Scatter(x=[-dd[-1], 0, dd[-1]], y=[-ke[0] * dd[-1] / 1000, 0, ke[0] * dd[-1] / 1000], mode="lines", line=dict(color="#2b6cb0", width=2, dash="dash"), name="Elastic kₑ"))
            matfig.add_vline(x=dy[0], line_dash="dot", line_color="#777", annotation_text="Δᵧ")
            matfig.add_vline(x=dsu[0], line_dash="dot", line_color="#777", annotation_text="Δᵤ")
            matfig.update_layout(height=370, xaxis_title="Lateral spring deformation Δ (m)", yaxis_title="Spring force F (kN)", margin=dict(l=18, r=12, t=18, b=25), paper_bgcolor="white", plot_bgcolor="white", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(matfig, use_container_width=True)
            st.caption("Đường đỏ là envelope đơn điệu; vòng trễ hoàn chỉnh chỉ hình thành khi áp dụng lịch sử tải đảo chiều.")
        st.markdown("#### Phần tử lò xo và cản nhớt")
        st.markdown("Mỗi tầng có bốn lò xo ngang **song song**, mỗi lò xo nhận một vật liệu đàn hồi hoặc `HystereticSM`; bốn dashpot nhớt `Viscous` cũng đặt song song. Một phần tử `zeroLength` kết nối nút tầng dưới–tầng trên theo phương ngang. Lực phục hồi và lực cản được cộng để tạo lực tầng:")
        st.latex(r"V_i(\Delta_i,\dot\Delta_i)=\sum_{j=1}^{4}F_{s,ij}(\Delta_i,z_{ij})+\sum_{j=1}^{4}C_{ij}|\dot\Delta_i|^{\alpha}\,\mathrm{sgn}(\dot\Delta_i)")
        st.markdown("**Phạm vi diễn giải.** Đây là lò xo tổng hợp đại diện lực ngang tầng, không phải lò xo nền móng và cũng không phải khớp dẻo vật lý đặt ở đầu dầm/cột. Vì vậy mô hình phù hợp minh họa đáp ứng toàn cục shear-building; để khảo sát phá hoại cấu kiện khung cần fiber/beam-column elements, vật liệu bê tông-cốt thép, liên kết nút và điều kiện nền tảng được mô hình hóa riêng.")

    elif op10_view == "Cơ sở OpenSees":
        st.markdown("#### Chuỗi xây dựng mô hình bằng OpenSeesPy")
        st.markdown("Mô hình dùng đơn vị N–m–kg; `-ndm 1 -ndf 1` nghĩa là một phương vị ngang duy nhất trên mỗi nút.")
        st.code("""ops.wipe()
ops.model('basic', '-ndm', 1, '-ndf', 1)
ops.node(1, 0.0)                 # chân ngàm
ops.fix(1, 1)
for floor in range(1, 11):
    ops.node(floor + 1, 0.0)
    ops.mass(floor, M_total / 10)

# Tại mỗi tầng, tạo 4 cặp vật liệu lò xo + cản nhớt
ops.uniaxialMaterial('HystereticSM', matTag, '-posEnv', *KP,
                     '-negEnv', *KN, '-pinch', 1, 1)
ops.uniaxialMaterial('Elastic', matTag, k_e)  # lựa chọn đàn hồi
ops.uniaxialMaterial('Viscous', damperTag, C_i, 1.0)
ops.element('zeroLength', eleTag, lowerNode, upperNode,
            '-mat', matTag, damperTag, '-dir', 1, 1)
""", language="python")
        st.markdown("#### Ý nghĩa toán học")
        st.latex(r"\mathbf M\ddot{\mathbf u}+\mathbf C\dot{\mathbf u}+\mathbf f_s(\mathbf u,\mathbf z)=-\mathbf M\mathbf r\,a_g(t)")
        st.markdown("`u` là vector chuyển vị 10 tầng; `M` là ma trận khối lượng chéo; `C` là cản; `fₛ` là lực phục hồi phụ thuộc chuyển vị và trạng thái lịch sử `z`; `a_g(t)` là gia tốc nền. Trong miền đàn hồi, `fₛ = Ku`, và bài toán riêng là `Kφᵣ = ωᵣ²Mφᵣ`. Với mô hình shear-building, ma trận `K` là ma trận tam đường chéo lắp ghép từ độ cứng tầng.")
        st.latex(r"k_{i,\mathrm{story}}=\sum_{j=1}^{4}k_{ij};\qquad \omega_r=\sqrt{\lambda_r};\qquad T_r=\frac{2\pi}{\omega_r}")
        st.markdown("Lệnh `ops.eigen('-fullGenLapack', ...)` giải trị riêng. Rayleigh damping được hiệu chỉnh theo hai mode mục tiêu: `C = αM M + βK K`; phần tải động đất dùng `Path` time series và `UniformExcitation`, còn đáp ứng thời gian dùng Newmark.")

    elif op10_view == "Dao động":
        st.markdown("#### Dạng dao động đàn hồi và phổ chu kỳ")
        modal = go.Figure()
        floors = np.arange(1, 11)
        for j in range(3):
            modal.add_trace(go.Scatter(x=mode_shapes[:, j], y=floors, mode="lines+markers", name=f"Mode {j+1} · T={periods[j]:.3f}s", line=dict(width=3)))
        modal.update_layout(height=370, xaxis_title="Dạng chuẩn hóa (u mái = 1)", yaxis_title="Tầng", yaxis=dict(dtick=1, gridcolor="#e6ebef"), margin=dict(l=15, r=15, t=15, b=15), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(modal, use_container_width=True)
        st.caption("Tính từ ma trận shear-building đàn hồi lắp ghép bằng bốn độ cứng cột mỗi tầng và khối lượng phân đều; đây là kết quả lý thuyết của idealization nguồn, chưa chạy lại OpenSees.")
        periods_fig = go.Figure(go.Bar(x=np.arange(1, 11), y=periods, marker_color="#528b70"))
        periods_fig.update_layout(height=260, title="Chu kỳ riêng các mode đàn hồi", xaxis_title="Mode r", yaxis_title="Tᵣ (s)", margin=dict(l=15, r=15, t=45, b=15), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(periods_fig, use_container_width=True)
        st.markdown("#### Ước tính cản nhớt tại tầng")
        st.latex(r"C_i=2\,\xi_i\,\omega_i\,m_i;\qquad \xi_{Rayleigh,r}=\frac{1}{2}\left(\frac{\alpha_M}{\omega_r}+\beta_K\omega_r\right)")
        st.caption("Mỗi tầng có tỷ số cản riêng để quy đổi cản nhớt cho phần tử zeroLength. Phân tích động đất có hiệu chỉnh Rayleigh theo các mode mục tiêu.")

        st.divider()
        st.markdown("#### Đáp ứng mô phỏng 10 tầng · hình biến dạng 3D")
        st.caption("Mô hình khung shear-building 10 tầng: xanh dương là trạng thái ban đầu, cam là biến dạng tức thời, các nút tím đánh dấu tầng đã chảy. Chuyển vị hiển thị được khuếch đại để quan sát.")
        amp_col, damp_col = st.columns(2)
        with amp_col:
            visual_amp = st.slider("Khuếch đại chuyển vị khi vẽ", 1.0, 50.0, 15.0, 1.0, key="op10_visual_amp")
        with damp_col:
            sim_damping = st.slider("Tỷ số cản modal (%)", 0.0, 10.0, 3.0, 0.5, key="op10_sim_damping") / 100
        nonlinear_demo = st.checkbox("Mô phỏng phi tuyến có vòng trễ", value=True, key="op10_nonlinear_demo")
        selected_step = st.slider("Thời điểm xem đáp ứng (s)", 0.0, 12.0, 0.0, 0.02, key="op10_response_time")
        dt_sim = 0.02
        t_sim = np.arange(0.0, 12.0 + dt_sim, dt_sim)
        pulse = np.sin(np.pi * np.clip(t_sim / 6.0, 0, 1)) ** 2
        ag_sim = 0.15 * 9.80665 * pulse * np.sin(2 * np.pi * t_sim / first_period)
        if nonlinear_demo:
            # Backbone values are from the source example; map the four parallel
            # spring groups to one equivalent story spring in the reduced model.
            fy_story = np.full(10, float(np.sum(fy)))
            dy_story = fy_story / story_k
            du_story = np.maximum(np.full(10, float(np.max(dsu))), dy_story * 1.05)
            try:
                response = inelastic_mdof_response(
                    np.full(10, mass_each), story_k, sim_damping, ag_sim, dt_sim,
                    dy_story, hardening_ratio=0.02, failure_drifts=du_story,
                    yield_forces=fy_story,
                )
            except ValueError as exc:
                st.warning(f"Mô hình phi tuyến không hội tụ với tham số hiện tại ({exc}); đang hiển thị đáp ứng đàn hồi tuyến tính.")
                response = mdof_response(np.full(10, mass_each), story_k, sim_damping, ag_sim, dt_sim)
                nonlinear_demo = False
        else:
            response = mdof_response(np.full(10, mass_each), story_k, sim_damping, ag_sim, dt_sim)
        sim_cols = st.columns(3)
        sim_cols[0].metric("Gia tốc nền đỉnh", f"{np.max(np.abs(ag_sim))/9.80665:.2f} g")
        sim_cols[1].metric("Chuyển vị mái cực đại", f"{np.max(np.abs(response['displacement'][:, -1]))*1000:.1f} mm")
        sim_cols[2].metric("Lệch tầng lớn nhất", f"{np.max(np.abs(np.diff(np.c_[np.zeros(len(t_sim)), response['displacement']], axis=1)))*1000:.1f} mm")

        # Build a frame-like 3D view with diaphragms and yielded-story markers.
        plan_nodes = [(0, 0), (3, 0), (3, 1.5), (0, 1.5)]
        vertical_edges = [(i, i) for i in range(4)]
        horizontal_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        nlevels = 11
        edge_x, edge_y, edge_z = [], [], []
        node_x, node_y, node_z = [], [], []
        selected_idx = int(np.clip(round(selected_step / dt_sim), 0, len(t_sim)-1))
        drift_history = response.get("story_drift")
        yielded_stories = (np.abs(drift_history[selected_idx]) >= dy_story) if drift_history is not None and nonlinear_demo else np.zeros(10, dtype=bool)
        def frame_geometry(step_index, deformed=True):
            u = np.r_[0.0, response["displacement"][step_index]] * visual_amp if deformed else np.zeros(nlevels)
            ex, ey, ez, nx, ny, nz = [], [], [], [], [], []
            hx, hy, hz = [], [], []
            for level in range(nlevels):
                shift = u[level]
                for px, py in plan_nodes:
                    nx.append(px + shift); ny.append(py); nz.append(float(level))
                for a, b in horizontal_edges:
                    ex.extend([plan_nodes[a][0] + shift, plan_nodes[b][0] + shift, None])
                    ey.extend([plan_nodes[a][1], plan_nodes[b][1], None]); ez.extend([level, level, None])
            for level in range(nlevels - 1):
                for a, b in vertical_edges:
                    ex.extend([plan_nodes[a][0] + u[level], plan_nodes[b][0] + u[level + 1], None])
                    ey.extend([plan_nodes[a][1], plan_nodes[b][1], None]); ez.extend([level, level + 1, None])
                    if deformed and level < 10 and yielded_stories[level]:
                        hx.append(plan_nodes[a][0] + (u[level] + u[level+1]) / 2)
                        hy.append(plan_nodes[a][1]); hz.append(level + 0.5)
            return ex, ey, ez, nx, ny, nz, hx, hy, hz

        # Decimate frames while retaining the complete response for slider/history plots.
        frame_ids = np.unique(np.r_[np.arange(0, len(t_sim), 10), len(t_sim) - 1]).astype(int)
        init_step = int(selected_idx)
        initial_deformed = frame_geometry(init_step, True)
        original_lines = frame_geometry(0, False)
        sim_fig = go.Figure(data=[
            go.Scatter3d(x=original_lines[0], y=original_lines[1], z=original_lines[2], mode="lines", line=dict(color="#2458cc", width=4), name="Ban đầu"),
            go.Scatter3d(x=original_lines[3], y=original_lines[4], z=original_lines[5], mode="markers", marker=dict(color="#2458cc", size=3), showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=initial_deformed[0], y=initial_deformed[1], z=initial_deformed[2], mode="lines", line=dict(color="#d77835", width=6), name="Biến dạng"),
            go.Scatter3d(x=initial_deformed[3], y=initial_deformed[4], z=initial_deformed[5], mode="markers", marker=dict(color="#d82e2e", size=3), showlegend=False, hoverinfo="skip"),
            go.Scatter3d(x=initial_deformed[6], y=initial_deformed[7], z=initial_deformed[8], mode="markers", marker=dict(color="#bf39b9", size=8, line=dict(color="white",width=1)), name="Tầng vượt ngưỡng chảy"),
        ])
        frames = []
        for idx in frame_ids:
            geom = frame_geometry(int(idx), True)
            frames.append(go.Frame(name=str(idx), data=[
                go.Scatter3d(x=geom[0], y=geom[1], z=geom[2]),
                go.Scatter3d(x=geom[3], y=geom[4], z=geom[5]),
                go.Scatter3d(x=geom[6], y=geom[7], z=geom[8]),
            ], traces=[2, 3, 4]))
        sim_fig.frames = frames
        max_shift = max(float(np.max(np.abs(response["displacement"]))) * visual_amp * 1.25, 0.3)
        sim_fig.update_layout(height=520, margin=dict(l=0, r=0, t=25, b=0), scene=dict(xaxis=dict(title="X (khuếch đại)", range=[-max_shift, 3 + max_shift]), yaxis=dict(title="Y", range=[-0.2, 1.7]), zaxis=dict(title="Tầng", range=[0, 10]), aspectmode="manual", aspectratio=dict(x=1.1, y=0.8, z=1.7), camera=dict(eye=dict(x=1.55, y=-1.75, z=1.25))), legend=dict(orientation="h", y=1.02), updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0, buttons=[dict(label="▶ Phát", method="animate", args=[None, {"frame": {"duration": 80, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]), dict(label="⏸ Dừng", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}])])], sliders=[dict(steps=[dict(method="animate", args=[[str(int(i))], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}], label=f"{t_sim[i]:.1f}s") for i in frame_ids], x=0.2, len=0.74)])
        left3, right3 = st.columns([0.58, 0.42])
        with left3:
            st.plotly_chart(sim_fig, use_container_width=True)
        with right3:
            hist = go.Figure()
            for floor_id in [0, 4, 9]:
                hist.add_trace(go.Scatter(x=response["time"], y=response["displacement"][:, floor_id] * 1000, mode="lines", name=f"Tầng {floor_id+1}"))
            hist.update_layout(height=270, title="Chuyển vị tương đối theo thời gian", xaxis_title="t (s)", yaxis_title="u (mm)", margin=dict(l=12, r=8, t=45, b=30), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(hist, use_container_width=True)
            agfig = go.Figure(go.Scatter(x=t_sim, y=ag_sim / 9.80665, mode="lines", line=dict(color="#3d6b55", width=1.5)))
            agfig.update_layout(height=210, title="Gia tốc nền minh họa", xaxis_title="t (s)", yaxis_title="a₉ (g)", margin=dict(l=12, r=8, t=45, b=20), paper_bgcolor="white", plot_bgcolor="white")
            st.plotly_chart(agfig, use_container_width=True)
        if nonlinear_demo:
            st.caption("Đáp ứng dùng mô hình shear-building phi tuyến song tuyến tính (Newmark + cân bằng bước); cốt truyện tải là xung điều hòa minh họa 0,15g, không phải bản ghi động đất. Khung 3D chỉ trực quan hóa chuyển vị tầng; màu đỏ được khuếch đại để nhìn rõ.")
            st.markdown("#### Vòng trễ lực cắt đáy – chuyển vị mái")
            roof_mm = response["displacement"][:, -1] * 1000
            base_shear_kn = np.sum(response["story_shear"], axis=1) / 1000
            hyst_fig = go.Figure()
            hyst_fig.add_trace(go.Scatter(x=roof_mm, y=base_shear_kn, mode="lines", name="Quỹ đạo tích lũy", line=dict(color="#d77835", width=2)))
            hyst_fig.add_trace(go.Scatter(x=[roof_mm[0]], y=[base_shear_kn[0]], mode="markers", name="Bước tải hiện tại", marker=dict(size=10, color="#1596b5", line=dict(color="white", width=1.5))))
            hyst_frames = []
            # Keep all points for the accumulated loop while limiting animation payload.
            hyst_ids = np.unique(np.r_[np.arange(0, len(t_sim), 5), len(t_sim)-1]).astype(int)
            for idx in hyst_ids:
                hyst_frames.append(go.Frame(name=str(idx), data=[
                    go.Scatter(x=roof_mm[:idx+1], y=base_shear_kn[:idx+1]),
                    go.Scatter(x=[roof_mm[idx]], y=[base_shear_kn[idx]]),
                ], traces=[0, 1]))
            hyst_fig.frames = hyst_frames
            hyst_fig.update_layout(height=490, title="Vòng V–u theo lịch sử tải động đất", xaxis_title="Chuyển vị mái (mm)", yaxis_title="Lực cắt đáy (kN)", margin=dict(l=20,r=20,t=55,b=20), paper_bgcolor="white", plot_bgcolor="white", xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="#697783"), yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor="#697783"), updatemenus=[dict(type="buttons", showactive=False, x=0, y=1.16, buttons=[dict(label="▶ Phát theo thời gian", method="animate", args=[None,{"frame":{"duration":35,"redraw":True},"transition":{"duration":0},"fromcurrent":True}]),dict(label="⏸ Tạm dừng",method="animate",args=[[None],{"mode":"immediate","frame":{"duration":0,"redraw":False}}])])], sliders=[dict(steps=[dict(method="animate",args=[[str(int(i))],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}],label=f"{t_sim[i]:.2f}s") for i in hyst_ids], x=0.12,len=0.82, currentvalue=dict(prefix="Thời gian gia tải: "))])
            st.plotly_chart(hyst_fig, use_container_width=True)
            st.caption("Đường được dựng trực tiếp từ lực cắt các tầng và chuyển vị mái ở từng bước thời gian; không dùng dữ liệu giả lập vòng. Mô hình song tuyến tính hiện chưa có luật pinching, nên vòng trễ chưa thắt như ảnh mẫu. Muốn khớp pinching cần hiệu chuẩn tham số vật liệu/lịch sử chu kỳ.")
        else:
            st.caption("Đang hiển thị đáp ứng đàn hồi tuyến tính Newmark dưới xung sin bao biên 0,15g. Khung đỏ được khuếch đại; đồ thị thời gian dùng chuyển vị tính thật.")

    elif op10_view == "Phi tuyến":
        st.markdown("#### Luật vật liệu trễ nhiều đoạn")
        disp = np.array([0, dy[0], dsu[0], 1.1 * dsu[0], 1.25 * dsu[0]])
        force = np.array([0, fy[0], fu[0], 0.2 * fu[0], 0.1 * fu[0]])
        hyst = go.Figure()
        hyst.add_trace(go.Scatter(x=np.r_[-disp[:0:-1], disp], y=np.r_[-force[:0:-1], force], mode="lines+markers", name="Backbone đối xứng", line=dict(color="#c83d3d", width=3)))
        hyst.add_vline(x=dy[0], line_dash="dash", annotation_text="Δᵧ")
        hyst.add_vline(x=dsu[0], line_dash="dash", annotation_text="Δᵤ")
        hyst.update_layout(height=300, xaxis_title="Chuyển vị lò xo (m)", yaxis_title="Lực lò xo (kN)", margin=dict(l=15, r=15, t=25, b=15), paper_bgcolor="white", plot_bgcolor="white")
        hyst.update_traces(y=np.r_[-force[:0:-1], force] / 1000)
        st.plotly_chart(hyst, use_container_width=True)
        st.caption("Đồ thị vẽ các điểm backbone đối xứng: (0,0), (Δᵧ,Fᵧ), (Δᵤ,Fᵤ), (1.1Δᵤ,0.2Fᵤ), (1.25Δᵤ,0.1Fᵤ). Vòng trễ phụ thuộc lịch sử tải và được tạo bởi vật liệu Hysteretic của OpenSees.")
        st.markdown("#### Tám nhánh phân tích trong OpenSees")
        st.dataframe(pd.DataFrame([
            ("PERIOD", "Trị riêng đàn hồi, chu kỳ mode"),
            ("STATIC", "Thiết lập trạng thái tải tĩnh"),
            ("PUSHOVER", "Điều khiển chuyển vị, đường cong lực cắt đáy – chuyển vị mái"),
            ("CYCLIC_DISPLACEMENT", "Tải chuyển vị chu kỳ, vòng trễ và năng lượng tiêu tán"),
            ("STATIC_EXTERNAL_TIME-DEPENDENT_LOADING", "Tải lực tuần hoàn phụ thuộc thời gian, phân tích tĩnh từng bước"),
            ("DYNAMIC_EXTERNAL_TIME-DEPENDENT_LOADING", "Tải động lực học theo thời gian"),
            ("FREE-VIBRATION", "Dao động tự do, suy giảm logarit và tỷ số cản"),
            ("SEISMIC", "Kích động nền UniformExcitation, cản Rayleigh, Newmark"),
        ], columns=["Phân tích", "Mục tiêu / đại lượng quan sát"]), hide_index=True, use_container_width=True)
        st.markdown("#### Chỉ số hiệu năng được xây dựng từ đáp ứng")
        st.latex(r"\mu=\frac{\Delta_{max}}{\Delta_y};\qquad E_D=\oint F\,d\Delta;\qquad \xi_{eq}=\frac{E_D}{4\pi E_S};\qquad EDCI=\frac{E_{earthquake}}{E_{cyclic,capacity}}")
        st.markdown("Pushover cung cấp đường cong sức kháng và độ dẻo; phân tích chu kỳ/động đất ghi đáp ứng chuyển vị, lực, vận tốc, gia tốc, chu kỳ biến đổi và năng lượng. Các đại lượng này mô tả mô hình shear-building theo những giả thiết đã nêu, không trực tiếp suy ra thời điểm phá hoại thực của công trình.")
        st.info("Vòng trễ động lực học thực và bước tải điều khiển được hiển thị trong nội dung ‘Dao động’. Phần backbone ở trên chỉ mô tả đường bao sức kháng; nó không phải kết quả chu kỳ OpenSees.")

    st.divider()
    st.markdown("#### Giới hạn của phép minh họa")
    st.markdown("Đây là mô hình shear-building lý tưởng hóa: mỗi tầng có một bậc tự do ngang, khối lượng tập trung, bốn nhánh lò xo/cản và tường shear-wall giả thiết EI vô hạn. Mô hình không biểu diễn xoắn, khung dầm-cột phân bố, cốt thép, móng hay đất nền. Chu kỳ và mode trong tab được tính từ ma trận đàn hồi theo thông số đã trình bày; đánh giá công trình thực cần mô hình cấu kiện, tải trọng và tiêu chuẩn phù hợp.")


def _tab_mdof_dynamics():
    st.subheader("Input mô hình 3D và mô phỏng dao động nhà nhiều tầng")
    a, b = st.columns([0.34, 0.66])
    with a:
        floors = st.number_input("Số tầng", min_value=2, max_value=20, value=4, step=1, key="mdof_floors")
        st.caption("Tất cả input được phản ánh đồng thời trên mô hình 3D bên phải.")
        shape = st.selectbox("Hình dạng mặt bằng", ["Chữ nhật", "Chữ L", "Chữ T"], key="bldg_shape")
        area = st.number_input("Diện tích sàn mỗi tầng (m²)", min_value=4.0, value=400.0, step=10.0, key="bldg_area")
        span_x = st.number_input("Chiều dài nhịp X (m)", min_value=2.0, value=5.0, step=0.5, key="bldg_span_x")
        span_y = st.number_input("Chiều dài nhịp Y (m)", min_value=2.0, value=5.0, step=0.5, key="bldg_span_y")
        story_height = st.number_input("Chiều cao tầng (m)", min_value=2.0, value=3.3, step=0.1, key="bldg_height")
        aspect = st.number_input("Tỷ lệ cạnh X/Y", min_value=0.25, max_value=4.0, value=1.0, step=0.1, key="bldg_aspect")
        target_x, target_y = np.sqrt(area * aspect), np.sqrt(area / aspect)
        bays_x = int(np.clip(round(target_x / span_x), 1, 8))
        bays_y = int(np.clip(round(target_y / span_y), 1, 8))
        bay_x, bay_y = float(span_x), float(span_y)
        st.caption(f"Lưới FEM: {bays_x} × {bays_y} nhịp · diện tích hình học xấp xỉ {bays_x*bay_x*bays_y*bay_y:.1f} m²/tầng")
        shape_code = {"Chữ nhật": 0, "Chữ L": 1, "Chữ T": 2}[shape]
        st.markdown("**Tiết diện cấu kiện**")
        col_type = st.selectbox("Tiết diện cột", ["Chữ nhật", "Ống tròn"], key="col_section_type")
        col_b = st.number_input("Cột: cạnh b (m) / đường kính (m)", min_value=0.1, value=0.5, step=0.05, key="col_b")
        col_h = st.number_input("Cột: cạnh h (m)", min_value=0.1, value=0.5, step=0.05, key="col_h")
        col_bars = st.number_input("Cột: số thanh dọc (chẵn)", min_value=4, max_value=20, value=8, step=2, key="col_bars")
        col_bar_d = st.number_input("Cột: đường kính thép dọc (mm)", min_value=10, max_value=32, value=20, step=2, key="col_bar_d") / 1000
        col_tie_d = st.number_input("Cột: đường kính đai (mm)", min_value=6, max_value=16, value=10, step=2, key="col_tie_d") / 1000
        col_tie_spacing = st.number_input("Cột: bước đai (mm)", min_value=50, max_value=300, value=100, step=10, key="col_tie_spacing") / 1000
        beam_type = st.selectbox("Tiết diện dầm", ["Chữ nhật", "Chữ I tương đương"], key="beam_section_type")
        beam_b = st.number_input("Dầm: b (m)", min_value=0.1, value=0.3, step=0.05, key="beam_b")
        beam_h = st.number_input("Dầm: h (m)", min_value=0.1, value=0.5, step=0.05, key="beam_h")
        beam_bars_top = st.number_input("Dầm: số thép dọc lớp trên", min_value=2, max_value=10, value=3, step=1, key="beam_bars_top")
        beam_bars_bottom = st.number_input("Dầm: số thép dọc lớp dưới", min_value=2, max_value=10, value=3, step=1, key="beam_bars_bottom")
        beam_bar_d = st.number_input("Dầm: đường kính thép dọc (mm)", min_value=10, max_value=32, value=20, step=2, key="beam_bar_d") / 1000
        beam_tie_d = st.number_input("Dầm: đường kính đai (mm)", min_value=6, max_value=16, value=8, step=2, key="beam_tie_d") / 1000
        beam_tie_spacing = st.number_input("Dầm: bước đai (mm)", min_value=50, max_value=300, value=100, step=10, key="beam_tie_spacing") / 1000
        rebar_cover = st.number_input("Lớp bê tông bảo vệ (m)", min_value=0.02, max_value=0.08, value=0.04, step=0.005, format="%.3f", key="rc_cover")
        E_gpa = st.number_input("Môđun đàn hồi E (GPa)", min_value=1.0, value=25.0, step=1.0, key="bldg_E")
        concrete_grade = st.selectbox("Cấp độ bền bê tông (TCVN 5574:2018)", ["B20", "B25", "B30", "B35", "B40", "B45", "B50", "B60"], index=2, key="bldg_concrete_grade")
        steel_grade = st.selectbox("Cốt thép dọc", ["CB300-V", "CB400-V", "CB500-V"], index=1, key="bldg_steel_grade")
        if st.checkbox("Dùng mô đun bê tông theo TCVN 5574:2018", value=True, key="bldg_use_tcvn_E"):
            E_gpa = {"B20":27.5,"B25":30.0,"B30":32.5,"B35":34.5,"B40":36.0,"B45":37.0,"B50":38.0,"B60":39.5}[concrete_grade]
        rebar_yield_design = {"CB300-V":260.0,"CB400-V":350.0,"CB500-V":435.0}[steel_grade]
        rebar_yield_strain = rebar_yield_design / 200000.0
        concrete_ultimate_strain = 0.0035  # TCVN 5574:2018, 6.1.4.2, concrete <= B60
        # Principal inertia about the frame bending axis from entered sections.
        concrete_E = E_gpa * 1e9
        steel_E = 200e9
        if col_type == "Ống tròn":
            col_props = rc_circular_column_properties(col_b, concrete_E, steel_E, col_bar_d, int(col_bars), rebar_cover, rebar_yield_design*1e6)
            col_A = np.pi * col_b**2 / 4
            col_depth = col_b
        else:
            col_props = rc_section_properties(col_b, col_h, concrete_E, steel_E, col_bar_d, int(col_bars)//2, int(col_bars)//2, rebar_cover, rebar_yield_design*1e6)
            col_A = col_b * col_h
            col_depth = col_h
        beam_props = rc_section_properties(beam_b, beam_h, concrete_E, steel_E, beam_bar_d, int(beam_bars_top), int(beam_bars_bottom), rebar_cover, rebar_yield_design*1e6)
        # Stirrups increase the section strain limit in this reduced model;
        # this transparent confinement heuristic is not a code equation.
        tie_ratio = min(col_tie_spacing/max(min(col_b,col_h),0.05), 1.0)
        confined_eps_cu = concrete_ultimate_strain * (1.0 + 0.35 * max(0.0, 1.0-tie_ratio))
        col_I, beam_I = col_props["effective_I"], beam_props["effective_I"]
        st.caption(f"Ieff cột = {col_I:.4g} m⁴ · Ieff dầm = {beam_I:.4g} m⁴ · cốt dọc/đai được đưa vào FEM và mô hình chảy dẻo")
        analysis_method = st.selectbox(
            "Phương pháp phân tích động đất",
            [
                "Tĩnh lực ngang tương đương | Equivalent Lateral Force (ELF)",
                "Phổ phản ứng dạng dao động | Modal Response Spectrum Analysis (MRSA)",
                "Lịch sử thời gian | Time-History Analysis (THA)",
            ], key="mdof_earthquake_method",
        )
        st.caption("ELF và MRSA có sẵn phổ đàn hồi mẫu Type 1 theo TCVN 9386; người dùng cũng có thể tự nhập Sa(T₁) hoặc nạp phổ riêng. THA nhận bản ghi theo thời gian hoặc tín hiệu minh họa QCVN 02.")
        with st.expander("Phân biệt ba phương pháp phân tích động đất"):
            st.markdown("""
            - **Tĩnh lực ngang tương đương · Equivalent Lateral Force (ELF):** dùng một lực ngang tĩnh phân bố theo khối lượng và cao độ tầng; phù hợp khi điều kiện áp dụng của tiêu chuẩn được thỏa mãn.
            - **Phổ phản ứng dạng dao động · Modal Response Spectrum Analysis (MRSA):** tính đáp ứng đàn hồi từng mode từ phổ thiết kế rồi tổ hợp cực đại bằng SRSS hoặc CQC; không tạo lịch sử theo thời gian.
            - **Lịch sử thời gian · Time-History Analysis (THA):** tích phân đáp ứng từng bước theo gia tốc nền; mô hình hiện tại dùng shear-building phi tuyến giảm bậc và Newmark–Newton.

            Cả ba dùng độ cứng tầng rút gọn từ FEM khung không gian. ELF và MRSA là kết quả đàn hồi; THA có thể phi tuyến trong mô hình giảm bậc. Dữ liệu phổ/động đất phải phù hợp với tiêu chuẩn thiết kế, nền đất và công trình.
            """)
        damping = st.number_input("Tỷ số cản mode ξ", min_value=0.0, max_value=1.0, value=0.05, step=0.01, key="mdof_damping")
        floor_mass = st.number_input("Khối lượng sàn + tải quy đổi (kg/m²)", min_value=50.0, value=500.0, step=25.0, key="floor_mass_area") * area
        dt, ag, qcvn_record = 0.01, None, None
        excitation_mode, combination, spectrum_unit = "", "CQC", "m/s²"
        sa_t1, lambda_factor, spectrum_periods, spectrum_sa = None, 1.0, None, None
        factor = 1.0
        spectrum_signature, spectrum_model_parameters = None, None
        spectrum_source = ""

        def load_tcvn_demo_spectrum(key_prefix):
            loc1, loc2 = st.columns(2)
            province_options = qcvn02_provinces()
            default_province = "Thành phố Hà Nội" if "Thành phố Hà Nội" in province_options else province_options[0]
            with loc1:
                province = st.selectbox("Tỉnh / thành phố theo QCVN 02:2021", province_options,
                                        index=province_options.index(default_province), key=f"{key_prefix}_province")
            with loc2:
                admin_options = qcvn02_places(province)
                default_admin = "Quận Ba Đình" if province == "Thành phố Hà Nội" and "Quận Ba Đình" in admin_options else admin_options[0]
                admin = st.selectbox("Quận / huyện / địa danh", admin_options,
                                     index=admin_options.index(default_admin), key=f"{key_prefix}_admin")
            record = qcvn02_lookup(province, admin)
            soil = st.selectbox("Loại nền theo phổ Type 1", ["A", "B", "C", "D", "E"], index=1, key=f"{key_prefix}_soil")
            gamma_i = st.number_input("Hệ số tầm quan trọng γI", min_value=0.1, max_value=3.0, value=1.0, step=0.1, key=f"{key_prefix}_gamma_i")
            ag_ref = record["agR_g"] * 9.80665
            sample_T = np.geomspace(0.001, 20.0, 800)
            sample_Sa = tcvn9386_type1_elastic_spectrum(sample_T, ag_ref, soil, gamma_i, damping)
            S, TB, TC, TD = TYPE1_GROUND_PARAMETERS[soil]
            st.metric("a₍gR₎ QCVN 02 · nền A, 500 năm", f"{record['agR_g']:.3f} g")
            st.caption(f"Phổ mẫu Type 1, nền {soil}: S={S:.2f}, TB={TB:.2f}s, TC={TC:.2f}s, TD={TD:.2f}s; γI={gamma_i:.2f}; ξ={damping*100:.1f}%. Dùng a₍gR₎ tại {admin}, {province} (Bảng 6.1, trang {record['source_page']}) để dựng phổ đàn hồi q=1.")
            st.info("Đây là phổ mẫu đàn hồi để chạy thử, theo dạng TCVN 9386-1:2025 / EN 1998-1 Type 1; không giảm theo hệ số ứng xử q. Khi thiết kế, kỹ sư cần dùng phổ thiết kế phù hợp loại động đất, nền đất, cấp dẻo, q và các yêu cầu của tiêu chuẩn.")
            params = (record["province"], record["admin"], float(record["agR_g"]), soil, float(gamma_i), float(damping))
            return record, sample_T, sample_Sa, params

        if analysis_method.startswith("Tĩnh lực ngang"):
            spectrum_source = st.radio("Nguồn gia tốc phổ tại T₁", ["Phổ mẫu TCVN 9386-1:2025 · Type 1", "Tự nhập Sa(T₁) từ phổ riêng"], horizontal=True, key="mdof_elf_spectrum_source")
            if spectrum_source.startswith("Phổ mẫu"):
                qcvn_record, spectrum_periods, spectrum_sa, spectrum_model_parameters = load_tcvn_demo_spectrum("mdof_elf_demo")
                spectrum_signature = hash(np.asarray([spectrum_periods, spectrum_sa]).tobytes())
            else:
                sa_unit = st.selectbox("Đơn vị Sa(T₁)", ["g", "m/s²"], key="mdof_elf_sa_unit")
                sa_t1_input = st.number_input("Sa(T₁) từ phổ thiết kế riêng", min_value=0.0, value=0.0, step=0.01, format="%.4f", key="mdof_elf_sa")
                sa_t1 = sa_t1_input * 9.80665 if sa_unit == "g" else sa_t1_input
            lambda_factor = st.number_input("Hệ số λ theo tiêu chuẩn áp dụng", min_value=0.1, max_value=1.5, value=1.0, step=0.05, key="mdof_elf_lambda")
            if spectrum_source.startswith("Tự nhập"):
                st.caption("Nhập Sa(T₁) theo phổ thiết kế riêng; T₁ được chương trình tìm từ ma trận FEM khi chạy phân tích.")
        elif analysis_method.startswith("Phổ phản ứng"):
            spectrum_source = st.radio("Nguồn phổ phản ứng", ["Phổ mẫu TCVN 9386-1:2025 · Type 1", "Nạp phổ thiết kế riêng (CSV)"], horizontal=True, key="mdof_mrsa_spectrum_source")
            combination = st.selectbox("Tổ hợp đáp ứng modal", ["CQC", "SRSS"], key="mdof_spectrum_combination")
            if spectrum_source.startswith("Phổ mẫu"):
                qcvn_record, spectrum_periods, spectrum_sa, spectrum_model_parameters = load_tcvn_demo_spectrum("mdof_mrsa_demo")
                spectrum_signature = hash(np.asarray([spectrum_periods, spectrum_sa]).tobytes())
                preview = go.Figure(go.Scatter(x=spectrum_periods,y=spectrum_sa/9.80665,mode="lines",line=dict(color="#315c76",width=2)))
                preview.update_layout(title="Phổ đàn hồi mẫu Type 1 · q=1",xaxis_title="T (s)",yaxis_title="Sa (g)",height=260,margin=dict(l=10,r=10,t=40,b=20))
                st.plotly_chart(preview,use_container_width=True)
            else:
                spectrum_upload = st.file_uploader("Phổ thiết kế riêng CSV (2 cột: T[s], Sa)", type=["csv", "txt"], key="mdof_spectrum_upload")
                spectrum_unit = st.selectbox("Đơn vị Sa trong file phổ", ["m/s²", "g"], key="mdof_spectrum_unit")
                st.caption("File không tiêu đề, gồm hai cột số T(s), Sa; ít nhất hai hàng, T tăng dần và Sa không âm. Không ngoại suy ngoài dải chu kỳ đã nạp.")
                if spectrum_upload is not None:
                    try:
                        spectrum_table = np.asarray(np.loadtxt(spectrum_upload, delimiter="," if spectrum_upload.name.lower().endswith(".csv") else None, dtype=float))
                        if spectrum_table.ndim != 2 or spectrum_table.shape[1] < 2:
                            raise ValueError("Cần hai cột T và Sa.")
                        spectrum_periods, spectrum_sa = spectrum_table[:, 0], spectrum_table[:, 1]
                        if spectrum_unit == "g":
                            spectrum_sa = spectrum_sa * 9.80665
                        spectrum_signature = hash(np.asarray(spectrum_table[:, :2], dtype=float).tobytes())
                        st.caption(f"Đã nạp {len(spectrum_periods)} điểm phổ: T = {spectrum_periods.min():.3f}–{spectrum_periods.max():.3f} s.")
                    except Exception as exc:
                        spectrum_periods = spectrum_sa = None
                        st.error(f"Không đọc được phổ phản ứng: {exc}")
        else:
            excitation_mode = st.radio(
                "Nguồn gia tốc nền",
                ["Tra địa điểm theo QCVN 02-2021 · tín hiệu minh họa", "Nạp bản ghi gia tốc thực"],
                horizontal=True, key="mdof_excitation_mode",
            )
            dt = st.number_input("Bước thời gian Δt (s)", min_value=0.0001, value=0.01, format="%.4f", key="mdof_dt")
            if excitation_mode.startswith("Tra địa điểm"):
                loc1, loc2 = st.columns(2)
                with loc1:
                    province = st.selectbox("Tỉnh / thành phố (tên theo QC 2021)", qcvn02_provinces(), key="mdof_qcvn_province")
                with loc2:
                    admin = st.selectbox("Quận / huyện / địa danh", qcvn02_places(province), key="mdof_qcvn_admin")
                qcvn_record = qcvn02_lookup(province, admin)
                gravity = 9.80665
                ref_pga = qcvn_record["agR_g"] * gravity
                factor = st.number_input(
                    "Hệ số hiệu chỉnh theo tiêu chuẩn thiết kế áp dụng",
                    min_value=0.1, max_value=5.0, value=1.0, step=0.05, format="%.2f",
                    help="QCVN 02 cung cấp agR tham chiếu trên nền A, chu kỳ lặp 500 năm. Nếu tiêu chuẩn thiết kế yêu cầu nền đất hoặc chu kỳ lặp khác, kỹ sư phải nhập hệ số quy đổi có căn cứ; giá trị 1.00 giữ nguyên agR tham chiếu.",
                    key="mdof_qcvn_factor",
                )
                pga = ref_pga * factor
                q1, q2, q3 = st.columns(3)
                q1.metric("a₍gR₎ tham chiếu QC02", f"{qcvn_record['agR_g']:.3f} g")
                q2.metric("PGA quy đổi đang dùng", f"{pga:.3f} m/s²")
                q3.metric("Phổ tham khảo Sₛ / S₁", f"{qcvn_record.get('Ss_g', float('nan')):.2f} / {qcvn_record.get('S1_g', float('nan')):.2f} g")
                st.caption(f"Nguồn: QCVN 02:2021/BXD, Bảng 6.1 trang {qcvn_record['source_page']} (a₍gR₎); Bảng 6.2 trang {qcvn_record.get('spectrum_source_page', '—')} (Sₛ/S₁). a₍gR₎ là PGA tham chiếu trên nền A, chu kỳ lặp 500 năm. Sₛ/S₁ là phổ tham khảo trên nền B, chu kỳ lặp 2.500 năm.")
                st.warning("QCVN 02 không cấp bản ghi gia tốc theo thời gian. Chế độ tra cứu tạo sóng điều hòa bao dạng với PGA tương ứng chỉ để minh họa; không phải bản ghi thực, không khớp phổ và không dùng thay hồ sơ thiết kế.")
                duration = st.number_input("Thời lượng tín hiệu minh họa (s)", min_value=1.0, value=12.0, step=1.0, key="mdof_qcvn_duration")
                excitation_period = st.number_input("Chu kỳ sóng điều hòa minh họa (s)", min_value=0.05, value=1.0, step=0.1, key="mdof_qcvn_period")
                if st.checkbox("Tạo tín hiệu minh họa để chạy mô phỏng", value=True, key="mdof_qcvn_generate"):
                    nt = max(2, int(round(duration / dt)) + 1)
                    tt = np.arange(nt) * dt
                    envelope = np.sin(np.pi * np.clip(tt / duration, 0.0, 1.0)) ** 2
                    demo = envelope * np.sin(2.0 * np.pi * tt / excitation_period)
                    peak = float(np.max(np.abs(demo)))
                    ag = demo * (pga / peak) if peak > 0 else np.zeros_like(demo)
                    st.caption(f"Tín hiệu minh họa được chuẩn hóa để có PGA = {np.max(np.abs(ag)):.3f} m/s².")
            else:
                upload = st.file_uploader("Bản ghi gia tốc (m/s²), một cột số theo thời gian", type=["txt", "csv"], key="mdof_upload")
                if upload is not None:
                    try:
                        ag = np.asarray(np.loadtxt(upload, delimiter="," if upload.name.lower().endswith(".csv") else None), dtype=float).reshape(-1)
                    except Exception:
                        st.error("File phải có một cột số gia tốc nền tính bằng m/s².")
        foundation_embed = st.number_input("Chiều sâu chôn cổ cột dưới cốt nền (m)", min_value=0.3, max_value=2.5, value=0.8, step=0.1, key="foundation_embed")
        footing_width = st.number_input("Bề rộng đài/móng minh họa (m)", min_value=0.8, max_value=5.0, value=1.8, step=0.1, key="foundation_width")
        footing_depth = st.number_input("Chiều dày đài/móng minh họa (m)", min_value=0.3, max_value=2.0, value=0.7, step=0.1, key="foundation_depth")
        qcvn_signature = (qcvn_record["province"], qcvn_record["admin"], float(factor)) if qcvn_record is not None else None
        model_signature = (int(floors), shape, float(area), int(bays_x), int(bays_y), float(bay_x), float(bay_y), float(story_height), col_type, float(col_b), float(col_h), int(col_bars), col_bar_d, col_tie_d, col_tie_spacing, beam_type, float(beam_b), float(beam_h), int(beam_bars_top), int(beam_bars_bottom), beam_bar_d, beam_tie_d, beam_tie_spacing, rebar_cover, float(E_gpa), concrete_grade, steel_grade, float(damping), float(dt), float(floor_mass), foundation_embed, footing_width, footing_depth, analysis_method, excitation_mode, spectrum_source, spectrum_model_parameters, float(sa_t1) if sa_t1 is not None else None, float(lambda_factor), spectrum_signature, spectrum_unit, combination, hash(np.asarray(ag).tobytes()) if ag is not None else None, qcvn_signature)
        button_label = "Tính động đất · " + analysis_method.split("|")[-1].strip()
        if st.button(button_label, type="primary", key="mdof_run"):
            try:
                if analysis_method.startswith("Tĩnh lực ngang") and sa_t1 is None and spectrum_periods is None:
                    raise ValueError("Nhập Sa(T₁) hoặc chọn phổ mẫu TCVN.")
                if analysis_method.startswith("Tĩnh lực ngang") and sa_t1 is not None and sa_t1 <= 0:
                    raise ValueError("Nhập Sa(T₁) dương lấy từ phổ thiết kế áp dụng.")
                if analysis_method.startswith("Phổ phản ứng") and (spectrum_periods is None or spectrum_sa is None):
                    raise ValueError("Nạp phổ phản ứng hai cột T, Sa trước khi tính.")
                if analysis_method.startswith("Lịch sử thời gian") and (ag is None or len(ag) < 2):
                    raise ValueError("Cần tối thiểu hai mẫu gia tốc nền hoặc chọn tạo tín hiệu minh họa QCVN.")
                footprint = {"Chữ nhật": "rect", "Chữ L": "L", "Chữ T": "T"}[shape]
                stiffnesses = frame_story_stiffness(int(floors), int(bays_x), int(bays_y), bay_x, bay_y, story_height, E_gpa*1e9, col_I, beam_I, footprint)
                masses = np.full(int(floors), floor_mass)
                floor_heights = np.arange(1, int(floors) + 1, dtype=float) * story_height
                if analysis_method.startswith("Tĩnh lực ngang"):
                    if sa_t1 is None:
                        t1 = float(modal_properties(masses, stiffnesses)["periods"][0])
                        if t1 < spectrum_periods[0] or t1 > spectrum_periods[-1]:
                            raise ValueError(f"T₁={t1:.3f}s nằm ngoài dải phổ mẫu đã tạo.")
                        sa_t1 = float(np.interp(t1, spectrum_periods, spectrum_sa))
                    result = equivalent_lateral_force(masses, stiffnesses, floor_heights, sa_t1, lambda_factor)
                    result["analysis_type"] = "elf"
                    result["floor_displacement"] = result["displacement"].copy()
                    result["method_name"] = "Equivalent Lateral Force (ELF)"
                    result["spectrum_source"] = spectrum_source
                    result["qcvn02_lookup"] = qcvn_record
                    result["spectrum_parameters"] = spectrum_model_parameters
                elif analysis_method.startswith("Phổ phản ứng"):
                    result = modal_response_spectrum(masses, stiffnesses, floor_heights, spectrum_periods, spectrum_sa, damping, combination)
                    result["analysis_type"] = "mrsa"
                    result["method_name"] = "Modal Response Spectrum Analysis (MRSA)"
                    result["spectrum_source"] = spectrum_source
                    result["qcvn02_lookup"] = qcvn_record
                    result["spectrum_parameters"] = spectrum_model_parameters
                    result["mode_table"] = pd.DataFrame({
                        "Mode": np.arange(1, len(result["periods"]) + 1),
                        "T (s)": result["periods"],
                        "Sa(T) (m/s²)": result["mode_sa"],
                        "Sd(T) (m)": result["mode_sd"],
                        "Khối lượng hữu hiệu (%)": result["effective_mass_ratio"] * 100,
                    })
                    result["effective_mass_total"] = float(np.sum(result["effective_mass_ratio"]))
                    result["floor_displacement"] = result["floor_displacement"].copy()
                else:
                    if ag is None:
                        raise ValueError("Không đọc được bản ghi gia tốc nền.")
                # TCVN design yield strain Rs/Es; section-strain capacity is
                # mapped to story drift with a transparent reduced-order hinge
                # relation (not a substitute for a member fiber model).
                    yield_drift_ratio = max(0.0002, rebar_yield_strain * 0.7 * story_height / col_depth)
                    failure_drift_ratio = confined_eps_cu * 0.7 * story_height / col_depth
                    yield_drifts = np.full(int(floors), yield_drift_ratio * story_height)
                    failure_drifts = np.full(int(floors), max(failure_drift_ratio, yield_drift_ratio*1.05) * story_height)
                    if shape == "Chữ L": ncols=max(4,(bays_x+1)*(bays_y+1)-1)
                    elif shape == "Chữ T": ncols=max(4,2*(bays_x+1)+max(0,bays_y-1)*2)
                    else: ncols=(bays_x+1)*(bays_y+1)
                    Vcol = ncols * 2.0 * col_props["yield_moment"] / story_height
                    nbeams = bays_x*(bays_y+1) + bays_y*(bays_x+1)
                    Vbeam = nbeams * 2.0 * beam_props["yield_moment"] / story_height
                    story_yield_force = np.full(int(floors), max(1.0,min(Vcol,Vbeam)))
                    result = inelastic_mdof_response(masses, stiffnesses, damping, ag, dt, yield_drifts, 0.02, failure_drifts, story_yield_force)
                    result["analysis_type"] = "time_history"
                    result["method_name"] = "Nonlinear Time-History Analysis (THA) · reduced-order model"
                    result["yield_drift_ratio"] = yield_drift_ratio
                    result["failure_drift_ratio"] = failure_drift_ratio
                    result["column_story_yield_force"] = Vcol
                    result["beam_story_yield_force"] = Vbeam
                    result["column_yield_moment"] = col_props["yield_moment"]
                    result["beam_yield_moment"] = beam_props["yield_moment"]
                    result["reinforcement_summary"] = f"Cột {int(col_bars)}Ø{int(col_bar_d*1000)}; dầm {int(beam_bars_top)}Ø{int(beam_bar_d*1000)} trên + {int(beam_bars_bottom)}Ø{int(beam_bar_d*1000)} dưới"
                    result["n_base_columns"] = ncols
                    rsw = {"CB300-V":210.0,"CB400-V":280.0,"CB500-V":300.0}[steel_grade] * 1e6
                    av_tie = 2*np.pi*col_tie_d**2/4
                    result["base_tie_shear_capacity"] = ncols*av_tie*rsw*(0.8*min(col_b,col_h))/col_tie_spacing
                    result["material_summary"] = f"{concrete_grade} · {steel_grade}"
                    result["qcvn02_lookup"] = qcvn_record
                    result["excitation_is_demo"] = qcvn_record is not None
                result["story_stiffnesses"] = stiffnesses
                st.session_state["mdof_result"] = result
                st.session_state["mdof_mass"] = masses
                st.session_state["mdof_stiffness"] = stiffnesses
                st.session_state["mdof_geometry"] = (int(floors), int(bays_x), int(bays_y), bay_x, bay_y, story_height, shape_code)
                st.session_state["mdof_signature"] = model_signature
            except (ValueError, np.linalg.LinAlgError) as exc:
                st.error(f"Không chạy được mô phỏng: {exc}")
        target_drift_pct = st.number_input("Mục tiêu lệch tầng cực đại (%)", min_value=0.2, max_value=3.0, value=1.5, step=0.1, key="mdof_target_drift")
        target_ductility = st.number_input("Giới hạn độ dẻo mục tiêu μ", min_value=1.0, max_value=12.0, value=4.0, step=0.5, key="mdof_target_mu")
        opt_method = st.selectbox(
            "Phương pháp tối ưu",
            ["Quét toàn miền + đường Pareto", "Thuật toán di truyền (GA)"],
            key="mdof_opt_method",
            help="Cả hai phương pháp tìm kích thước tuyến tính đồng dạng; quét lưới cho đường đáp ứng đầy đủ, GA minh họa tìm kiếm ngẫu nhiên có chọn lọc.",
        )
        st.caption("Hàm mục tiêu: giảm khối lượng tương đối (tỷ lệ với s²); ràng buộc: lệch tầng, độ dẻo và không chạm ngưỡng phá hoại. Hệ số s thay đổi đồng dạng tiết diện dầm/cột.")
        opt_disabled = (not analysis_method.startswith("Lịch sử thời gian") or ag is None)
        if analysis_method.startswith("Lịch sử thời gian") and ag is None:
            st.info("Để chạy tối ưu, hãy tạo tín hiệu gia tốc minh họa hoặc nạp bản ghi gia tốc THA trước.")
        elif not analysis_method.startswith("Lịch sử thời gian"):
            st.info("Tối ưu hiện chạy trên đáp ứng phi tuyến THA. Hãy chọn ‘Lịch sử thời gian · Time-History Analysis (THA)’ và cung cấp gia tốc nền.")
        if st.button("Tối ưu sơ bộ tiết diện theo động đất", key="mdof_optimize", disabled=opt_disabled):
            try:
                if ag is None:
                    raise ValueError("Không đọc được bản ghi gia tốc nền.")
                footprint = {"Chữ nhật": "rect", "Chữ L": "L", "Chữ T": "T"}[shape]
                masses = np.full(int(floors), floor_mass)
                if shape == "Chữ L": ncols=max(4,(bays_x+1)*(bays_y+1)-1)
                elif shape == "Chữ T": ncols=max(4,2*(bays_x+1)+max(0,bays_y-1)*2)
                else: ncols=(bays_x+1)*(bays_y+1)
                Vcol=ncols*2.0*col_props["yield_moment"]/story_height
                nbeams=bays_x*(bays_y+1)+bays_y*(bays_x+1)
                Vbeam=nbeams*2.0*beam_props["yield_moment"]/story_height
                rsw={"CB300-V":210.0,"CB400-V":280.0,"CB500-V":300.0}[steel_grade]*1e6
                base_cap=ncols*(2*np.pi*col_tie_d**2/4)*rsw*(0.8*min(col_b,col_h))/col_tie_spacing
                min_scale = max(0.25 / col_b, 0.25 / col_depth,
                                0.20 / beam_b, 0.30 / beam_h, 0.50)
                max_scale = 2.50
                if min_scale >= max_scale:
                    raise ValueError("Kích thước đầu vào không nằm trong miền tối ưu sơ bộ khả dụng.")

                evaluated = {}
                def evaluate_scale(scale):
                    scale = float(np.clip(scale, min_scale, max_scale))
                    cache_key = round(scale, 7)
                    if cache_key in evaluated:
                        return evaluated[cache_key]
                    ks = frame_story_stiffness(
                        int(floors), int(bays_x), int(bays_y), bay_x, bay_y,
                        story_height, E_gpa*1e9, col_I*scale**4,
                        beam_I*scale**4, footprint,
                    )
                    yd_ratio = max(0.0002, rebar_yield_strain*0.7*story_height/(col_depth*scale))
                    fd_ratio = confined_eps_cu*0.7*story_height/(col_depth*scale)
                    # Geometrically similar RC members with constant steel ratio:
                    # flexural/shear capacity scales approximately with s^3.
                    story_fy = min(Vcol, Vbeam) * scale**3
                    try:
                        response = inelastic_mdof_response(
                            masses, ks, damping, ag, dt,
                            np.full(int(floors), yd_ratio*story_height), 0.02,
                            np.full(int(floors), max(fd_ratio, yd_ratio*1.05)*story_height),
                            np.full(int(floors), story_fy),
                        )
                    except (ValueError, np.linalg.LinAlgError) as exc:
                        # One nonconvergent trial must not abort the entire
                        # parameter search; mark it infeasible and continue.
                        item={"scale":scale,"mass_index":scale**2,
                              "drift_pct":float(target_drift_pct*5),
                              "mu":float(target_ductility*5),"failure_time":0.0,
                              "feasible":False,"stiffness":ks,"response":None,
                              "yield_drift_ratio":yd_ratio,"failure_drift_ratio":fd_ratio,
                              "failure_reason":str(exc)}
                        evaluated[cache_key]=item
                        return item
                    peak_drift = float(np.max(np.abs(response["story_drift"])) / story_height * 100.0)
                    peak_mu = float(np.max(np.abs(response["story_drift"]) /
                                            response["yield_story_displacement"][None, :]))
                    feasible = (response["failure_time"] is None and
                                peak_drift <= target_drift_pct and peak_mu <= target_ductility)
                    item = {"scale": scale, "mass_index": scale**2,
                            "drift_pct": peak_drift, "mu": peak_mu,
                            "failure_time": response["failure_time"],
                            "feasible": feasible, "stiffness": ks,
                            "response": response, "yield_drift_ratio": yd_ratio,
                            "failure_drift_ratio": fd_ratio, "failure_reason": ""}
                    evaluated[cache_key] = item
                    return item

                with st.spinner("Đang phân tích các phương án theo THA phi tuyến giảm bậc…"):
                    if opt_method.startswith("Quét"):
                        grid = np.unique(np.r_[np.linspace(min_scale, max_scale, 43), 1.0])
                        candidates = [evaluate_scale(s) for s in grid]
                        convergence = []
                    else:
                        rng = np.random.default_rng(42)
                        population_size, generations = 12, 8
                        population = np.r_[np.linspace(min_scale, max_scale, 8), 1.0,
                                           rng.uniform(min_scale, max_scale, population_size-9)]
                        convergence = []
                        for generation in range(generations):
                            ranked = sorted((evaluate_scale(s) for s in population),
                                            key=lambda r: (not r["feasible"], r["mass_index"] if r["feasible"] else
                                                           1e4 + 1e3*max(r["drift_pct"]/target_drift_pct-1, 0) +
                                                           1e3*max(r["mu"]/target_ductility-1, 0) +
                                                           (1e4 if r["failure_time"] is not None else 0)))
                            feasible_so_far = [r for r in evaluated.values() if r["feasible"]]
                            convergence.append(min((r["scale"] for r in feasible_so_far), default=np.nan))
                            elites = np.array([r["scale"] for r in ranked[:4]])
                            children = list(elites)
                            mutation = (max_scale-min_scale)*0.12*(1-generation/generations)
                            while len(children) < population_size:
                                parents = rng.choice(elites, size=2, replace=True)
                                child = float(np.mean(parents) + rng.normal(0, mutation))
                                children.append(float(np.clip(child, min_scale, max_scale)))
                            population = np.asarray(children)
                        candidates = list(evaluated.values())

                feasible_candidates = [item for item in candidates if item["feasible"]]
                if opt_method.startswith("GA") and not feasible_candidates:
                    feasible_candidates = [item for item in evaluated.values() if item["feasible"]]
                case_summaries = [{key: item[key] for key in
                                   ("scale", "mass_index", "drift_pct", "mu", "failure_time", "feasible", "failure_reason")}
                                  for item in evaluated.values()]
                study_signature = (model_signature, opt_method, float(target_drift_pct),
                                   float(target_ductility), float(dt), hash(np.asarray(ag).tobytes()))
                if not feasible_candidates:
                    st.session_state["mdof_opt_study"] = {
                        "signature": study_signature, "method": opt_method,
                        "cases": case_summaries, "best": None,
                        "convergence": convergence, "bounds": (min_scale, max_scale),
                    }
                    st.warning(f"Không tìm thấy phương án thỏa đồng thời các ràng buộc trong s={min_scale:.2f}–{max_scale:.2f}. Xem đường đáp ứng và mức vi phạm để điều chỉnh thiết kế hoặc mục tiêu.")
                else:
                    best = min(feasible_candidates, key=lambda item: item["mass_index"])
                    scale, ks, rr = best["scale"], best["stiffness"], best["response"]
                    rr["yield_drift_ratio"] = best["yield_drift_ratio"]
                    rr["failure_drift_ratio"] = best["failure_drift_ratio"]
                    rr["material_summary"] = f"{concrete_grade} · {steel_grade} · sơ bộ đồng dạng ×{scale:.2f}"
                    rr["reinforcement_summary"] = f"Giả thiết đồng dạng, hàm lượng thép giữ nguyên · ×{scale:.2f}"
                    rr["column_story_yield_force"] = Vcol*scale**3
                    rr["beam_story_yield_force"] = Vbeam*scale**3
                    rr["n_base_columns"] = ncols
                    rr["base_tie_shear_capacity"] = base_cap*scale
                    rr["story_stiffnesses"] = ks
                    st.session_state["mdof_result"] = rr
                    st.session_state["mdof_stiffness"] = ks
                    st.session_state["mdof_opt_scale"] = scale
                    st.session_state["mdof_signature"] = model_signature
                    st.session_state["mdof_opt_study"] = {
                        "signature": study_signature, "method": opt_method,
                        "cases": case_summaries, "best": best,
                        "convergence": convergence, "bounds": (min_scale, max_scale),
                    }
                    st.success(f"Phương án nhẹ nhất trong miền khảo sát: s={scale:.3f}; khối lượng tương đối ≈ {best['mass_index']*100:.1f}% so với đầu vào; drift={best['drift_pct']:.3f}%, μmax={best['mu']:.2f}.")
            except (ValueError, np.linalg.LinAlgError, RuntimeError, FloatingPointError) as exc:
                st.error(f"Không tối ưu được: {exc}")

        study_signature = (model_signature, opt_method, float(target_drift_pct),
                           float(target_ductility), float(dt), hash(np.asarray(ag).tobytes()) if ag is not None else None)
        study = st.session_state.get("mdof_opt_study")
        if study and study.get("signature") == study_signature:
            cases = study["cases"]
            if cases:
                sorted_cases = sorted(cases, key=lambda item: item["mass_index"])
                opt_fig = make_subplots(rows=1, cols=2,
                                        subplot_titles=("Pareto: khối lượng – lệch tầng",
                                                        "Độ dẻo theo kích thước tiết diện"))
                colors = ["#32805a" if q["feasible"] else "#aab2b9" for q in sorted_cases]
                opt_fig.add_trace(go.Scatter(
                    x=[q["mass_index"]*100 for q in sorted_cases],
                    y=[q["drift_pct"] for q in sorted_cases], mode="markers",
                    marker=dict(color=colors, size=8), name="Các phương án đã phân tích",
                    text=[f"s={q['scale']:.3f}; μ={q['mu']:.2f}" for q in sorted_cases],
                    hovertemplate="Khối lượng tương đối=%{x:.1f}%<br>Drift=%{y:.3f}%<br>%{text}<extra></extra>"),
                    row=1, col=1)
                front = []
                feasible_points = [q for q in sorted_cases if q["feasible"]]
                for item in feasible_points:
                    dominated = any(other["mass_index"] <= item["mass_index"] and
                                    other["drift_pct"] <= item["drift_pct"] and
                                    (other["mass_index"] < item["mass_index"] or
                                     other["drift_pct"] < item["drift_pct"])
                                    for other in feasible_points)
                    if not dominated:
                        front.append(item)
                if front:
                    front.sort(key=lambda item: item["mass_index"])
                    opt_fig.add_trace(go.Scatter(
                        x=[q["mass_index"]*100 for q in front],
                        y=[q["drift_pct"] for q in front], mode="lines+markers",
                        line=dict(color="#d77835", width=2), marker=dict(size=9),
                        name="Pareto khả thi"), row=1, col=1)
                opt_fig.add_hline(y=float(target_drift_pct), line_dash="dot",
                                  line_color="#ba3434", annotation_text="Giới hạn drift",
                                  row=1, col=1)
                opt_fig.add_trace(go.Scatter(
                    x=[q["scale"] for q in sorted_cases], y=[q["mu"] for q in sorted_cases],
                    mode="lines+markers", line=dict(color="#286b8e"),
                    marker=dict(color=colors, size=7), name="μ demand"), row=1, col=2)
                opt_fig.add_hline(y=float(target_ductility), line_dash="dot",
                                  line_color="#ba3434", annotation_text="Giới hạn μ",
                                  row=1, col=2)
                opt_fig.add_vline(x=1.0, line_dash="dash", line_color="#7a6b9b",
                                  annotation_text="Tiết diện hiện tại", row=1, col=2)
                if study["best"] is not None:
                    opt_fig.add_trace(go.Scatter(
                        x=[study["best"]["mass_index"]*100],
                        y=[study["best"]["drift_pct"]], mode="markers",
                        marker=dict(color="#157347", size=13, symbol="star"),
                        name="Phương án nhẹ nhất đạt yêu cầu"), row=1, col=1)
                opt_fig.update_layout(title="So sánh phương án thiết kế theo THA",
                                      height=390, margin=dict(l=8,r=8,t=55,b=15),
                                      legend=dict(orientation="h", y=-0.2), template="plotly_white")
                opt_fig.update_xaxes(title_text="Khối lượng tương đối (%)", row=1, col=1)
                opt_fig.update_yaxes(title_text="Drift cực đại (%)", row=1, col=1)
                opt_fig.update_xaxes(title_text="Hệ số đồng dạng tiết diện s", row=1, col=2)
                opt_fig.update_yaxes(title_text="Độ dẻo demand μ", row=1, col=2)
                st.plotly_chart(opt_fig, use_container_width=True,
                                key="mdof_opt_tradeoff")
                table = pd.DataFrame([{
                    "s": item["scale"], "Khối lượng tương đối (%)": item["mass_index"]*100,
                    "Drift max (%)": item["drift_pct"], "μ max": item["mu"],
                    "Phá hoại (s)": item["failure_time"] if item["failure_time"] is not None else "Không",
                    "Đạt ràng buộc": "Có" if item["feasible"] else "Không",
                } for item in sorted_cases])
                st.dataframe(table.round(4), hide_index=True, use_container_width=True)
                if study["convergence"]:
                    conv = go.Figure(go.Scatter(y=study["convergence"], mode="lines+markers",
                                                line=dict(color="#32805a", width=2),
                                                name="s tốt nhất khả thi đến thế hệ này"))
                    conv.update_layout(title="Hội tụ GA · hệ số s tốt nhất khả thi",
                                       xaxis_title="Thế hệ", yaxis_title="s", height=230,
                                       margin=dict(l=8,r=8,t=45,b=15), template="plotly_white")
                    st.plotly_chart(conv, use_container_width=True, key="mdof_opt_ga_convergence")
            if study["best"] is None:
                st.info("Không có phương án đạt cả drift, độ dẻo và điều kiện chưa phá hoại trong miền khảo sát.")
    with b:
        g_floors, g_bx, g_by = int(floors), int(bays_x), int(bays_y)
        gx, gy = g_floors*story_height, max(g_bx*bay_x, g_by*bay_y)
        outline_x = [0, g_bx*bay_x, g_bx*bay_x, 0, 0]
        outline_y = [0, 0, g_by*bay_y, g_by*bay_y, 0]
        if shape == "Chữ L":
            outline_x, outline_y = [0, g_bx*bay_x, g_bx*bay_x, bay_x, bay_x, 0, 0], [0, 0, bay_y, bay_y, g_by*bay_y, g_by*bay_y, 0]
        elif shape == "Chữ T":
            outline_x, outline_y = [0, g_bx*bay_x, g_bx*bay_x, (g_bx+1)*bay_x/2, (g_bx+1)*bay_x/2, (g_bx-1)*bay_x/2, (g_bx-1)*bay_x/2, 0, 0], [0, 0, bay_y, bay_y, g_by*bay_y, g_by*bay_y, bay_y, bay_y, 0]
        fig3d = go.Figure()
        for k in range(g_floors+1):
            z=k*story_height
            fig3d.add_trace(go.Scatter3d(x=outline_x,y=outline_y,z=[z]*len(outline_x),mode="lines",line=dict(color="#5b8def" if k else "#333",width=5),showlegend=False,hoverinfo="skip"))
            for i in range(g_bx):
                for j in range(g_by):
                    if shape == "Chữ L" and i == g_bx-1 and j == g_by-1:
                        continue
                    if shape == "Chữ T" and not (j == 0 or (i == (g_bx-1)//2 and j > 0)):
                        continue
                    fig3d.add_trace(go.Mesh3d(x=[i*bay_x,(i+1)*bay_x,(i+1)*bay_x,i*bay_x],y=[j*bay_y,j*bay_y,(j+1)*bay_y,(j+1)*bay_y],z=[z]*4,i=[0,0],j=[1,2],k=[2,3],color="#9ec5fe",opacity=0.22,showscale=False,hovertemplate=f"Sàn tầng {k}<extra></extra>"))
        for i in range(g_bx+1):
            for j in range(g_by+1):
                xx, yy=i*bay_x,j*bay_y
                if shape == "Chữ L" and i==g_bx and j==g_by: continue
                fig3d.add_trace(go.Scatter3d(x=[xx]*2,y=[yy]*2,z=[-foundation_embed,g_floors*story_height],mode="lines",line=dict(color="#e67300",width=7),showlegend=False,hovertemplate=f"Cột ngàm chân · X={xx:.1f} m · Y={yy:.1f} m<extra></extra>"))
                hx=footing_width/2; hy=footing_width/2; z0=-foundation_embed-footing_depth
                vx=[xx-hx,xx+hx,xx+hx,xx-hx,xx-hx,xx+hx,xx+hx,xx-hx]
                vy=[yy-hy,yy-hy,yy+hy,yy+hy,yy-hy,yy-hy,yy+hy,yy+hy]
                vz=[z0,z0,z0,z0,-foundation_embed,-foundation_embed,-foundation_embed,-foundation_embed]
                fig3d.add_trace(go.Mesh3d(x=vx,y=vy,z=vz,i=[0,0,4,4,0,0,1,1,2,2,3,3],j=[1,2,5,6,1,5,2,6,3,7,0,4],k=[2,3,6,7,5,4,6,5,7,6,4,7],color="#8b7355",opacity=.35,showscale=False,showlegend=False,hovertemplate="Đài/móng minh họa<extra></extra>"))
        input_view = st.segmented_control("Nội dung trực quan", ["Mô hình 3D & móng", "Mặt cắt & bố trí cốt thép"],
                                          default="Mô hình 3D & móng", key="mdof_input_view", width="stretch")
        if input_view == "Mô hình 3D & móng":
            fig3d.update_layout(height=530,margin=dict(l=0,r=0,t=25,b=0),scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Z (m)",aspectmode="data"),title="Mô hình 3D · sàn, khung, cổ cột và móng")
            st.plotly_chart(fig3d,use_container_width=True)
        elif input_view == "Mặt cắt & bố trí cốt thép":
            def section_plot(name,width,depth,is_round,top_count,bottom_count,bar_d,tie_d,tie_spacing,cover):
                f=go.Figure()
                nbar=max(1,int(top_count+bottom_count))
                if is_round:
                    f.add_shape(type="circle",x0=-width/2,y0=-depth/2,x1=width/2,y1=depth/2,
                                line=dict(color="#244a35",width=3),fillcolor="#d8e6d5",layer="below")
                    tie_r=max(0.0,min(width,depth)/2-cover-tie_d/2)
                    f.add_shape(type="circle",x0=-tie_r,y0=-tie_r,x1=tie_r,y1=tie_r,
                                line=dict(color="#e67300",width=2),fillcolor="rgba(0,0,0,0)")
                    nbar=max(4,nbar)
                    bar_r=max(0.0,min(width,depth)/2-cover-tie_d-bar_d/2)
                    ang=np.linspace(0,2*np.pi,nbar,endpoint=False)
                    bar_x=bar_r*np.cos(ang); bar_y=bar_r*np.sin(ang)
                elif name.startswith("Mặt cắt cột"):
                    f.add_shape(type="rect",x0=-width/2,y0=-depth/2,x1=width/2,y1=depth/2,
                                line=dict(color="#244a35",width=3),fillcolor="#d8e6d5",layer="below")
                    tie_x=max(0.0,width/2-cover-tie_d/2); tie_y=max(0.0,depth/2-cover-tie_d/2)
                    f.add_shape(type="rect",x0=-tie_x,y0=-tie_y,x1=tie_x,y1=tie_y,
                                line=dict(color="#e67300",width=2),fillcolor="rgba(0,0,0,0)")
                    bar_xoff=max(0.0,width/2-cover-tie_d-bar_d/2)
                    bar_yoff=max(0.0,depth/2-cover-tie_d-bar_d/2)
                    inner_w,inner_h=2*bar_xoff,2*bar_yoff
                    perim=2*(inner_w+inner_h)
                    bx,by=[],[]
                    distances=np.arange(nbar)*perim/nbar if perim>0 else np.zeros(nbar)
                    for dist in distances:
                        if dist < inner_w:
                            bx.append(-bar_xoff+dist); by.append(-bar_yoff)
                        elif dist < inner_w+inner_h:
                            bx.append(bar_xoff); by.append(-bar_yoff+dist-inner_w)
                        elif dist < 2*inner_w+inner_h:
                            bx.append(bar_xoff-(dist-inner_w-inner_h)); by.append(bar_yoff)
                        else:
                            bx.append(-bar_xoff); by.append(bar_yoff-(dist-2*inner_w-inner_h))
                    bar_x,bar_y=np.asarray(bx),np.asarray(by)
                else:
                    f.add_shape(type="rect",x0=-width/2,y0=-depth/2,x1=width/2,y1=depth/2,
                                line=dict(color="#244a35",width=3),fillcolor="#d8e6d5",layer="below")
                    tie_x=max(0.0,width/2-cover-tie_d/2); tie_y=max(0.0,depth/2-cover-tie_d/2)
                    f.add_shape(type="rect",x0=-tie_x,y0=-tie_y,x1=tie_x,y1=tie_y,
                                line=dict(color="#e67300",width=2),fillcolor="rgba(0,0,0,0)")
                    bar_xoff=max(0.0,width/2-cover-tie_d-bar_d/2)
                    bar_yoff=max(0.0,depth/2-cover-tie_d-bar_d/2)
                    top_x=np.linspace(-bar_xoff,bar_xoff,max(1,int(top_count)))
                    bottom_x=np.linspace(-bar_xoff,bar_xoff,max(1,int(bottom_count)))
                    bar_x=np.r_[top_x,bottom_x]
                    bar_y=np.r_[np.full(len(top_x),bar_yoff),np.full(len(bottom_x),-bar_yoff)]
                f.add_trace(go.Scatter(x=bar_x,y=bar_y,mode="markers",
                    marker=dict(size=13,color="#bf3028",symbol="circle",line=dict(color="white",width=1.5)),
                    name="Thép dọc",hovertemplate=f"Cốt dọc Ø{bar_d*1000:.0f} mm<extra></extra>"))
                f.update_layout(title=name,height=360,margin=dict(l=15,r=15,t=50,b=15),
                    xaxis=dict(title="bề rộng (m)",scaleanchor="y",range=[-width*.65,width*.65]),
                    yaxis=dict(title="chiều cao (m)",range=[-depth*.65,depth*.65]),showlegend=False)
                f.add_annotation(x=0,y=-depth*.59,
                    text=f"{nbar} thanh dọc · đai Ø{tie_d*1000:.0f} @ {tie_spacing*1000:.0f} mm · cover={cover*1000:.0f} mm",
                    showarrow=False,font=dict(size=11))
                return f
            csec,bsec=st.columns(2)
            with csec: st.plotly_chart(section_plot("Mặt cắt cột · bố trí thép dọc và đai",col_b,col_depth,col_type=="Ống tròn",int(col_bars)//2,int(col_bars)//2,col_bar_d,col_tie_d,col_tie_spacing,rebar_cover),use_container_width=True)
            with bsec: st.plotly_chart(section_plot("Mặt cắt dầm · thép trên/dưới và đai",beam_b,beam_h,False,int(beam_bars_top),int(beam_bars_bottom),beam_bar_d,beam_tie_d,beam_tie_spacing,rebar_cover),use_container_width=True)
            col_clearance=min(col_b,col_depth)/2-rebar_cover-col_tie_d-col_bar_d/2
            beam_clearance=min(beam_b,beam_h)/2-rebar_cover-beam_tie_d-beam_bar_d/2
            if col_clearance<=0 or beam_clearance<=0:
                st.warning("Cấu hình tiết diện, lớp bảo vệ, đai và đường kính thép dọc không còn đủ khoảng đặt thép thực tế. Hãy tăng kích thước tiết diện hoặc điều chỉnh bố trí; hình chấm khi đó chỉ báo vị trí giới hạn.")
            st.caption(f"As cột ≈ {col_bars*np.pi*col_bar_d**2/4*1e6:.0f} mm² · As dầm trên/dưới ≈ {beam_props['steel_area_top']*1e6:.0f}/{beam_props['steel_area_bottom']*1e6:.0f} mm² · Ieff cột/dầm = {col_I:.4g}/{beam_I:.4g} m⁴")
        if input_view == "Mô hình 3D & móng":
            basefig=go.Figure()
            basefig.add_trace(go.Scatter3d(x=[-2,2,2,-2,-2],y=[-2,-2,2,2,-2],z=[0]*5,mode="lines",line=dict(color="#508b55",width=5),name="Cốt nền"))
            bx0,by0=footing_width/2,footing_width/2; zbot=-foundation_embed-footing_depth
            verts_x=[-bx0,bx0,bx0,-bx0,-bx0,bx0,bx0,-bx0]; verts_y=[-by0,-by0,by0,by0,-by0,-by0,by0,by0]; verts_z=[zbot]*4+[-foundation_embed]*4
            basefig.add_trace(go.Mesh3d(x=verts_x,y=verts_y,z=verts_z,i=[0,0,4,4,0,0,1,1,2,2,3,3],j=[1,2,5,6,1,5,2,6,3,7,0,4],k=[2,3,6,7,5,4,6,5,7,6,4,7],color="#8b7355",opacity=.5,name="Đài/móng"))
            basefig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[zbot,g_floors*story_height],mode="lines",line=dict(color="#e67300",width=12),name="Cổ cột ngàm liên tục"))
            basefig.update_layout(height=450,title="Chi tiết chân cột xuyên cốt nền và ngàm vào đài",scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Z (m)",aspectmode="data",zaxis=dict(range=[zbot-0.2,2])),margin=dict(l=0,r=0,t=40,b=0))
            st.plotly_chart(basefig,use_container_width=True)
            st.info("Trong mô hình FEM, chân cột được ngàm bằng khóa 6 bậc tự do tại nút chân cột. Hình móng là minh họa; mô hình hiện chưa tính độ mềm đất, trượt/lật móng, neo cốt thép chờ hay phá hoại mặt tiếp xúc.")
        result = st.session_state.get("mdof_result")
        if result is None or st.session_state.get("mdof_signature") != model_signature:
            st.info("Input đang thay đổi so với kết quả gần nhất. Chạy lại phân tích để đồng bộ hoạt ảnh, độ cứng FEM và biểu đồ.")
            return
        def render_preliminary_strength_check(story_shear_demand):
            """Elastic stress screening based on reduced story shear, not member FEM forces."""
            Vstory=np.abs(np.asarray(story_shear_demand,dtype=float).reshape(-1)[:int(floors)])
            ncol=max(1,(int(bays_x)+1)*(int(bays_y)+1))
            Ncol=np.cumsum(np.full(int(floors),floor_mass*9.80665/ncol)[::-1])[::-1]
            Mcol=Vstory*story_height/(2*ncol)
            ccol=col_depth/2
            sig_c=Ncol/max(col_A,1e-12)+Mcol*ccol/max(col_props["effective_I"],1e-12)
            nmod=steel_E/concrete_E
            sig_s=nmod*Mcol*ccol/max(col_props["effective_I"],1e-12)
            fck_map={"B20":20.0,"B25":25.0,"B30":30.0,"B35":35.0,"B40":40.0,"B45":45.0,"B50":50.0,"B60":60.0}
            fcd=0.85*fck_map[concrete_grade]/1.5
            eta_c=sig_c/1e6/fcd; eta_s=sig_s/1e6/rebar_yield_design
            fl=np.arange(1,int(floors)+1)
            st.markdown("### Kiểm tra ứng suất sơ bộ · demand/capacity")
            st.caption("Ước tính từ lực cắt tầng của mô hình giảm bậc: chia đều lực cho các cột ở lưới chân, lấy M≈V·h/(2·n cột), lực dọc từ tải sàn quy đổi phía trên. Đây không phải nội lực phần tử FEM như ETABS/STAAD.Pro.")
            table=pd.DataFrame({"Tầng":fl,"V tầng (kN)":Vstory/1000,"σ nén bê tông (MPa)":sig_c/1e6,"fcd sàng lọc (MPa)":fcd,"η bê tông":eta_c,"σ thép quy đổi (MPa)":sig_s/1e6,"fy (MPa)":rebar_yield_design,"η thép":eta_s})
            st.dataframe(table.round(3),hide_index=True,use_container_width=True)
            dc=go.Figure()
            dc.add_trace(go.Bar(y=fl,x=eta_c,orientation="h",name="Bê tông · σc/fcd",marker_color="#d77835"))
            dc.add_trace(go.Bar(y=fl,x=eta_s,orientation="h",name="Thép · σs/fy",marker_color="#3468a8"))
            dc.add_vline(x=1.0,line_dash="dash",line_color="#b83333",annotation_text="η = 1.0")
            dc.update_layout(title="Tỷ số ứng suất / sức kháng theo tầng",barmode="group",xaxis_title="Demand / capacity (η)",yaxis_title="Tầng",height=max(350,int(floors)*32),margin=dict(l=20,r=20,t=50,b=35),paper_bgcolor="white",plot_bgcolor="white")
            st.plotly_chart(dc,use_container_width=True)
            sm=go.Figure(go.Heatmap(x=["Ứng suất nén cột"],y=fl,z=(sig_c/1e6).reshape(-1,1),colorscale=[[0,"#d9ead3"],[0.65,"#f6b26b"],[1,"#cc3333"]],colorbar=dict(title="MPa"),hovertemplate="Tầng %{y}<br>σc ≈ %{z:.2f} MPa<extra></extra>"))
            sm.update_layout(title="Bản đồ ứng suất nén ước tính theo chiều cao",xaxis_title="Vị trí kiểm tra quy đổi",yaxis_title="Tầng",height=max(350,int(floors)*32),margin=dict(l=20,r=20,t=50,b=35),paper_bgcolor="white")
            st.plotly_chart(sm,use_container_width=True)
            st.markdown("#### Mô hình khung tô màu theo tỷ số sử dụng")
            st.caption("Màu được gán theo η lớn hơn giữa bê tông và thép, trên các đoạn cột từng tầng. Đây là bản đồ demand/capacity ước tính của shear-building, không phải contour ứng suất từ nội lực phần tử 3D.")
            eta_story=np.maximum(eta_c,eta_s)
            bands=[(0.50,"#25834a","η < 0,50"),(0.75,"#8dbb3f","0,50–0,75"),(0.90,"#e5c642","0,75–0,90"),(1.00,"#e58532","0,90–1,00"),(np.inf,"#c83333","η > 1,00")]
            def color_band(value):
                for limit,color,label in bands:
                    if value < limit: return color,label
                return bands[-1][1],bands[-1][2]
            npx,npy=int(bays_x)+1,int(bays_y)+1
            coords=[]
            for jj in range(npy):
                for ii in range(npx):
                    if shape=="Chữ L" and ii==npx-1 and jj==npy-1: continue
                    if shape=="Chữ T" and jj>0 and ii not in {(npx-2)//2,(npx-1)//2}: continue
                    coords.append((ii*bay_x,jj*bay_y))
            stress3d=go.Figure()
            used_labels=set()
            for level in range(int(floors)):
                color,label=color_band(float(eta_story[level]))
                hover=f"Tầng {level+1}<br>η bê tông={eta_c[level]:.3f}<br>η thép={eta_s[level]:.3f}<br>σc≈{sig_c[level]/1e6:.2f} MPa<br>σs≈{sig_s[level]/1e6:.2f} MPa<extra></extra>"
                for col_index,(xx,yy) in enumerate(coords):
                    stress3d.add_trace(go.Scatter3d(x=[xx,xx],y=[yy,yy],z=[level*story_height,(level+1)*story_height],mode="lines",line=dict(color=color,width=8),name=label,legendgroup=label,showlegend=(label not in used_labels and col_index==0),hovertemplate=hover))
                used_labels.add(label)
            for xx,yy in coords:
                stress3d.add_trace(go.Scatter3d(x=[xx]*2,y=[yy]*2,z=[0,int(floors)*story_height],mode="lines",line=dict(color="#333b45",width=2),showlegend=False,hoverinfo="skip"))
            stress3d.update_layout(height=540,title="Dải màu η theo tầng · mô hình 3D",scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Cao độ (m)",aspectmode="data",camera=dict(eye=dict(x=1.7,y=-1.8,z=1.25))),legend=dict(title="Tỷ số sử dụng η"),margin=dict(l=0,r=0,t=45,b=0),paper_bgcolor="white")
            st.markdown("**Chú giải:** xanh đậm < 0,50 · xanh nhạt 0,50–0,75 · vàng 0,75–0,90 · cam 0,90–1,00 · đỏ > 1,00. η = ứng suất ước tính / sức kháng tham chiếu; ngưỡng màu là phân lớp hiển thị, không phải giới hạn tiêu chuẩn độc lập.")
            st.plotly_chart(stress3d,use_container_width=True)
            eta_max=max(float(np.max(eta_c)),float(np.max(eta_s)))
            if eta_max<=1: st.success(f"Sàng lọc đàn hồi theo giả thiết hiện tại: η lớn nhất = {eta_max:.2f} (< 1,0).")
            else: st.error(f"Sàng lọc đàn hồi theo giả thiết hiện tại: η lớn nhất = {eta_max:.2f} (> 1,0); cần xem xét lại thiết kế.")
            with st.expander("Công thức và giới hạn kiểm tra"):
                st.latex(r"N_i\approx\frac{gA}{n_c}(n-i+1),\quad M_i\approx\frac{|V_i|h}{2n_c},\quad \sigma_c=\frac{N_i}{A_c}+\frac{M_i c}{I_{eff}},\quad \sigma_s\approx\frac{E_s}{E_c}\frac{M_i c}{I_{eff}}")
                st.markdown(f"Đối chiếu sàng lọc với fcd = 0,85·fck/1,5 = {fcd:.2f} MPa và fy = {rebar_yield_design:.0f} MPa. Chưa kiểm tra tương tác N–M tiết diện, cắt/đai, ổn định, nút, tải trọng tổ hợp hay cấu tạo kháng chấn. Kết quả không được gọi là thiết kế đạt chuẩn.")
        if result.get("analysis_type") in {"elf", "mrsa"}:
            # Use the result's floor response as the source of truth, then
            # validate every companion vector before handing it to pandas.
            # This prevents a stale/legacy result (for example, one carrying
            # an extra fixed-base entry) from crashing the entire Streamlit page.
            disp = np.asarray(result["floor_displacement"], dtype=float).reshape(-1)
            n_floor = len(disp)
            floor_ids = np.arange(1, n_floor + 1)
            force = np.asarray(result["floor_forces"], dtype=float).reshape(-1)
            shear = np.asarray(result["story_shear"], dtype=float).reshape(-1)
            if result["analysis_type"] == "elf":
                # For ELF, inter-storey drift is exactly the floor displacement
                # difference with a fixed zero-displacement base.
                drift = np.diff(np.r_[0.0, disp])
            else:
                drift = np.asarray(result["story_drift"], dtype=float).reshape(-1)
            vector_lengths = {
                "cao độ tầng": n_floor,
                "lực tầng Fi": len(force),
                "lực cắt tầng Vi": len(shear),
                "lệch tầng": len(drift),
            }
            if n_floor != int(floors) or any(length != n_floor for length in vector_lengths.values()):
                details = " · ".join(f"{name}: {length}" for name, length in vector_lengths.items())
                st.error(
                    "Kết quả động đất không đồng nhất số tầng nên chưa thể lập bảng. "
                    f"Số tầng input: {int(floors)}; số tầng chuyển vị: {n_floor}; {details}. "
                    "Hãy chạy lại phân tích để tạo kết quả mới theo đúng mô hình hiện tại."
                )
                return
            m1, m2, m3 = st.columns(3)
            m1.metric("Chu kỳ cơ bản FEM · T₁", f"{result['periods'][0]:.3f} s")
            m2.metric("Lực cắt đáy", f"{result['base_shear']/1000:.2f} kN")
            if result["analysis_type"] == "mrsa":
                m3.metric("Tổng khối lượng hữu hiệu modal", f"{result['effective_mass_total']*100:.1f}%")
                st.caption(f"Phổ đã tổ hợp {result['combination']} với ξ = {result['damping_ratio']*100:.1f}%. Kết quả là cực đại đàn hồi theo phổ; không có dấu thời gian hay chuỗi đáp ứng.")
                spectrum_fig = go.Figure()
                spectrum_fig.add_trace(go.Scatter(x=result["spectrum_periods"], y=result["spectrum_accelerations"]/9.80665, mode="lines", name="Phổ Sa", line=dict(color="#315c76", width=2)))
                spectrum_fig.add_trace(go.Scatter(x=result["periods"], y=result["mode_sa"]/9.80665, mode="markers", name="Mode giữ lại", marker=dict(color="#e67300", size=9)))
                spectrum_fig.update_layout(title="Phổ gia tốc đầu vào và các mode được giữ lại", xaxis_title="Chu kỳ T (s)", yaxis_title="Sa (g)", height=320, margin=dict(l=15,r=15,t=45,b=30))
                st.plotly_chart(spectrum_fig, use_container_width=True)
                st.dataframe(result["mode_table"], hide_index=True, use_container_width=True)
            else:
                sa_label = "Sa(T₁) phổ mẫu" if str(result.get("spectrum_source", "")).startswith("Phổ mẫu") else "Sa(T₁) nhập"
                m3.metric(sa_label, f"{result['spectral_acceleration']/9.80665:.3f} g")
                st.caption(f"Lực cắt đáy tương đương Fb = Sa(T₁) · Σm · λ, λ = {result['lambda_factor']:.2f}; phân bố lực tầng theo Fi ∝ mi·zi.")
            if result.get("qcvn02_lookup"):
                qr = result["qcvn02_lookup"]
                st.caption(f"Phổ mẫu được scale theo a₍gR₎={qr['agR_g']:.3f}g tại {qr['admin']}, {qr['province']} tra từ QCVN 02:2021/BXD; dạng phổ Type 1, q=1 (đàn hồi).")
            if result.get("spectrum_parameters"):
                _, _, _, soil_type, gamma_i, spectrum_damping = result["spectrum_parameters"]
                st.caption(f"Thông số phổ mẫu: nền {soil_type}, γI={gamma_i:.2f}, ξ={spectrum_damping*100:.1f}%.")
            seismic_table = pd.DataFrame({
                "Tầng": floor_ids,
                "Cao độ (m)": floor_ids * story_height,
                "Lực tầng Fi (kN)": force / 1000,
                "Lực cắt tầng Vi (kN)": shear / 1000,
                "Chuyển vị đỉnh tầng (mm)": disp * 1000,
                "Lệch tầng (mm)": drift * 1000,
                "Lệch tầng / h (%)": drift / story_height * 100,
            })
            st.dataframe(seismic_table, hide_index=True, use_container_width=True)
            force_fig = go.Figure(go.Bar(x=floor_ids, y=force/1000, marker_color="#e67300", name="Fi"))
            force_fig.update_layout(title="Phân bố lực ngang theo tầng", xaxis_title="Tầng", yaxis_title="Lực tầng (kN)", height=320, margin=dict(l=15,r=15,t=45,b=30))
            drift_fig = go.Figure(go.Bar(x=floor_ids, y=drift*1000, marker_color="#3e7f66", name="Lệch tầng"))
            drift_fig.update_layout(title="Lệch tầng cực đại theo phương X", xaxis_title="Tầng", yaxis_title="Lệch tầng (mm)", height=320, margin=dict(l=15,r=15,t=45,b=30))
            ch1, ch2 = st.columns(2)
            with ch1: st.plotly_chart(force_fig, use_container_width=True)
            with ch2: st.plotly_chart(drift_fig, use_container_width=True)
            render_preliminary_strength_check(shear)
            with st.expander("Cơ sở tính và phạm vi mô hình"):
                if result["analysis_type"] == "elf":
                    st.latex(r"F_b=S_a(T_1)\,\sum_i m_i\,\lambda;\qquad F_i=F_b\frac{m_i z_i}{\sum_j m_j z_j};\qquad K u=F")
                    st.markdown("Sa(T₁) và λ do người dùng lấy/khai báo theo tiêu chuẩn và dự án. Phân bố Fi theo mi·zi là phân bố dạng tuyến tính theo chiều cao; đây là phương pháp tĩnh tương đương, không tái tạo dao động theo thời gian.")
                else:
                    st.latex(r"u_{i,r}=\phi_{i,r}\,\Gamma_r\,S_d(T_r);\qquad S_d=S_a/\omega_r^2;\qquad R=\sqrt{\sum_r\sum_s\rho_{rs}R_rR_s}")
                    st.markdown("Các mode được lấy từ Kφ = ω²Mφ; lực, chuyển vị và lệch tầng đỉnh được tổ hợp SRSS/CQC. Phổ phải bao phủ chu kỳ mode được giữ lại. Kết quả phụ thuộc vào phổ đã nạp và mô hình giảm bậc; chưa bao gồm xoắn, bất quy tắc không gian hay hiệu chỉnh theo giới hạn lực cắt đáy của tiêu chuẩn thiết kế.")
                st.info("Mô hình FEM 3D ở đây được ngưng tụ thành shear-building một bậc tự do ngang mỗi tầng. Kết quả đàn hồi theo phương X này chưa thay thế các kiểm tra và yêu cầu đầy đủ theo TCVN 9386-1:2025.")
            st.download_button("Tải bảng kết quả động đất CSV", seismic_table.to_csv(index=False).encode("utf-8-sig"), file_name="dop_tool_seismic_results.csv", mime="text/csv")
            return
        if result.get("displacement", np.empty((0,0))).ndim != 2 or result["displacement"].shape[1] != int(floors) or "story_drift" not in result or "yield_story_displacement" not in result:
            st.info("Kết quả lịch sử thời gian không còn đồng bộ. Hãy chạy lại phân tích.")
            return
        U = result["displacement"]
        if result.get("qcvn02_lookup"):
            qr = result["qcvn02_lookup"]
            st.info(f"Kích động mô phỏng dùng địa danh {qr['admin']}, {qr['province']}: a₍gR₎ = {qr['agR_g']:.3f}g theo QCVN 02:2021/BXD Bảng 6.1, trang {qr['source_page']}. Tín hiệu theo thời gian là sóng minh họa đã chuẩn hóa PGA, không phải bản ghi động đất hay tín hiệu khớp phổ.")
        story_drift = result.get("story_drift", np.diff(np.vstack([np.zeros((1,U.shape[0])),U.T]),axis=0).T)
        peak_story = np.max(np.abs(U), axis=0)
        drift_ratio = np.abs(story_drift) / story_height
        peak_drift_ratio = np.max(drift_ratio, axis=0)
        metric_cols = st.columns(min(4, int(floors)))
        for i, col in enumerate(metric_cols):
            floor = int(round(i * (int(floors) - 1) / max(1, len(metric_cols) - 1)))
            col.metric(f"Tầng {floor + 1} lệch tầng|max", f"{peak_drift_ratio[floor]*100:.3f}%")
        st.metric("Chu kỳ mode 1 FEM", f"{result['periods'][0]:.3f} s")
        base_v=np.abs(result["story_shear"][:,0])
        max_base_v=float(np.max(base_v))
        base_cap=float(result.get("base_tie_shear_capacity",0.0))
        shear_cols=st.columns(3)
        shear_cols[0].metric("Lực cắt đáy cực đại",f"{max_base_v/1000:.1f} kN")
        shear_cols[1].metric("Sức kháng đai cổ cột (ước tính)",f"{base_cap/1000:.1f} kN")
        shear_cols[2].metric("V đáy / ΣV đai (tham khảo)",f"{max_base_v/max(base_cap,1):.2f}")
        render_preliminary_strength_check(np.max(np.abs(result["story_shear"]),axis=0))
        material_summary = result.get("material_summary", "")
        st.caption(f"Vật liệu: {material_summary} · εy thép tính toán ≈ {rebar_yield_strain:.5f} · εcu bê tông = {concrete_ultimate_strain:.4f}")
        fail_t = result.get("failure_time")
        if fail_t is None:
            st.success("Chưa chạm ngưỡng biến dạng vật liệu đã chọn trong khoảng thời gian bản ghi gia tốc.")
        else:
            fail_idx = int(np.argmin(np.abs(result["time"] - fail_t)))
            fail_floor = int(np.argmax(np.abs(result["story_drift"][fail_idx]))) + 1
            st.error(f"Mốc vượt ngưỡng biến dạng vật liệu ước tính: t = {fail_t:.3f} s (tầng {fail_floor}). Đây là mốc theo mô hình giảm bậc, không phải dự báo sụp đổ được chứng nhận.")
        floor_count = U.shape[1]
        drift_scale = max(float(np.max(np.abs(U))), 0.02)
        # Animation geometry is taken from the current inputs, not from the
        # last analysis run, so changing the footprint is visible immediately.
        anx, any_, abx, aby, ah = int(bays_x), int(bays_y), float(bay_x), float(bay_y), float(story_height)
        # Color the frame by the peak estimated demand/capacity ratio, so the
        # structural utilization is visible on the same animated model.
        peak_V_story=np.max(np.abs(result["story_shear"]),axis=0)
        color_ncol=max(1,(anx+1)*(any_+1))
        color_N=np.cumsum(np.full(floor_count,floor_mass*9.80665/color_ncol)[::-1])[::-1]
        color_M=peak_V_story[:floor_count]*ah/(2*color_ncol)
        color_sigma_c=color_N/max(col_A,1e-12)+color_M*(col_depth/2)/max(col_props["effective_I"],1e-12)
        color_sigma_s=(steel_E/concrete_E)*color_M*(col_depth/2)/max(col_props["effective_I"],1e-12)
        color_fcd=0.85*{"B20":20.0,"B25":25.0,"B30":30.0,"B35":35.0,"B40":40.0,"B45":45.0,"B50":50.0,"B60":60.0}[concrete_grade]/1.5
        color_eta=np.maximum(color_sigma_c/1e6/color_fcd,color_sigma_s/1e6/rebar_yield_design)
        color_bands=[(0.50,"#25834a","η < 0,50"),(0.75,"#8dbb3f","0,50–0,75"),(0.90,"#e5c642","0,75–0,90"),(1.00,"#e58532","0,90–1,00"),(np.inf,"#c83333","η > 1,00")]
        def frame_color(eta):
            for limit,color,label in color_bands:
                if eta < limit: return color,label
            return color_bands[-1][1],color_bands[-1][2]
        stride=max(1,len(result["time"])//120)
        sample_ids=list(range(0,len(result["time"]),stride)); sample_ids[-1]=len(result["time"])-1
        footprint_polygons=[]
        for i in range(anx):
            for j in range(any_):
                if shape == "Chữ L" and i == anx-1 and j == any_-1:
                    continue
                if shape == "Chữ T" and not (j == 0 or (i == (anx-1)//2 and j > 0)):
                    continue
                footprint_polygons.append((i*abx,(i+1)*abx,j*aby,(j+1)*aby))
        def roof_trace(q):
            xvals=[]; yvals=[]; zvals=[]; xs=[]; ys=[]; zs=[]; fx=[]; fy=[]; fz=[]
            cx=[[] for _ in range(floor_count)]; cy=[[] for _ in range(floor_count)]; cz=[[] for _ in range(floor_count)]
            for k in range(floor_count+1):
                dx=0.0 if k==0 else float(U[q,k-1])
                for x0,x1,y0,y1 in footprint_polygons:
                    xvals.extend([x0+dx,x1+dx,x1+dx,x0+dx,x0+dx,None])
                    yvals.extend([y0,y0,y1,y1,y0,None])
                    zvals.extend([k*ah]*6)
                if k==0:
                    for i in range(anx+1):
                        for j in range(any_+1):
                            if shape == "Chữ L" and i==anx and j==any_: continue
                            if shape == "Chữ T" and j>0 and i not in {(anx-1)//2,(anx+1)//2}: continue
                            xs.extend([i*abx+dx,i*abx+dx,None]); ys.extend([j*aby,j*aby,None]); zs.extend([-foundation_embed,floor_count*ah,None])
                            half=footing_width/2; low=-foundation_embed-footing_depth; high=-foundation_embed
                            # Wireframe footing gives the fixed base and its
                            # buried column stub an explicit spatial extent.
                            corners=[(i*abx-half,j*aby-half),(i*abx+half,j*aby-half),(i*abx+half,j*aby+half),(i*abx-half,j*aby+half)]
                            for a0,b0 in [(0,1),(1,2),(2,3),(3,0)]:
                                for zz in (low,high):
                                    fx.extend([corners[a0][0],corners[b0][0],None]); fy.extend([corners[a0][1],corners[b0][1],None]); fz.extend([zz,zz,None])
                            for a0 in range(4):
                                fx.extend([corners[a0][0],corners[a0][0],None]); fy.extend([corners[a0][1],corners[a0][1],None]); fz.extend([low,high,None])
                if k<floor_count:
                    dx_next=float(U[q,k])
                    for i in range(anx+1):
                        for j in range(any_+1):
                            if shape=="Chữ L" and i==anx and j==any_: continue
                            if shape=="Chữ T" and j>0 and i not in {(anx-1)//2,(anx+1)//2}: continue
                            xx=i*abx; yy=j*aby
                            cx[k].extend([xx+dx,xx+dx_next,None]); cy[k].extend([yy,yy,None]); cz[k].extend([k*ah,(k+1)*ah,None])
            outline=[]
            if shape == "Chữ L":
                outline=[(0,0),(anx*abx,0),(anx*abx,aby),(abx,aby),(abx,any_*aby),(0,any_*aby),(0,0)]
            elif shape == "Chữ T":
                left=(anx-1)//2*abx; right=(anx+1)//2*abx
                outline=[(0,0),(anx*abx,0),(anx*abx,aby),(right,aby),(right,any_*aby),(left,any_*aby),(left,aby),(0,aby),(0,0)]
            else:
                outline=[(0,0),(anx*abx,0),(anx*abx,any_*aby),(0,any_*aby),(0,0)]
            for k in range(floor_count+1):
                dx=0 if k==0 else float(U[q,k-1])
                for x,y in outline:
                    xs.append(x+dx); ys.append(y); zs.append(k*ah)
                xs.append(None); ys.append(None); zs.append(None)
            traces=[go.Scatter3d(x=xvals,y=yvals,z=zvals,mode="lines",line=dict(color="#5b8def",width=3),name="Biên dạng sàn",showlegend=False,hoverinfo="skip"),go.Scatter3d(x=xs,y=ys,z=zs,mode="lines",line=dict(color="#647987",width=5),name="Dầm + cổ cột chôn",showlegend=False,hovertemplate="Khung ngàm tại móng<extra></extra>"),go.Scatter3d(x=fx,y=fy,z=fz,mode="lines",line=dict(color="#8b7355",width=5),name="Đài/móng",showlegend=False,hovertemplate="Đài/móng minh họa<extra></extra>")]
            shown=set()
            for k in range(floor_count):
                color,label=frame_color(float(color_eta[k]))
                show=label not in shown; shown.add(label)
                hover=f"Tầng {k+1}<br>η≈{color_eta[k]:.3f}<br>σc≈{color_sigma_c[k]/1e6:.2f} MPa<br>σs≈{color_sigma_s[k]/1e6:.2f} MPa<extra></extra>"
                traces.append(go.Scatter3d(x=cx[k],y=cy[k],z=cz[k],mode="lines",line=dict(color=color,width=11),name=label,legendgroup=label,showlegend=show,hovertemplate=hover))
            return traces
        initial=sample_ids[0]
        anim=go.Figure(data=roof_trace(initial))
        frames=[go.Frame(data=roof_trace(q),name=str(q)) for q in sample_ids]
        anim.frames=frames
        anim.update_layout(height=600,margin=dict(l=0,r=0,t=45,b=0),scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Z (m)",aspectmode="data",xaxis=dict(range=[-drift_scale*1.5,anx*abx+drift_scale*1.5]),yaxis=dict(range=[-aby*.1,any_*aby+aby*.1]),zaxis=dict(range=[0,floor_count*ah])),title=f"Hoạt ảnh dao động + dải η · t={result['time'][initial]:.2f} s",updatemenus=[dict(type="buttons",showactive=False,x=0.05,y=0,buttons=[dict(label="▶ Phát",method="animate",args=[None,{"frame":{"duration":35,"redraw":True},"transition":{"duration":0},"fromcurrent":True}]),dict(label="⏸ Dừng",method="animate",args=[[None],{"frame":{"duration":0,"redraw":False},"mode":"immediate"}])])],sliders=[dict(steps=[dict(method="animate",args=[[str(q)],{"mode":"immediate","frame":{"duration":0,"redraw":True},"transition":{"duration":0}}],label=f"{result['time'][q]:.1f}") for q in sample_ids],x=0.2,len=0.75)])
        st.plotly_chart(anim,use_container_width=True)
        st.caption("Nhấn Phát để xem hoạt ảnh dao động theo lịch sử thời gian đã tính bằng FEM/Newmark. Hệ số khuếch đại chuyển vị được giữ bằng 1 để thể hiện đúng biên độ tính toán.")
        fig=go.Figure()
        for i in range(floor_count): fig.add_trace(go.Scatter(x=result["time"],y=drift_ratio[:,i]*100,mode="lines",name=f"Tầng {i+1}"))
        fig.update_layout(height=340,xaxis_title="Thời gian (s)",yaxis_title="Lệch tầng Δ/h (%)",margin=dict(l=15,r=15,t=20,b=15))
        fig.add_hline(y=float(result.get("failure_drift_ratio",0.02))*100,line_dash="dash",line_color="#d62728",annotation_text="Ngưỡng biến dạng quy đổi")
        st.plotly_chart(fig,use_container_width=True)
        shear_fig=go.Figure(go.Scatter(x=result["time"],y=result["story_shear"][:,0]/1000,mode="lines",fill="tozeroy",line=dict(color="#c04b3a"),name="Vbase"))
        shear_fig.update_layout(height=300,title="Lực cắt truyền qua chân cột / móng",xaxis_title="Thời gian (s)",yaxis_title="V đáy (kN)",margin=dict(l=15,r=15,t=45,b=15))
        st.plotly_chart(shear_fig,use_container_width=True)
        st.caption(f"Sơ bộ đai cổ cột: tổng sức kháng cắt thép ngang của {result.get('n_base_columns',0)} cột được tính từ 2 nhánh đai Ø{col_tie_d*1000:.0f} @ {col_tie_spacing*1000:.0f} mm. Tỷ số trên không phải kiểm tra an toàn hoàn chỉnh; chưa gồm bê tông, ma sát mặt tiếp giáp, cốt thép chờ/neo, phá hoại đài/móng và tương tác đất.")
        drift_table=pd.DataFrame({"Tầng":np.arange(1,floor_count+1),"Lệch tầng cực đại (%)":peak_drift_ratio*100,"Độ dẻo cực đại μ":np.max(np.abs(result["story_drift"])/result["yield_story_displacement"][None,:],axis=0)})
        st.dataframe(drift_table,hide_index=True,use_container_width=True)
        st.info("Độ dẻo μ = |Δ tầng|/Δy; sức kháng chảy lấy từ bố trí cốt thép đã nhập. Đai được đưa vào độ cứng quy đổi và chỉ số kiềm chế biến dạng giản lược. Đây chưa phải mô hình thớ bê tông-cốt thép: không thay thế phân tích tiết diện theo TCVN 5574 hoặc thiết kế kháng chấn theo TCVN 9386-1:2025.")
        st.write("Độ cứng tầng tự động phân tích từ FEM (kN/m):",np.round(st.session_state.get("mdof_stiffness",[])/1000,2))
        columns = {"time": result["time"], "ground_acceleration": result["ground_acceleration"]}
        for i in range(floor_count):
            columns[f"floor_{i+1}_displacement_m"] = U[:, i]
            columns[f"story_{i+1}_drift_m"] = result["story_drift"][:, i]
            columns[f"story_{i+1}_drift_ratio"] = result["story_drift"][:, i] / story_height
            columns[f"story_{i+1}_plastic_drift_m"] = result["plastic_story_drift"][:, i]
            columns[f"story_{i+1}_shear_N"] = result["story_shear"][:, i]
            columns[f"floor_{i+1}_absolute_acceleration_m_s2"] = result["absolute_acceleration"][:, i]
        st.download_button("Tải lịch sử đáp ứng các tầng CSV", pd.DataFrame(columns).to_csv(index=False).encode("utf-8-sig"), file_name="dop_tool_story_response.csv", mime="text/csv", key="mdof_download")


def main():
    inject_css()
    st.markdown(
        '<div class="course-kicker"><span class="course-code">801120</span>'
        '<span>ĐẠI HỌC TÔN ĐỨC THẮNG · KHOA KỸ THUẬT XÂY DỰNG</span></div>',
        unsafe_allow_html=True,
    )
    title_logo, title_text, title_qr = st.columns([0.08, 0.72, 0.20], vertical_alignment="center")
    title_logo.image(os.path.join(HERE, "assets", "dop_tool_logo.png"), width=56)
    with title_text:
        st.title("Đớp Tool")
        st.caption("Tối ưu hóa kết cấu · Phép tính biến phân, FEM và động lực học")
    with title_qr:
        st.image(os.path.join(HERE, "assets", "dop_tool_qr.png"), width=110)
        st.caption("Quét mã để mở Đớp Tool")
    section = st.segmented_control("Khu vực làm việc", [
        "Cơ sở lý thuyết",
        "Tối ưu kết cấu",
        "Động lực học",
        "Động đất không gian",
    ], default="Cơ sở lý thuyết", key="main_workspace_section", width="stretch")
    st.caption("Chọn một khu vực, sau đó chọn bài toán hoặc nội dung cần xem trong danh sách bên dưới.")
    if section == "Cơ sở lý thuyết":
        tab_theory()
    elif section == "Tối ưu kết cấu":
        optimization_page = st.selectbox(
            "Chọn bài toán",
            ["Giàn / khung", "Cột", "Dầm"],
            key="main_optimization_page",
        )
        if optimization_page == "Giàn / khung":
            tab_struct()
        elif optimization_page == "Cột":
            tab_column()
        else:
            tab_beam()
    elif section == "Động lực học":
        dynamics_page = st.radio(
            "Chọn mô hình động lực học",
            ["SDOF / nhà nhiều tầng", "OpenSees · bài mẫu 10 tầng"],
            horizontal=True, key="main_dynamics_page",
        )
        if dynamics_page.startswith("SDOF"):
            tab_dynamics()
        else:
            tab_opensees_10_story()
    elif section == "Động đất không gian":
        render_earthquake_tab()


main()
