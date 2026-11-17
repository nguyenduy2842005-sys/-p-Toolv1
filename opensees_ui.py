"""Streamlit page for spatial OpenSees earthquake analyses and comparison."""
from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from opensees_core import (FrameModel, OpenSeesUnavailable, estimate_periods, frame_wire,
                           run_elf, run_mrsa, run_tha,
                           _rectangular_column_bars, _rectangular_beam_bars)
from qcvn02_lookup import lookup as qcvn_lookup, places as qcvn_places, provinces as qcvn_provinces
from seismic_methods import TYPE1_GROUND_PARAMETERS, tcvn9386_type1_elastic_spectrum


SOURCE_REPO = "https://github.com/salardelavar/OPENSEES_SALAR"
SPECTRUM_EXAMPLE = "https://github.com/salardelavar/OPENSEES_SALAR/tree/main/RESPONSE_SPECTRUM_ANALYSIS/2D_ELASTIC_FRAME"
THA_EXAMPLE = "https://github.com/salardelavar/OPENSEES_SALAR/tree/main/SDOF_TIME_HISTORY_OPENSEES_vs_SEISMOSTRUCT"


def _parse_two_columns(upload, delimiter):
    raw = np.loadtxt(BytesIO(upload.getvalue()), delimiter=delimiter, comments="#", ndmin=2)
    if raw.shape[1] < 2:
        raise ValueError("Phổ cần tối thiểu hai cột: T (s), Sa (m/s²).")
    t, sa = raw[:, 0], raw[:, 1]
    order = np.argsort(t)
    t, sa = t[order], sa[order]
    if len(t) < 2 or not np.isfinite(t).all() or not np.isfinite(sa).all() or np.any(t <= 0) or np.any(sa < 0) or np.any(np.diff(t) <= 0):
        raise ValueError("Phổ cần tối thiểu hai hàng hợp lệ, T tăng dần và Sa không âm.")
    return t, sa


@st.cache_data(show_spinner=False)
def _cached_fundamental_period(model):
    return float(estimate_periods(model)[0])


def _summary(result, model):
    if result["analysis_type"] == "time_history":
        roof = float(np.max(np.abs(result["displacement"][:, -1])))
        drift = float(np.max(np.abs(result["story_drift"])))
    else:
        roof = float(abs(result["floor_displacement"][-1]))
        drift = float(np.max(np.abs(result["story_drift"])))
    return {"roof_mm": roof * 1000.0,
            "drift_mm": drift * 1000.0,
            "drift_pct": drift / model.story_height * 100.0,
            "base_shear_kn": float(result["base_shear"]) / 1000.0}


def _comparison_table(results, model):
    names = {"elf": "ELF", "mrsa": "MRSA", "time_history": "THA"}
    summaries = {key: _summary(value, model) for key, value in results.items()}
    reference = summaries.get("time_history")
    demo_tha = bool(results.get("time_history", {}).get("motion_is_demo", False))
    rows = []
    for key in ("elf", "mrsa", "time_history"):
        if key not in summaries:
            continue
        value = summaries[key]
        row = {"Phương pháp": names[key],
               "Chuyển vị mái max (mm)": value["roof_mm"],
               "Lệch tầng max (mm)": value["drift_mm"],
               "Lệch tầng max / h (%)": value["drift_pct"],
               "Lực cắt đáy max (kN)": value["base_shear_kn"]}
        for metric, column in (("roof_mm", "Δ mái so với THA (%)"),
                               ("drift_pct", "Δ lệch tầng so với THA (%)"),
                               ("base_shear_kn", "Δ lực cắt đáy so với THA (%)")):
            ref = reference[metric] if reference else 0.0
            if demo_tha:
                row[column] = "—"
            else:
                row[column] = 0.0 if key == "time_history" else ((value[metric] / ref - 1.0) * 100.0 if ref > 1e-12 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def _response_figure(result, model):
    fig = go.Figure()
    if result["analysis_type"] == "time_history":
        fig.add_trace(go.Scatter(x=result["time"], y=result["displacement"][:, -1] * 1000,
                                 mode="lines", name="Chuyển vị mái", line=dict(color="#286b8e", width=2)))
        fig.update_layout(title="Lịch sử chuyển vị mái · OpenSees THA",
                          xaxis_title="Thời gian (s)", yaxis_title="Chuyển vị (mm)")
    else:
        elevations = np.arange(1, model.floors + 1) * model.story_height
        fig.add_trace(go.Scatter(x=np.asarray(result["floor_displacement"]) * 1000, y=elevations,
                                 mode="lines+markers", name="Chuyển vị tầng", line=dict(color="#286b8e", width=3)))
        fig.update_layout(title=f"Dạng chuyển vị · {result['analysis_type'].upper()}",
                          xaxis_title="Chuyển vị (mm)", yaxis_title="Cao độ (m)")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=15), template="plotly_white")
    return fig


def _ground_motion_figure(result):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result["time"], y=np.asarray(result["ground_acceleration"]) / 9.80665,
        mode="lines", name="Gia tốc nền", line=dict(color="#c25b37", width=1.7),
    ))
    fig.update_layout(
        title="Bản ghi kích động nền",
        xaxis_title="Thời gian (s)", yaxis_title="a₉ (g)",
        height=260, margin=dict(l=10, r=10, t=45, b=15), template="plotly_white",
    )
    return fig


def _drift_profile_figure(result, model):
    ratios = np.asarray(result.get("story_drift_ratio", result["story_drift"] / model.story_height))
    if ratios.ndim == 2:
        # Plot the governing absolute drift at each story over the full record.
        ratios = np.max(np.abs(ratios), axis=0)
    floor_ids = np.arange(1, model.floors + 1)
    limit_pct = 1.0
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=ratios * 100.0, y=floor_ids, orientation="h", name="Lệch tầng",
        marker_color=np.where(np.abs(ratios) * 100.0 <= limit_pct, "#357a63", "#d6604d"),
        hovertemplate="Tầng %{y}<br>Drift = %{x:.3f}%<extra></extra>",
    ))
    fig.add_vline(x=limit_pct, line_dash="dash", line_color="#b33b32",
                  annotation_text="Mốc minh họa 1.0%", annotation_position="top")
    fig.add_vline(x=-limit_pct, line_dash="dash", line_color="#b33b32")
    fig.update_layout(
        title="Phân bố lệch tầng theo chiều cao",
        xaxis_title="|Δᵢ| / hᵢ (%)", yaxis_title="Tầng",
        height=max(280, 34 * model.floors + 100),
        margin=dict(l=10, r=15, t=55, b=20), template="plotly_white",
        showlegend=False,
    )
    return fig


def _convex_hull_xy(x, y):
    """Monotone-chain convex hull for the accumulated response path."""
    points = sorted(set(zip(np.asarray(x, dtype=float), np.asarray(y, dtype=float))))
    if len(points) < 3:
        return np.asarray(points, dtype=float).reshape(-1, 2)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _hysteresis_figure(result, animation_key="os3d_hysteresis"):
    roof_mm = np.asarray(result["roof_displacement"], dtype=float) * 1000.0
    base_kn = np.asarray(result["base_shear_history"], dtype=float) / 1000.0
    if roof_mm.size < 2 or base_kn.size != roof_mm.size:
        raise ValueError("Dữ liệu lịch sử THA không đủ để vẽ quỹ đạo V–u.")
    duration = float(result["time"][-1]) if len(result.get("time", ())) else 0.0
    frame_count = min(100, len(roof_mm))
    frame_indices = np.unique(np.linspace(1, len(roof_mm), frame_count, dtype=int))
    frames = []
    for end in frame_indices:
        hull = _convex_hull_xy(roof_mm[:end], base_kn[:end])
        hull_x = np.r_[hull[:, 0], hull[0, 0]] if len(hull) >= 3 else hull[:, 0]
        hull_y = np.r_[hull[:, 1], hull[0, 1]] if len(hull) >= 3 else hull[:, 1]
        frames.append(go.Frame(
            name=str(int(end)),
            data=[
                go.Scatter(x=hull_x, y=hull_y),
                go.Scatter(x=roof_mm[:end], y=base_kn[:end]),
                go.Scatter(x=[roof_mm[end - 1]], y=[base_kn[end - 1]]),
            ],
            layout=go.Layout(title=f"Vòng đáp ứng tích lũy · t = {result['time'][end - 1]:.3f} s"),
        ))
    initial_end = int(frame_indices[0]) if len(frame_indices) else 1
    initial_hull = _convex_hull_xy(roof_mm[:initial_end], base_kn[:initial_end])
    initial_hull_x = np.r_[initial_hull[:, 0], initial_hull[0, 0]] if len(initial_hull) >= 3 else initial_hull[:, 0]
    initial_hull_y = np.r_[initial_hull[:, 1], initial_hull[0, 1]] if len(initial_hull) >= 3 else initial_hull[:, 1]
    fig = go.Figure(
        data=[
            go.Scatter(
                x=initial_hull_x, y=initial_hull_y,
                mode="lines", name="Convex hull · bao quỹ đạo",
                fill="toself", fillcolor="rgba(232, 110, 110, 0.24)",
                line=dict(color="#d73535", width=1.8, dash="dash"),
                hoverinfo="skip",
            ),
            go.Scatter(
                x=roof_mm[:initial_end], y=base_kn[:initial_end],
                mode="lines", name="Lịch sử lực–chuyển vị OpenSees",
                line=dict(color="#20262b", width=1.15),
                hovertemplate="u mái = %{x:.2f} mm<br>V đáy = %{y:.2f} kN<extra></extra>",
            ),
            go.Scatter(
                x=[roof_mm[initial_end - 1]], y=[base_kn[initial_end - 1]],
                mode="markers", name="Bước tải hiện tại",
                marker=dict(size=11, color="#168aad", line=dict(color="white", width=1.5)),
                hovertemplate="Bước tải hiện tại<br>u = %{x:.2f} mm<br>V = %{y:.2f} kN<extra></extra>",
            ),
        ],
        frames=frames,
    )
    max_u = max(float(np.max(np.abs(roof_mm))), 1e-6)
    max_v = max(float(np.max(np.abs(base_kn))), 1e-6)
    slider_steps = [{
        "label": f"{result['time'][int(frame.name) - 1]:.2f}",
        "method": "animate",
        "args": [[frame.name], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}, "transition": {"duration": 0}}],
    } for frame in frames]
    fig.update_layout(
        title=f"Vòng đáp ứng tích lũy · t = {result['time'][initial_end - 1]:.3f} s / {duration:.3f} s",
        xaxis_title="Chuyển vị mái (mm)", yaxis_title="Lực cắt đáy (kN)",
        height=620, margin=dict(l=15, r=15, t=70, b=75), template="plotly_white",
        hovermode="closest", legend=dict(orientation="h", y=1.13),
        xaxis=dict(domain=[0.20, 0.80], range=[-1.12 * max_u, 1.12 * max_u], zeroline=True, zerolinecolor="#65717a"),
        yaxis=dict(range=[-1.12 * max_v, 1.12 * max_v], zeroline=True, zerolinecolor="#65717a"),
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.02, "y": 1.2,
            "buttons": [
                {"label": "▶ Phát theo thời gian", "method": "animate",
                 "args": [None, {"fromcurrent": True, "mode": "immediate", "frame": {"duration": 55, "redraw": True}, "transition": {"duration": 0}}]},
                {"label": "⏸ Tạm dừng", "method": "animate",
                 "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]},
            ],
        }],
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "Thời gian gia tải: ", "suffix": " s"},
            "pad": {"t": 35}, "steps": slider_steps,
        }],
    )
    return fig


def _hysteresis_placeholder():
    drift = np.linspace(-1.0, 1.0, 320)
    restoring = 0.72 * drift + 0.28 * np.tanh(4.0 * drift)
    phase = np.linspace(0.0, 2.0 * np.pi, 360)
    x = 0.78 * np.sin(phase) + 0.12 * np.sin(3.0 * phase)
    y = 0.68 * np.sin(phase - 0.55) + 0.08 * np.sin(2.0 * phase)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=drift, y=restoring, mode="lines",
                             line=dict(color="#aeb8bf", dash="dash", width=2),
                             name="Quan hệ đàn hồi tĩnh (minh họa)"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="#d28a4f", width=2),
                             name="Dạng quỹ đạo vòng trễ (minh họa)"))
    fig.update_layout(
        title="Ví dụ dạng đồ thị lực cắt đáy – chuyển vị mái",
        xaxis_title="Chuyển vị mái, d (mm)", yaxis_title="Lực cắt đáy, V (kN)",
        height=380, margin=dict(l=10, r=10, t=55, b=20), template="plotly_white",
        annotations=[dict(
            text="Hình dáng minh họa, chưa phải kết quả mô hình hiện tại. Chọn THA, nhập hoặc tạo bản ghi, rồi chạy phân tích để vẽ đường đáp ứng thực.",
            xref="paper", yref="paper", x=0.5, y=-0.24, showarrow=False,
            font=dict(size=12, color="#5f6b73"), align="center",
        )],
    )
    return fig


def _animate_tha_figure(result, model):
    columns, beams = frame_wire(model)
    base_x, base_y, base_z = [], [], []
    base_bx, base_by, base_bz = [], [], []
    for segments, xx, yy, zz in ((columns, base_x, base_y, base_z),
                                 (beams, base_bx, base_by, base_bz)):
        for a, b in segments:
            xx.extend([a[0], b[0], None])
            yy.extend([a[1], b[1], None])
            zz.extend([a[2], b[2], None])

    time = np.asarray(result["time"], dtype=float)
    displacement = np.asarray(result["displacement"], dtype=float)
    frame_count = min(48, len(time))
    indices = np.unique(np.linspace(0, len(time) - 1, frame_count, dtype=int))
    max_u = float(np.max(np.abs(displacement))) if displacement.size else 0.0
    plan_span = max(model.bay_x * model.bays_x, model.bay_y * model.bays_y, 1.0)
    scale = min(14.0, max(1.0, 0.12 * plan_span / max(max_u, 1e-6)))
    heights = (np.arange(model.floors + 1, dtype=float) * model.story_height)
    reference_x = np.asarray(base_x, dtype=float)
    reference_y = np.asarray(base_y, dtype=float)
    reference_z = np.asarray(base_z, dtype=float)
    # This OpenSees model is elastic and currently returns floor resultants,
    # not section force/stress recorders. Use a clearly labeled story-level
    # stress estimate to color the frame, without presenting it as member FEM.
    story_shear = np.asarray(result.get("story_shear", np.zeros((len(time), model.floors))), dtype=float)
    if story_shear.ndim != 2 or story_shear.shape[1] != model.floors:
        story_shear = np.zeros((len(time), model.floors))
    col_count = max(1, len(columns))
    col_area = model.col_b * model.col_h
    col_iy = model.col_h * model.col_b**3 / 12.0
    col_iz = model.col_b * model.col_h**3 / 12.0
    c_y, c_z = model.col_b / 2.0, model.col_h / 2.0
    peak_story_stress = np.max(np.abs(story_shear), axis=0) * model.story_height / (2.0 * col_count)
    peak_story_stress = peak_story_stress * np.sqrt((c_y / max(col_iy, 1e-12))**2 + (c_z / max(col_iz, 1e-12))**2)
    stress_scale = max(float(np.max(peak_story_stress)), 1e-9)
    stress_bands = [(0.20, "#238b45", "0–20% ứng suất đỉnh"),
                    (0.40, "#8cba3c", "20–40% ứng suất đỉnh"),
                    (0.60, "#f0d23c", "40–60% ứng suất đỉnh"),
                    (0.80, "#f28e2b", "60–80% ứng suất đỉnh"),
                    (1.01, "#d62728", "80–100% ứng suất đỉnh")]

    def stress_color(ratio):
        for limit, color, label in stress_bands:
            if ratio <= limit:
                return color, label
        return stress_bands[-1][1], stress_bands[-1][2]

    frames = []
    for idx in indices:
        floor_u = np.r_[0.0, displacement[idx]] * scale
        x = reference_x.copy()
        y = reference_y.copy()
        z = reference_z.copy()
        valid = np.isfinite(z)
        levels = np.rint(np.nan_to_num(z) / model.story_height).astype(int)
        if model.direction.upper() == "X":
            x[valid] += floor_u[np.clip(levels[valid], 0, model.floors)]
        else:
            y[valid] += floor_u[np.clip(levels[valid], 0, model.floors)]

        # Color column segments by instantaneous equivalent bending stress
        # estimated from story shear; fixed scale makes colors comparable over time.
        current_shear = np.abs(story_shear[idx])
        current_stress = current_shear * model.story_height / (2.0 * col_count)
        current_stress *= np.sqrt((c_y / max(col_iy, 1e-12))**2 + (c_z / max(col_iz, 1e-12))**2)
        colored_columns = []
        legend_seen = set()
        for floor in range(model.floors):
            ratio = min(float(current_stress[floor] / stress_scale), 1.0)
            color, label = stress_color(ratio)
            seg_x, seg_y, seg_z = [], [], []
            for (a, b) in columns:
                level_a = int(round(a[2] / model.story_height))
                level_b = int(round(b[2] / model.story_height))
                if min(level_a, level_b) != floor or abs(level_a-level_b) != 1:
                    continue
                ax, ay, az = a; bx0, by0, bz0 = b
                if model.direction.upper() == "X":
                    ax += floor_u[floor]; bx0 += floor_u[floor+1]
                else:
                    ay += floor_u[floor]; by0 += floor_u[floor+1]
                seg_x.extend([ax, bx0, None]); seg_y.extend([ay, by0, None]); seg_z.extend([az, bz0, None])
            colored_columns.append(go.Scatter3d(
                x=seg_x, y=seg_y, z=seg_z, mode="lines",
                line=dict(color=color, width=10), name=label,
                legendgroup=label, showlegend=label not in legend_seen,
                hovertemplate=(f"Tầng {floor+1}<br>Ứng suất uốn cột ước tính: "
                               f"{current_stress[floor]/1e6:.3f} MPa<br>"
                               f"Tỷ lệ so với đỉnh toàn lịch sử: {ratio*100:.1f}%<extra></extra>"),
            ))
            legend_seen.add(label)

        bx = np.asarray(base_bx, dtype=float)
        by = np.asarray(base_by, dtype=float)
        bz = np.asarray(base_bz, dtype=float)
        beam_levels = np.rint(np.nan_to_num(bz) / model.story_height).astype(int)
        beam_shifts = floor_u[np.clip(beam_levels, 0, model.floors)]
        if model.direction.upper() == "X":
            bx += beam_shifts
        else:
            by += beam_shifts
        node_x, node_y, node_z = [], [], []
        plan_nodes = sorted({(round(seg[0][0], 8), round(seg[0][1], 8)) for seg in columns})
        for floor in range(model.floors + 1):
            ux = floor_u[floor] if model.direction.upper() == "X" else 0.0
            uy = floor_u[floor] if model.direction.upper() == "Y" else 0.0
            for px, py in plan_nodes:
                node_x.append(px + ux); node_y.append(py + uy); node_z.append(heights[floor])
        frames.append(go.Frame(
            name=f"{time[idx]:.3f}",
            data=[
                go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color="#315c76", width=5), name="Cột biến dạng"),
                go.Scatter3d(x=bx, y=by, z=bz, mode="lines", line=dict(color="#70848e", width=4), name="Dầm biến dạng"),
                go.Scatter3d(x=node_x, y=node_y, z=node_z, mode="markers", marker=dict(size=3.5, color="#bf5eae"), name="Tầng"),
                *colored_columns,
            ],
            layout=go.Layout(title=f"Biến dạng phóng đại ×{scale:.1f} · t = {time[idx]:.3f} s"),
        ))

    initial = frames[0].data if frames else []
    fig = go.Figure(data=initial, frames=frames)
    span_x = max(model.bays_x * model.bay_x, 1.0)
    span_y = max(model.bays_y * model.bay_y, 1.0)
    fig.update_layout(
        title=f"Dao động + dải ứng suất cột ước tính · THA · phóng đại ×{scale:.1f}",
        height=570, margin=dict(l=0, r=0, t=50, b=0), template="plotly_white",
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            xaxis=dict(range=[-0.15 * span_x, 1.15 * span_x]),
            yaxis=dict(range=[-0.15 * span_y, 1.15 * span_y]),
            zaxis=dict(range=[0.0, model.floors * model.story_height * 1.05]),
            aspectmode="data",
        ),
        updatemenus=[{
            "type": "buttons", "showactive": False, "x": 0.03, "y": 1.08,
            "buttons": [
                {"label": "▶ Chạy", "method": "animate", "args": [None, {"frame": {"duration": 80, "redraw": True}, "fromcurrent": True}]},
                {"label": "⏸ Dừng", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]},
            ],
        }],
        sliders=[{
            "active": 0, "currentvalue": {"prefix": "Thời gian: "},
            "steps": [{"label": frame.name, "method": "animate",
                       "args": [[frame.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]}
                      for frame in frames],
        }],
    )
    return fig


def render_earthquake_tab():
    st.header("Động đất · mô hình không gian OpenSees")
    st.caption("Khung dầm-cột 3D, sàn diaphragm cứng. ELF/MRSA dùng đàn hồi tuyến tính; THA có thể chọn đàn hồi hoặc phi tuyến fiber RC với cốt thép khai báo. Đơn vị SI.")
    left, right = st.columns([0.38, 0.62], gap="large")
    with left:
        floors = int(st.number_input("Số tầng", 1, 20, 4, 1, key="os3d_floors"))
        bays_x = int(st.number_input("Số nhịp X", 1, 6, 2, 1, key="os3d_bx"))
        bays_y = int(st.number_input("Số nhịp Y", 1, 6, 2, 1, key="os3d_by"))
        bay_x = float(st.number_input("Bước khung X (m)", 2.0, 12.0, 5.0, 0.5, key="os3d_sx"))
        bay_y = float(st.number_input("Bước khung Y (m)", 2.0, 12.0, 5.0, 0.5, key="os3d_sy"))
        story_height = float(st.number_input("Chiều cao tầng (m)", 2.0, 6.0, 3.3, 0.1, key="os3d_h"))
        footprint_label = st.selectbox("Mặt bằng", ["Chữ nhật", "Chữ L", "Chữ T"], key="os3d_footprint")
        footprint = {"Chữ nhật": "rect", "Chữ L": "l", "Chữ T": "t"}[footprint_label]
        direction = st.radio("Phương kích động", ["X", "Y"], horizontal=True, key="os3d_direction")
        col_a, col_b = st.columns(2)
        with col_a:
            col_b_dim = float(st.number_input("Cột b (m)", 0.2, 1.5, 0.5, 0.05, key="os3d_col_b"))
            beam_b = float(st.number_input("Dầm b (m)", 0.15, 1.0, 0.3, 0.05, key="os3d_beam_b"))
            E_gpa = float(st.number_input("E bê tông (GPa)", 10.0, 45.0, 32.5, 0.5, key="os3d_E"))
        with col_b:
            col_h = float(st.number_input("Cột h (m)", 0.2, 1.5, 0.5, 0.05, key="os3d_col_h"))
            beam_h = float(st.number_input("Dầm h (m)", 0.2, 1.5, 0.5, 0.05, key="os3d_beam_h"))
            mass_area = float(st.number_input("Khối lượng sàn quy đổi (kg/m²)", 100.0, 2000.0, 500.0, 25.0, key="os3d_mass_area"))
        damping = float(st.number_input("Tỷ số cản ξ", 0.0, 0.20, 0.05, 0.01, key="os3d_damping"))
        with st.expander("Cốt thép và vật liệu tiết diện cho THA phi tuyến", expanded=True):
            conc_grade = st.selectbox("Bê tông f'c tham chiếu", ["B20", "B25", "B30", "B35", "B40", "B45", "B50"], index=2, key="os3d_concrete_grade")
            concrete_fc = {"B20":20,"B25":25,"B30":30,"B35":35,"B40":40,"B45":45,"B50":50}[conc_grade]*1e6
            steel_grade = st.selectbox("Thép dọc · fy khai báo", ["CB300-V · 260 MPa", "CB400-V · 350 MPa", "CB500-V · 435 MPa"], index=1, key="os3d_steel_grade")
            steel_fy = float(steel_grade.split("·")[-1].replace("MPa", "").strip())*1e6
            cover = float(st.number_input("Lớp bảo vệ cốt thép (m)",0.02,0.08,0.04,0.005,format="%.3f",key="os3d_cover"))
            tie_diameter = float(st.number_input("Đường kính đai (mm)",6,16,10,2,key="os3d_tie_d"))/1000
            r1,r2,r3=st.columns(3)
            column_bar_count=int(r1.selectbox("Số thép dọc cột",[4,6,8,10,12,16,20],index=2,key="os3d_col_bars"))
            column_bar_diameter=float(r2.selectbox("Ø thép cột (mm)",[12,14,16,18,20,22,25,28,32],index=4,key="os3d_col_bar_d"))/1000
            r3.number_input("Es thép (GPa)",min_value=180.0,max_value=220.0,value=200.0,step=5.0,key="os3d_steel_E")
            c1,c2,c3=st.columns(3)
            beam_top_bar_count=int(c1.number_input("Số thanh dầm lớp trên",2,8,3,1,key="os3d_beam_top"))
            beam_bottom_bar_count=int(c2.number_input("Số thanh dầm lớp dưới",2,8,3,1,key="os3d_beam_bottom"))
            beam_bar_diameter=float(c3.selectbox("Ø thép dầm (mm)",[12,14,16,18,20,22,25,28,32],index=4,key="os3d_beam_bar_d"))/1000
            st.caption("Tiết diện Fiber dùng Concrete01 và Steel02; lực dọc sàn được quy đổi từ khối lượng tầng để tạo trạng thái trọng lực trước THA. Đai hiện chỉ xác định vị trí lớp thép; chưa sinh confinement theo mô hình Mander. Cần xác minh f'c, fy và cấu tạo bằng hồ sơ vật liệu/thí nghiệm.")
        if footprint == "l" and bays_x > 1 and bays_y > 1:
            footprint_cells = bays_x * bays_y - 1
        elif footprint == "t" and bays_x > 1 and bays_y > 1:
            footprint_cells = bays_x + bays_y - 1
        else:
            footprint_cells = bays_x * bays_y
        area = footprint_cells * bay_x * bay_y
        floor_mass = mass_area * area
        model = FrameModel(floors, bays_x, bays_y, bay_x, bay_y, story_height,
                           floor_mass, E_gpa * 1.0e9, col_b_dim, col_h,
                           beam_b, beam_h, footprint, direction,
                           concrete_fc=concrete_fc, steel_fy=steel_fy,
                           steel_E=float(st.session_state["os3d_steel_E"])*1e9,
                           cover=cover,tie_diameter=tie_diameter,
                           column_bar_diameter=column_bar_diameter,
                           column_bar_count=column_bar_count,
                           beam_top_bar_count=beam_top_bar_count,
                           beam_bottom_bar_count=beam_bottom_bar_count,
                           beam_bar_diameter=beam_bar_diameter)
        st.caption(f"Khối lượng sàn quy đổi: {floor_mass:,.0f} kg/tầng · diện tích theo mặt bằng: {area:.1f} m².")

    with right:
        columns, beams = frame_wire(model)
        fig = go.Figure()

        def box_mesh(segments, kind):
            mx, my, mz, ii, jj, kk = [], [], [], [], [], []
            for a, b in segments:
                if kind == "column":
                    hx, hy, hz = model.col_b/2, model.col_h/2, 0.0
                    verts = [(a[0]-hx,a[1]-hy,a[2]),(a[0]+hx,a[1]-hy,a[2]),
                             (a[0]+hx,a[1]+hy,a[2]),(a[0]-hx,a[1]+hy,a[2]),
                             (b[0]-hx,b[1]-hy,b[2]),(b[0]+hx,b[1]-hy,b[2]),
                             (b[0]+hx,b[1]+hy,b[2]),(b[0]-hx,b[1]+hy,b[2])]
                else:
                    along_x = abs(b[0]-a[0]) >= abs(b[1]-a[1])
                    hw, hh = model.beam_b/2, model.beam_h/2
                    if along_x:
                        verts = [(a[0],a[1]-hw,a[2]-hh),(a[0],a[1]+hw,a[2]-hh),
                                 (a[0],a[1]+hw,a[2]+hh),(a[0],a[1]-hw,a[2]+hh),
                                 (b[0],b[1]-hw,b[2]-hh),(b[0],b[1]+hw,b[2]-hh),
                                 (b[0],b[1]+hw,b[2]+hh),(b[0],b[1]-hw,b[2]+hh)]
                    else:
                        verts = [(a[0]-hw,a[1],a[2]-hh),(a[0]+hw,a[1],a[2]-hh),
                                 (a[0]+hw,a[1],a[2]+hh),(a[0]-hw,a[1],a[2]+hh),
                                 (b[0]-hw,b[1],b[2]-hh),(b[0]+hw,b[1],b[2]-hh),
                                 (b[0]+hw,b[1],b[2]+hh),(b[0]-hw,b[1],b[2]+hh)]
                base = len(mx)
                for vx,vy,vz in verts:
                    mx.append(vx); my.append(vy); mz.append(vz)
                faces = ((0,1,2),(0,2,3),(4,6,5),(4,7,6),(0,4,5),(0,5,1),
                         (1,5,6),(1,6,2),(2,6,7),(2,7,3),(3,7,4),(3,4,0))
                for p,q,r in faces:
                    ii.append(base+p); jj.append(base+q); kk.append(base+r)
            return mx,my,mz,ii,jj,kk

        for segments,kind,color,label in (
            (columns,"column","#8eb8a0","Bê tông cột · trong suốt"),
            (beams,"beam","#b7c4cc","Bê tông dầm · trong suốt"),
        ):
            mesh=box_mesh(segments,kind)
            fig.add_trace(go.Mesh3d(x=mesh[0],y=mesh[1],z=mesh[2],i=mesh[3],j=mesh[4],k=mesh[5],
                color=color,opacity=0.22,name=label,flatshading=True,hoverinfo="skip"))
        for segments, color, name, width in ((columns, "#315c76", "Cột", 4), (beams, "#8d9aa3", "Dầm", 3)):
            xs, ys, zs = [], [], []
            for a, b in segments:
                xs.extend([a[0], b[0], None]); ys.extend([a[1], b[1], None]); zs.extend([a[2], b[2], None])
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", name=name,
                                       line=dict(color=color, width=width)))
        # Show the rebar layout within the spatial model. These offsets are
        # intentionally exaggerated so longitudinal bars remain visible at frame scale.
        bar_scale = 1.0
        col_bar_pts = _rectangular_column_bars(model)
        rebar_x, rebar_y, rebar_z = [], [], []
        for x0, y0 in sorted({(round(a[0], 8), round(a[1], 8)) for a, _ in columns}):
            for by, bz, _, _ in col_bar_pts:
                rebar_x.extend([x0 + by * bar_scale, x0 + by * bar_scale, None])
                rebar_y.extend([y0 + bz * bar_scale, y0 + bz * bar_scale, None])
                rebar_z.extend([0, floors * story_height, None])
        fig.add_trace(go.Scatter3d(x=rebar_x, y=rebar_y, z=rebar_z, mode="lines",
                                   name="Thép dọc cột (vị trí phóng đại)",
                                   line=dict(color="#c7463b", width=3),
                                   hovertemplate="Thép dọc cột · bố trí từ mặt cắt khai báo<extra></extra>"))
        beam_pts = _rectangular_beam_bars(model)
        beam_rx, beam_ry, beam_rz = [], [], []
        for (a, b) in beams:
            along_x = abs(b[0]-a[0]) >= abs(b[1]-a[1])
            for by, bz, _, _ in beam_pts:
                if along_x:
                    beam_rx.extend([a[0], b[0], None])
                    beam_ry.extend([a[1] + by * bar_scale, b[1] + by * bar_scale, None])
                else:
                    beam_rx.extend([a[0] + by * bar_scale, b[0] + by * bar_scale, None])
                    beam_ry.extend([a[1], b[1], None])
                beam_rz.extend([a[2] + bz * bar_scale, b[2] + bz * bar_scale, None])
        fig.add_trace(go.Scatter3d(x=beam_rx, y=beam_ry, z=beam_rz, mode="lines",
                                   name="Thép dọc dầm (vị trí phóng đại)",
                                   line=dict(color="#e29b35", width=2),
                                   hovertemplate="Thép dọc dầm · lớp trên/dưới<extra></extra>"))
        fig.update_layout(title="Mô hình khung 3D · bê tông trong suốt và cốt thép dọc", height=540,
                          scene=dict(xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)", aspectmode="data"),
                          margin=dict(l=0, r=0, t=45, b=0), legend=dict(orientation="h"), template="plotly_white")
        st.caption("Bê tông dầm/cột được dựng thành khối bán trong suốt; thép dọc màu đỏ/cam nằm bên trong theo kích thước tiết diện đã khai báo. Hình học cốt thép không phóng đại; đai được thể hiện ở mặt cắt bên dưới.")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Mặt cắt và cấu tạo cốt thép đang khai báo", expanded=False):
            section_cols=st.columns(2)
            section_defs = (
                ("Cột", col_b_dim, col_h, _rectangular_column_bars(model),
                 column_bar_diameter, column_bar_count),
                ("Dầm", beam_b, beam_h, _rectangular_beam_bars(model),
                 beam_bar_diameter, beam_top_bar_count + beam_bottom_bar_count),
            )
            for idx,(label,width,depth,bars,bar_diameter,bar_count) in enumerate(section_defs):
                sf=go.Figure()
                tie_x = max(0.0, width/2 - cover - tie_diameter/2)
                tie_y = max(0.0, depth/2 - cover - tie_diameter/2)
                sf.add_shape(type="rect",x0=-width/2,y0=-depth/2,x1=width/2,y1=depth/2,
                             line=dict(color="#244a35",width=3),fillcolor="#d8e6d5",layer="below")
                sf.add_shape(type="rect",x0=-tie_x,y0=-tie_y,x1=tie_x,y1=tie_y,
                             line=dict(color="#e67300",width=2),fillcolor="rgba(0,0,0,0)")
                sf.add_trace(go.Scatter(x=[p[0] for p in bars],y=[p[1] for p in bars],mode="markers",
                    marker=dict(size=13,color="#bf3028",symbol="circle",line=dict(color="white",width=1.5)),
                    name="Thép dọc",hovertemplate=f"Cốt dọc Ø{bar_diameter*1000:.0f} mm<extra></extra>"))
                sf.update_layout(title=f"Mặt cắt {label} · bố trí thép dọc và đai",height=360,
                    showlegend=False,margin=dict(l=20,r=10,t=48,b=15),
                    xaxis=dict(title="bề rộng (m)",scaleanchor="y",range=[-width*.65,width*.65]),
                    yaxis=dict(title="chiều cao (m)",range=[-depth*.65,depth*.65]),template="plotly_white")
                sf.add_annotation(x=0,y=-depth*.59,
                    text=f"{bar_count} thanh dọc · đai Ø{tie_diameter*1000:.0f} · cover={cover*1000:.0f} mm",
                    showarrow=False,font=dict(size=11,color="#637078"))
                section_cols[idx].plotly_chart(sf,use_container_width=True)
            column_area = column_bar_count * np.pi * column_bar_diameter**2 / 4
            beam_top_area = beam_top_bar_count * np.pi * beam_bar_diameter**2 / 4
            beam_bottom_area = beam_bottom_bar_count * np.pi * beam_bar_diameter**2 / 4
            st.caption(f"As cột ≈ {column_area*1e6:.0f} mm² · As dầm trên/dưới ≈ {beam_top_area*1e6:.0f}/{beam_bottom_area*1e6:.0f} mm² · đai màu cam · thép dọc màu đỏ. Khoảng cách đai chưa được khai báo; đai hiện chưa tạo hiệu ứng kiềm chế bê tông trong Fiber section.")

    st.divider()
    st.subheader("Tải trọng động đất và phương pháp")
    spectrum_source = st.radio("Phổ dùng cho ELF / MRSA", ["Phổ mẫu QCVN 02 + Type 1", "Nạp phổ riêng CSV"],
                               horizontal=True, key="os3d_spectrum_source")
    spectrum_periods = spectrum_sa = None
    qcvn_meta = None
    if spectrum_source == "Phổ mẫu QCVN 02 + Type 1":
        c1, c2, c3 = st.columns([1.4, 1.2, 1.0])
        provinces = qcvn_provinces()
        default_province = "Thành phố Hà Nội" if "Thành phố Hà Nội" in provinces else provinces[0]
        province = c1.selectbox("Tỉnh / thành phố", provinces, index=provinces.index(default_province), key="os3d_province")
        places = qcvn_places(province)
        default_place = "Quận Ba Đình" if province == "Thành phố Hà Nội" and "Quận Ba Đình" in places else places[0]
        admin = c2.selectbox("Quận / huyện", places, index=places.index(default_place), key="os3d_admin")
        soil = c3.selectbox("Nền Type 1", list(TYPE1_GROUND_PARAMETERS), index=1, key="os3d_soil")
        gamma_i = st.number_input("Hệ số tầm quan trọng γI", 0.1, 3.0, 1.0, 0.1, key="os3d_gamma")
        qcvn_meta = qcvn_lookup(province, admin)
        spectrum_periods = np.geomspace(0.01, 8.0, 800)
        spectrum_sa = tcvn9386_type1_elastic_spectrum(
            spectrum_periods, qcvn_meta["agR_g"] * 9.80665, soil, gamma_i, damping)
        st.caption(f"a₍gR₎ = {qcvn_meta['agR_g']:.3f} g · {admin}, {province} · QCVN 02:2021/BXD, Bảng 6.1. Phổ đàn hồi Type 1, q=1; cần xác nhận tham số nền và hệ số với tiêu chuẩn/dự án.")
    else:
        upload_spectrum = st.file_uploader("Phổ CSV/TXT hai cột: T (s), Sa (m/s²)", ["csv", "txt"], key="os3d_spectrum_upload")
        if upload_spectrum is not None:
            try:
                spectrum_periods, spectrum_sa = _parse_two_columns(upload_spectrum, "," if upload_spectrum.name.lower().endswith(".csv") else None)
            except Exception as exc:
                st.error(f"Không đọc được phổ: {exc}")

    method_label = st.selectbox("Phương pháp phân tích", [
        "Tĩnh lực ngang tương đương · Equivalent Lateral Force (ELF)",
        "Phổ phản ứng · Modal Response Spectrum Analysis (MRSA)",
        "Lịch sử thời gian · Time-History Analysis (THA)",
    ], key="os3d_method")
    nonlinear_tha = False
    if method_label.startswith("Lịch sử thời gian"):
        a, b, c = st.columns([1, 1, 1.4])
        dt = float(a.number_input("Bước thời gian Δt (s)", 0.0001, 0.1, 0.01, 0.001, format="%.4f", key="os3d_dt"))
        acc_unit = b.selectbox("Đơn vị gia tốc", ["m/s²", "g"], key="os3d_acc_unit")
        upload_motion = st.file_uploader("Bản ghi gia tốc một cột", ["csv", "txt"], key="os3d_motion_upload")
        motion = None
        motion_label = ""
        if upload_motion is not None:
            try:
                raw = np.loadtxt(BytesIO(upload_motion.getvalue()), delimiter="," if upload_motion.name.lower().endswith(".csv") else None, comments="#")
                motion = np.asarray(raw, dtype=float).reshape(-1)
                if acc_unit == "g":
                    motion *= 9.80665
                motion_label = upload_motion.name
            except Exception as exc:
                st.error(f"Không đọc được bản ghi: {exc}")
        demo = st.checkbox("Tạo tín hiệu động đất minh họa nhiều chu kỳ (không phải accelerogram thiết kế)",
                           value=True, key="os3d_demo_motion")
        if demo and motion is None:
            pga_g = float(st.number_input("PGA tín hiệu minh họa (g)", 0.01, 2.0,
                                          min(0.15, max(0.05, float(qcvn_meta["agR_g"]))) if qcvn_meta else 0.10,
                                          0.01, key="os3d_demo_pga"))
            try:
                t1_demo = _cached_fundamental_period(model)
                st.caption(f"Tín hiệu mặc định đặt gần T₁ = {t1_demo:.3f} s do OpenSees eigen tính từ khung FEM, để làm rõ đáp ứng cộng hưởng nhiều chu kỳ.")
            except Exception as exc:
                t1_demo = 1.0
                st.warning(f"Chưa lấy được T₁ từ FEM để đặt tín hiệu minh họa: {exc}. Tạm dùng T = 1.0 s.")
            auto_period = st.checkbox("Tự động đặt chu kỳ tín hiệu gần T₁", value=True, key="os3d_demo_auto_t1")
            if auto_period:
                period = float(t1_demo)
                st.number_input("Chu kỳ tín hiệu minh họa (s)", value=period, disabled=True,
                                format="%.3f", key="os3d_demo_period_auto")
            else:
                period = float(st.number_input("Chu kỳ tín hiệu (s)", 0.05, 5.0,
                                                max(0.05, min(5.0, t1_demo)), 0.05,
                                                key="os3d_demo_period"))
            cycles = int(st.number_input("Số chu kỳ tải đảo chiều", 2, 20, 6, 1,
                                         key="os3d_demo_cycles"))
            duration = min(60.0, max(dt * 2.0, period * cycles))
            st.caption(f"Thời lượng tự chọn theo số chu kỳ: {duration:.2f} s · {cycles} chu kỳ mục tiêu.")
            t = np.arange(int(round(duration / dt)) + 1) * dt
            # A band-limited, amplitude-modulated multi-cycle signal near T1.
            # The small neighbouring components make the elastic dynamic path
            # less like a single perfect ellipse while preserving a clear
            # sequence of load reversals for the animated V-u illustration.
            tau = np.clip(t / duration, 0.0, 1.0)
            envelope = np.sin(np.pi * tau)**0.65
            phase = 2.0 * np.pi * t / period
            carrier = (np.sin(phase) + 0.22 * np.sin(1.8 * phase + 0.7)
                       + 0.12 * np.sin(0.63 * phase + 1.1))
            motion = envelope * carrier
            motion *= (pga_g * 9.80665) / max(float(np.max(np.abs(motion))), 1e-12)
            motion_label = "Tín hiệu tổng hợp nhiều chu kỳ gần T₁ (minh họa)"
        nonlinear_tha = st.radio("Mô hình THA", ["Đàn hồi tuyến tính", "Phi tuyến fiber RC · có cốt thép"], horizontal=True, key="os3d_tha_model").startswith("Phi tuyến")
        if nonlinear_tha:
            st.info("OpenSees sẽ dùng Fiber section + nonlinearBeamColumn với Concrete01/Steel02 và phân tích trọng lực trước động đất. Bố trí thép dọc trong các mặt cắt bên trên tham gia trực tiếp vào sức kháng uốn và chảy dẻo. Kết quả còn phụ thuộc mô hình vật liệu; hiện chưa mô hình hóa nứt kéo bê tông, tuột neo, phá hoại cắt, confined concrete theo Mander hay tương tác đất–móng.")
    else:
        dt, motion, motion_label = 0.01, None, ""

    combination = st.selectbox("Tổ hợp modal cho MRSA", ["CQC", "SRSS"], key="os3d_combination")
    lambda_factor = float(st.number_input("Hệ số λ của ELF", 0.1, 1.0, 1.0, 0.05, key="os3d_lambda"))
    selected_method = "elf" if method_label.startswith("Tĩnh lực") else ("mrsa" if method_label.startswith("Phổ phản ứng") else "time_history")

    structure_signature = (floors, bays_x, bays_y, bay_x, bay_y, story_height, floor_mass,
                           E_gpa, col_b_dim, col_h, beam_b, beam_h, footprint, direction,
                           concrete_fc,steel_fy,float(st.session_state["os3d_steel_E"])*1e9,
                           cover,tie_diameter,column_bar_diameter,column_bar_count,
                           beam_top_bar_count,beam_bottom_bar_count,beam_bar_diameter,nonlinear_tha)
    spectrum_digest = hashlib.sha256(np.asarray(spectrum_periods if spectrum_periods is not None else [], dtype=float).tobytes() +
                                     np.asarray(spectrum_sa if spectrum_sa is not None else [], dtype=float).tobytes()).hexdigest()
    comparison_signature = (structure_signature, spectrum_digest, damping, lambda_factor)
    if st.session_state.get("os3d_comparison_signature") != comparison_signature:
        st.session_state["os3d_comparison_signature"] = comparison_signature
        st.session_state["os3d_results"] = {}
    results = st.session_state.setdefault("os3d_results", {})
    if selected_method == "time_history":
        motion_digest = hashlib.sha256(
            (np.asarray(motion, dtype=float).tobytes() if motion is not None else b"")
            + str(dt).encode("ascii") + b"tha-response-v2"
        ).hexdigest()
        if st.session_state.get("os3d_motion_signature") != motion_digest:
            results.pop("time_history", None)
            st.session_state["os3d_motion_signature"] = motion_digest

    run_disabled = (selected_method in {"elf", "mrsa"} and (spectrum_periods is None or spectrum_sa is None)) or \
                   (selected_method == "time_history" and (motion is None or len(motion) < 2))
    if st.button(f"Chạy OpenSees · {selected_method.upper()}", type="primary", disabled=run_disabled, key="os3d_run"):
        try:
            if selected_method == "elf":
                result = run_elf(model, spectrum_periods, spectrum_sa, lambda_factor)
            elif selected_method == "mrsa":
                result = run_mrsa(model, spectrum_periods, spectrum_sa, damping, combination)
            else:
                result = run_tha(model, motion, dt, damping, nonlinear=nonlinear_tha)
                result["motion_label"] = motion_label
                if nonlinear_tha:
                    # Same-record elastic reference makes the effect of the
                    # reinforcement-dependent fiber model inspectable.
                    elastic_ref=run_tha(model,motion,dt,damping,nonlinear=False)
                    result["elastic_reference"]={
                        "roof_mm":float(np.max(np.abs(elastic_ref["displacement"][:,-1]))*1000),
                        "drift_pct":float(np.max(np.abs(elastic_ref["story_drift"]))/model.story_height*100),
                        "base_shear_kn":float(np.max(np.abs(elastic_ref["base_shear_history"]))/1000),
                    }
            result["motion_is_demo"] = "minh họa" in motion_label.lower()
            result["direction"] = direction
            result["spectrum_source"] = spectrum_source
            result["qcvn_lookup"] = qcvn_meta
            results[selected_method] = result
            st.session_state["os3d_results"] = results
            st.success(f"Hoàn thành {result['method_name']} bằng OpenSees.")
        except OpenSeesUnavailable as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"OpenSees không hoàn tất phân tích: {exc}")

    result = results.get(selected_method)
    if result is not None:
        summary = _summary(result, model)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Chu kỳ tham gia lớn nhất", f"{result.get('dominant_period', result['periods'][0]):.3f} s")
        c2.metric("Chuyển vị mái lớn nhất", f"{summary['roof_mm']:.2f} mm")
        c3.metric("Lệch tầng cực đại", f"{summary['drift_pct']:.3f} %")
        c4.metric("Lực cắt đáy", f"{summary['base_shear_kn']:.2f} kN")
        if result.get("nonlinear") and result.get("elastic_reference"):
            ref=result["elastic_reference"]
            comp=pd.DataFrame([
                ("Chuyển vị mái lớn nhất (mm)",summary["roof_mm"],ref["roof_mm"]),
                ("Lệch tầng cực đại (%)",summary["drift_pct"],ref["drift_pct"]),
                ("Lực cắt đáy cực đại (kN)",summary["base_shear_kn"],ref["base_shear_kn"]),
            ],columns=["Đại lượng","Phi tuyến fiber RC","Đàn hồi tham chiếu"])
            st.markdown("#### So sánh cùng gia tốc nền: có dẻo và đàn hồi")
            st.dataframe(comp.round(3),hide_index=True,use_container_width=True)
            st.caption(f"Fiber section dùng f'c={model.concrete_fc/1e6:.0f} MPa, fy={model.steel_fy/1e6:.0f} MPa; cột {model.column_bar_count}Ø{model.column_bar_diameter*1000:.0f}, dầm {model.beam_top_bar_count}+{model.beam_bottom_bar_count}Ø{model.beam_bar_diameter*1000:.0f}. Hàng đàn hồi là đối chứng cùng hình học, tải và cản.")
        if result.get("analysis_type") == "time_history":
            with st.expander("Vòng V–u chuyển động theo thời gian gia tải", expanded=True):
                st.plotly_chart(_hysteresis_figure(result), use_container_width=True,
                                key="os3d_hysteresis")
                st.caption("Nhấn ▶ hoặc kéo thanh thời gian để xem lịch sử V–u. Với mô hình fiber RC, độ mở vòng phản ánh đáp ứng phi tuyến của tiết diện phân bố; với mô hình đàn hồi, đáp ứng không có tiêu tán dẻo.")
            st.plotly_chart(_animate_tha_figure(result, model), use_container_width=True,
                            key="os3d_tha_animation")
            if result.get("motion_is_demo"):
                st.caption("Hoạt ảnh dùng kết quả phân tích theo thời gian; biến dạng hiển thị đã phóng đại để quan sát, không theo tỷ lệ thực.")
            st.caption("Dải xanh–vàng–cam–đỏ tô theo ứng suất uốn cột ước tính từ lực cắt tầng và tỷ lệ so với mức lớn nhất trong chính bản ghi này; màu không phải tỷ số đạt/không đạt. Backend OpenSees hiện chưa ghi nội lực cục bộ từng phần tử, nên đây chưa phải contour ứng suất FEM như ETABS/STAAD.Pro.")
            chart_left, chart_right = st.columns(2)
            with chart_left:
                st.plotly_chart(_ground_motion_figure(result), use_container_width=True,
                                key="os3d_ground_motion")
            with chart_right:
                st.plotly_chart(_drift_profile_figure(result, model), use_container_width=True,
                                key="os3d_drift_profile")
            st.caption("Mốc lệch tầng 1.0% trên biểu đồ drift chỉ để tham khảo minh họa, không thay thế giới hạn theo tiêu chuẩn áp dụng.")
        else:
            response_col, drift_col = st.columns(2)
            with response_col:
                st.plotly_chart(_response_figure(result, model), use_container_width=True)
            with drift_col:
                st.plotly_chart(_drift_profile_figure(result, model), use_container_width=True)
        if result.get("mode_table"):
            st.dataframe(pd.DataFrame(result["mode_table"]).round(4), hide_index=True, use_container_width=True)
        if result.get("analysis_type") == "mrsa" and result.get("effective_mass_total", 0.0) < 0.90:
            st.warning(f"Các mode giữ lại mới đạt {result['effective_mass_total'] * 100:.1f}% khối lượng hữu hiệu theo phương kích động; cần tăng số mode trong mô hình trước khi dùng kết quả.")
        if result.get("analysis_type") == "time_history":
            if result.get("motion_is_demo"):
                st.warning("Đáp ứng THA dùng tín hiệu tổng hợp nhiều chu kỳ để minh họa tải đảo chiều; đây không phải accelerogram thực, chưa hiệu chỉnh phổ và không dùng làm kết quả thiết kế.")
            else:
                st.caption(f"Bản ghi: {result.get('motion_label', 'đã nạp')} · Rayleigh αM={result['rayleigh_alpha_m']:.4g} s⁻¹, βK={result['rayleigh_beta_k']:.4g} s.")

    if selected_method == "time_history" and result is None:
        with st.expander("Ví dụ vòng V–u chuyển động theo thời gian gia tải", expanded=True):
            st.plotly_chart(_hysteresis_placeholder(), use_container_width=True,
                            key="os3d_hysteresis_preview")

    st.subheader("So sánh kết quả ba phương pháp")
    if results:
        st.dataframe(_comparison_table(results, model).round(3), hide_index=True, use_container_width=True)
        if results.get("time_history", {}).get("motion_is_demo"):
            st.warning("THA hiện dùng tín hiệu tổng hợp nhiều chu kỳ đặt gần cộng hưởng T₁ để minh họa, chưa hiệu chỉnh theo phổ QCVN. Vì vậy đáp ứng THA có thể cao hơn nhiều so với ELF/MRSA; các cột Δ được để trống và không dùng để so sánh thiết kế. Hãy nạp accelerogram thực đã xử lý nền và hiệu chỉnh/scale theo phổ mục tiêu để đối chiếu.")
    else:
        st.info("Chạy từng phương pháp trên cùng mô hình và cùng phổ để lấp bảng so sánh ELF–MRSA–THA.")
    st.caption("Các cột Δ là chênh lệch đáp ứng so với THA: (kết quả phương pháp − THA) / THA × 100%. Đây là chênh lệch giữa các phương pháp, không phải sai số số học. So sánh có ý nghĩa khi THA dùng accelerogram thực đã xử lý và hiệu chỉnh theo phổ mục tiêu đang dùng cho ELF/MRSA.")

    with st.expander("Cơ sở mô hình OpenSees và nguồn tham khảo"):
        st.markdown(f"""
        - Lưới cột–dầm 3D đàn hồi dùng `elasticBeamColumn`; THA phi tuyến thay bằng `nonlinearBeamColumn` với Fiber sections. Nút chân cột ngàm, sàn liên kết bằng `rigidDiaphragm` và khối lượng tập trung tại tầng.
        - ELF lấy chu kỳ/mode từ eigen OpenSees, phân bố lực theo `mᵢzᵢ` và giải static.
        - MRSA lấy mode shape từ eigen OpenSees, nội suy Sa(T), tính đáp ứng modal rồi tổ hợp SRSS/CQC bằng Python.
        - THA dùng `UniformExcitation`, `Path` time series, cản Rayleigh và Newmark. Chế độ Fiber RC dùng `Concrete01` + `Steel02`, tích phân Lobatto 5 điểm, tải trọng trọng lực quy đổi và so sánh cùng bản ghi với mô hình đàn hồi.
        - Backend gọi trực tiếp `openseespy.opensees`. Trên Windows, wheel 3.8 yêu cầu Python 3.12 64-bit do liên kết với `python312.dll`. Mã được phát triển theo quy trình và ví dụ của Salar Delavar Ghashghaei, [OPENSEES_SALAR]({SOURCE_REPO}) (MIT License): [khung phổ phản ứng]({SPECTRUM_EXAMPLE}), [ví dụ lịch sử thời gian]({THA_EXAMPLE}). Phần dựng khung không gian và giao diện là phần phát triển riêng của Đớp Tool.
        """)
        st.warning("Fiber THA mới mô hình phi tuyến vật liệu uốn; chưa có confinement bê tông theo Mander, nứt kéo bê tông, tuột neo, phá hoại cắt, suy giảm cường độ theo chu kỳ, P-Delta, tương tác đất–móng hoặc kiểm tra đầy đủ theo tiêu chuẩn. Trọng lực dùng khối lượng sàn quy đổi, chưa gồm tự trọng cấu kiện. Đây là mô hình nghiên cứu/giảng dạy, không phải hồ sơ thiết kế đã thẩm tra.")

