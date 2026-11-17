"""Spatial elastic-frame seismic procedures driven by OpenSeesPy.

The model-generation and response-extraction workflow is adapted from the
OpenSeesPy examples by Salar Delavar Ghashghaei (OPENSEES_SALAR, MIT license),
with a purpose-built 3D moment-frame generator for this application.
All quantities use SI units: m, kg, N, s, and m/s².
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from pathlib import Path
import sys

import numpy as np


class OpenSeesUnavailable(RuntimeError):
    """Raised when the optional OpenSeesPy runtime is not installed."""


_DLL_DIRECTORY_HANDLES = []
_DLL_DIRECTORY_PATHS = set()


def _prepare_windows_dll_search():
    if sys.platform != "win32":
        return
    candidates = [Path(sys.prefix) / "Library" / "bin",
                  Path(sys.base_prefix) / "Library" / "bin",
                  Path.home() / "anaconda3" / "Library" / "bin",
                  Path.home() / "miniconda3" / "Library" / "bin"]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "Library" / "bin")
    candidates.extend(Path(folder) for folder in os.environ.get("PATH", "").split(os.pathsep) if folder)
    for folder in candidates:
        try:
            key = str(folder.resolve()).casefold()
            if key in _DLL_DIRECTORY_PATHS or not (folder / "libiomp5md.dll").is_file():
                continue
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(folder)))
            _DLL_DIRECTORY_PATHS.add(key)
        except (OSError, ValueError):
            continue


@dataclass(frozen=True)
class FrameModel:
    floors: int
    bays_x: int
    bays_y: int
    bay_x: float
    bay_y: float
    story_height: float
    floor_mass: float
    E: float
    col_b: float
    col_h: float
    beam_b: float
    beam_h: float
    footprint: str = "rect"
    direction: str = "X"
    concrete_fc: float = 30.0e6
    steel_fy: float = 350.0e6
    steel_E: float = 200.0e9
    cover: float = 0.04
    tie_diameter: float = 0.01
    column_bar_diameter: float = 0.02
    column_bar_count: int = 8
    beam_top_bar_count: int = 3
    beam_bottom_bar_count: int = 3
    beam_bar_diameter: float = 0.02


def _ops():
    _prepare_windows_dll_search()
    if sys.platform == "win32" and sys.version_info[:2] != (3, 12):
        raise OpenSeesUnavailable(
            "OpenSeesPy 3.8.0.0 trên Windows liên kết với python312.dll, nên cần Python 3.12 64-bit. "
            f"Đớp Tool hiện đang chạy Python {sys.version_info.major}.{sys.version_info.minor}. "
            "Khởi chạy bằng Dop_Tool.bat để dùng môi trường .venv312."
        )
    try:
        import openseespy.opensees as ops
        # Force loading and initialization of the native solver at the boundary.
        ops.version()
        return ops
    except Exception as exc:
        raise OpenSeesUnavailable(
            "Không nạp được OpenSeesPy trực tiếp. Hãy cài `openseespy>=3.8.0.0` "
            "vào đúng môi trường Python 3.12 64-bit và kiểm tra runtime Windows. "
            f"Chi tiết: {exc}"
        ) from exc


def _rectangle_torsion(width: float, depth: float) -> float:
    long_side, short_side = max(width, depth), min(width, depth)
    ratio = short_side / long_side
    return short_side * long_side**3 * (1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0))


def _valid_cells(model: FrameModel):
    nx, ny = int(model.bays_x), int(model.bays_y)
    shape = str(model.footprint).lower()
    cells = []
    for ix in range(nx):
        for iy in range(ny):
            if shape == "l" and nx > 1 and ny > 1 and ix == nx - 1 and iy == ny - 1:
                continue
            if shape == "t" and nx > 1 and ny > 1 and iy < ny - 1 and ix != nx // 2:
                continue
            cells.append((ix, iy))
    return cells


def frame_wire(model: FrameModel):
    """Return column and beam line segments for a spatial preview."""
    cells = _valid_cells(model)
    nodes = set()
    edges = set()
    for ix, iy in cells:
        corners = ((ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1))
        nodes.update(corners)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            edges.add(tuple(sorted((a, b))))
    columns, beams = [], []
    for ix, iy in sorted(nodes):
        x, y = ix * model.bay_x, iy * model.bay_y
        for floor in range(model.floors):
            columns.append(((x, y, floor * model.story_height),
                            (x, y, (floor + 1) * model.story_height)))
    for floor in range(1, model.floors + 1):
        z = floor * model.story_height
        for (ix1, iy1), (ix2, iy2) in sorted(edges):
            beams.append(((ix1 * model.bay_x, iy1 * model.bay_y, z),
                          (ix2 * model.bay_x, iy2 * model.bay_y, z)))
    return columns, beams


def _add_rc_fiber_section(ops, model: FrameModel, section_tag: int,
                          width: float, depth: float, bars_yz,
                          torsion_j: float, concrete_tag: int, steel_tag: int):
    """Create a rectangular RC fiber section with user-entered longitudinal bars."""
    bar_diameter = (model.column_bar_diameter if section_tag == 101
                    else model.beam_bar_diameter)
    if 2 * (model.cover + model.tie_diameter + bar_diameter / 2) >= min(width, depth):
        raise ValueError("Lớp bảo vệ, đai và thép dọc không vừa trong tiết diện.")
    ops.section("Fiber", section_tag, "-GJ", model.E / (2 * 1.2) * torsion_j)
    ops.patch("rect", concrete_tag, 12, 12, -width/2, -depth/2, width/2, depth/2)
    for y, z, area, dia in bars_yz:
        if abs(y) + dia/2 + model.cover + model.tie_diameter > width/2 + 1e-9 or abs(z) + dia/2 + model.cover + model.tie_diameter > depth/2 + 1e-9:
            raise ValueError("Vị trí cốt dọc nằm ngoài lõi tiết diện; giảm lớp bảo vệ/đai hoặc điều chỉnh bố trí.")
        ops.fiber(float(y), float(z), float(area), steel_tag)


def _rectangular_column_bars(model: FrameModel):
    b, h = model.col_b, model.col_h
    d = model.column_bar_diameter
    inset = model.cover + model.tie_diameter + d/2
    y0, z0 = -b/2 + inset, -h/2 + inset
    y1, z1 = b/2 - inset, h/2 - inset
    perimeter = 2*((y1-y0)+(z1-z0))
    count = max(4, int(model.column_bar_count))
    result=[]
    area=np.pi*d*d/4
    for i in range(count):
        s=perimeter*i/count
        if s <= y1-y0: y,z=y0+s,z0
        elif s <= (y1-y0)+(z1-z0): y,z=y1,z0+s-(y1-y0)
        elif s <= 2*(y1-y0)+(z1-z0): y,z=y1-(s-(y1-y0)-(z1-z0)),z1
        else: y,z=y0,z1-(s-2*(y1-y0)-(z1-z0))
        result.append((y,z,area,d))
    return result


def _rectangular_beam_bars(model: FrameModel):
    b,h=model.beam_b,model.beam_h
    d=model.beam_bar_diameter
    inset=model.cover+model.tie_diameter+d/2
    y0,y1=-b/2+inset,b/2-inset
    ztop, zbot=h/2-inset, -h/2+inset
    area=np.pi*d*d/4
    result=[]
    for n,z in ((max(2,int(model.beam_top_bar_count)),ztop),(max(2,int(model.beam_bottom_bar_count)),zbot)):
        ys=np.linspace(y0,y1,n) if n>1 else np.array([(y0+y1)/2])
        result.extend((float(y),float(z),area,d) for y in ys)
    return result


def _build_model(ops, model: FrameModel, nonlinear: bool = False):
    if model.direction.upper() not in {"X", "Y"}:
        raise ValueError("Phương kích động phải là X hoặc Y.")
    if min(model.floors, model.bays_x, model.bays_y) < 1:
        raise ValueError("Số tầng và số nhịp phải dương.")
    if min(model.bay_x, model.bay_y, model.story_height, model.floor_mass,
           model.E, model.col_b, model.col_h, model.beam_b, model.beam_h) <= 0:
        raise ValueError("Kích thước, khối lượng và mô đun đàn hồi phải dương.")

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    cells = _valid_cells(model)
    plan_nodes = set()
    edges = set()
    for ix, iy in cells:
        corners = ((ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1))
        plan_nodes.update(corners)
        for a, b in zip(corners, corners[1:] + corners[:1]):
            edges.add(tuple(sorted((a, b))))
    if not plan_nodes:
        raise ValueError("Mặt bằng không tạo được ô kết cấu.")

    node_tag = lambda floor, ix, iy: 100_000 + floor * 10_000 + ix * 100 + iy
    master_tag = lambda floor: 900_000 + floor
    base_nodes = []
    level_nodes = {}
    for floor in range(model.floors + 1):
        level_nodes[floor] = []
        for ix, iy in sorted(plan_nodes):
            tag = node_tag(floor, ix, iy)
            ops.node(tag, ix * model.bay_x, iy * model.bay_y,
                     floor * model.story_height)
            level_nodes[floor].append(tag)
            if floor == 0:
                ops.fix(tag, 1, 1, 1, 1, 1, 1)
                base_nodes.append(tag)

    # Rectangular gross section properties; torsion uses a Saint-Venant approximation.
    g = model.E / (2.0 * (1.0 + 0.20))
    col_area = model.col_b * model.col_h
    col_iy = model.col_h * model.col_b**3 / 12.0
    col_iz = model.col_b * model.col_h**3 / 12.0
    col_j = _rectangle_torsion(model.col_b, model.col_h)
    beam_area = model.beam_b * model.beam_h
    beam_i = model.beam_b * model.beam_h**3 / 12.0
    beam_j = _rectangle_torsion(model.beam_b, model.beam_h)
    if nonlinear:
        ops.uniaxialMaterial("Concrete01", 1, -abs(model.concrete_fc), -0.002,
                             -0.20*abs(model.concrete_fc), -0.006)
        ops.uniaxialMaterial("Steel02", 2, abs(model.steel_fy), model.steel_E,
                             0.01, 18.0, 0.925, 0.15)
        _add_rc_fiber_section(ops, model, 101, model.col_b, model.col_h,
                              _rectangular_column_bars(model), col_j, 1, 2)
        _add_rc_fiber_section(ops, model, 102, model.beam_b, model.beam_h,
                              _rectangular_beam_bars(model), beam_j, 1, 2)
        ops.beamIntegration("Lobatto", 101, 101, 5)
        ops.beamIntegration("Lobatto", 102, 102, 5)
    ops.geomTransf("Linear", 1, 1.0, 0.0, 0.0)
    ops.geomTransf("Linear", 2, 0.0, 0.0, 1.0)
    element_tag = 1
    column_element_tags = []
    beam_element_tags = []
    for floor in range(model.floors):
        for ix, iy in sorted(plan_nodes):
            if nonlinear:
                ops.element("nonlinearBeamColumn", element_tag,
                            node_tag(floor, ix, iy), node_tag(floor + 1, ix, iy),
                            5, 101, 1)
            else:
                ops.element("elasticBeamColumn", element_tag,
                            node_tag(floor, ix, iy), node_tag(floor + 1, ix, iy),
                            col_area, model.E, g, col_j, col_iy, col_iz, 1)
            column_element_tags.append(element_tag)
            element_tag += 1
    for floor in range(1, model.floors + 1):
        for (ix1, iy1), (ix2, iy2) in sorted(edges):
            if nonlinear:
                ops.element("nonlinearBeamColumn", element_tag,
                            node_tag(floor, ix1, iy1), node_tag(floor, ix2, iy2),
                            5, 102, 2)
            else:
                ops.element("elasticBeamColumn", element_tag,
                            node_tag(floor, ix1, iy1), node_tag(floor, ix2, iy2),
                            beam_area, model.E, g, beam_j,
                            beam_i, model.beam_h * model.beam_b**3 / 12.0, 2)
            beam_element_tags.append(element_tag)
            element_tag += 1

    floor_masters = []
    dir_dof = 1 if model.direction.upper() == "X" else 2
    cell_centers = np.asarray([[(ix + 0.5) * model.bay_x, (iy + 0.5) * model.bay_y]
                               for ix, iy in cells], dtype=float)
    center_x, center_y = np.mean(cell_centers, axis=0)
    polar_mass = model.floor_mass * (
        float(np.mean(np.sum((cell_centers - [center_x, center_y])**2, axis=1)))
        + (model.bay_x**2 + model.bay_y**2) / 12.0
    )
    eps_mass = max(model.floor_mass, 1.0) * 1e-9 / max(len(plan_nodes), 1)
    for floor in range(1, model.floors + 1):
        master = master_tag(floor)
        ops.node(master, center_x, center_y, floor * model.story_height)
        ops.fix(master, 0, 0, 1, 1, 1, 0)
        ops.mass(master, model.floor_mass, model.floor_mass, eps_mass,
                 eps_mass, eps_mass, max(polar_mass, eps_mass))
        ops.rigidDiaphragm(3, master, *level_nodes[floor])
        floor_masters.append(master)
        # Tiny vertical/rotational mass regularizes the eigen problem without
        # materially changing lateral inertia; lateral mass stays at diaphragm.
        for tag in level_nodes[floor]:
            ops.mass(tag, eps_mass, eps_mass, eps_mass, eps_mass, eps_mass, eps_mass)
    return {"base_nodes": base_nodes, "floor_masters": floor_masters,
            "direction_dof": dir_dof, "plan_nodes": plan_nodes,
            "level_nodes": level_nodes, "floor_mass": model.floor_mass,
            "column_element_tags": column_element_tags,
            "beam_element_tags": beam_element_tags}


def _eigen_data(ops, model: FrameModel, built):
    n_modes = max(1, min(3 * int(model.floors), 60))
    last_error = None
    while n_modes >= 1:
        try:
            eigenvalues = np.asarray(ops.eigen(n_modes), dtype=float).reshape(-1)
            if np.all(np.isfinite(eigenvalues)) and np.all(eigenvalues > 0):
                break
        except Exception as exc:  # OpenSees ARPACK can fail at a requested mode count.
            last_error = exc
        n_modes -= 1
    else:
        raise RuntimeError(f"OpenSees không tìm được mode dương: {last_error or 'eigen thất bại'}")
    omega_all = np.sqrt(eigenvalues)
    periods_all = 2.0 * np.pi / omega_all
    dof = built["direction_dof"]
    mass = float(built["floor_mass"])
    phi_all = np.array([[ops.nodeEigenvector(tag, mode + 1, dof)
                         for mode in range(len(eigenvalues))]
                        for tag in built["floor_masters"]], dtype=float)
    modal_mass = np.sum(mass * phi_all**2, axis=0)
    modal_load = np.sum(mass * phi_all, axis=0)
    participation = np.divide(modal_load, modal_mass,
                              out=np.zeros_like(modal_load), where=modal_mass > 0)
    effective_mass = participation**2 * modal_mass
    ratio = effective_mass / (model.floors * mass)
    order = np.arange(len(periods_all))
    periods = periods_all[order]
    phi = phi_all[:, order]
    omega = omega_all[order]
    gamma = participation[order]
    ratios = ratio[order]
    dominant = int(np.argmax(ratios))
    if ratios[dominant] <= 1e-12:
        raise RuntimeError("Không có mode tham gia theo phương kích động đã chọn.")
    return {"periods": periods, "omega": omega, "modes": phi,
            "participation": gamma, "effective_mass_ratio": ratios,
            "dominant_mode": dominant}


def estimate_periods(model: FrameModel):
    """Run the same OpenSees eigen model used by the seismic analyses."""
    ops = _ops()
    built = _build_model(ops, model)
    return _eigen_data(ops, model, built)["periods"].copy()


def _setup_analysis(ops):
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-9, 30, 0)


def run_elf(model: FrameModel, spectrum_periods, spectrum_accelerations,
            lambda_factor: float = 1.0):
    ops = _ops()
    built = _build_model(ops, model)
    modes = _eigen_data(ops, model, built)
    mode = modes["dominant_mode"]
    t1 = float(modes["periods"][mode])
    sa = float(np.interp(t1, spectrum_periods, spectrum_accelerations))
    z = np.arange(1, model.floors + 1, dtype=float) * model.story_height
    masses = np.full(model.floors, model.floor_mass)
    base_shear_target = sa * np.sum(masses) * float(lambda_factor)
    floor_forces = base_shear_target * masses * z / np.sum(masses * z)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    dof = built["direction_dof"]
    for tag, force in zip(built["floor_masters"], floor_forces):
        load = [0.0] * 6
        load[dof - 1] = float(force)
        ops.load(tag, *load)
    _setup_analysis(ops)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees ELF static analysis did not converge.")
    floor_u = np.asarray([ops.nodeDisp(tag, dof) for tag in built["floor_masters"]])
    drift = np.diff(np.r_[0.0, floor_u])
    story_shear = np.cumsum(floor_forces[::-1])[::-1]
    ops.reactions()
    base_shear = abs(sum(float(ops.nodeReaction(tag, dof)) for tag in built["base_nodes"]))
    return {"analysis_type": "elf", "method_name": "OpenSees 3D · Equivalent Lateral Force (ELF)",
            "periods": modes["periods"], "dominant_period": t1,
            "base_shear": base_shear, "target_base_shear": base_shear_target,
            "floor_forces": floor_forces, "story_shear": story_shear,
            "floor_displacement": floor_u, "story_drift": drift,
            "story_drift_ratio": drift / model.story_height,
            "effective_mass_ratio": modes["effective_mass_ratio"],
            "spectral_acceleration": sa, "lambda_factor": float(lambda_factor)}


def _cqc_matrix(omega, damping_ratio):
    count = len(omega)
    rho = np.eye(count)
    xi = float(damping_ratio)
    for i in range(count):
        for j in range(i + 1, count):
            r = min(omega[i], omega[j]) / max(omega[i], omega[j])
            den = (1.0 - r * r) ** 2 + 4.0 * xi * xi * r * (1.0 + r) ** 2
            rho[i, j] = rho[j, i] = (8.0 * xi * xi * (1.0 + r) * r**1.5 / den
                                      if den else 1.0)
    return rho


def _combine(responses, rho, combination):
    # responses has shape (n_modes, n_response_quantities).
    if combination == "SRSS":
        return np.sqrt(np.sum(responses**2, axis=0))
    return np.sqrt(np.maximum(np.einsum("ij,ik,jk->k", rho, responses, responses), 0.0))


def run_mrsa(model: FrameModel, spectrum_periods, spectrum_accelerations,
             damping_ratio: float = 0.05, combination: str = "CQC"):
    ops = _ops()
    built = _build_model(ops, model)
    modes = _eigen_data(ops, model, built)
    ratios = modes["effective_mass_ratio"]
    active = np.flatnonzero(ratios > 1e-10)
    if active.size == 0:
        raise RuntimeError("Không tìm được mode có khối lượng tham gia theo phương đã chọn.")
    cumulative = np.cumsum(ratios[active])
    count = min(len(active), int(np.searchsorted(cumulative, 0.90)) + 1)
    selected = active[:count]
    periods = modes["periods"][selected]
    if np.min(periods) < spectrum_periods[0] or np.max(periods) > spectrum_periods[-1]:
        raise ValueError(f"Phổ T={spectrum_periods[0]:.3g}–{spectrum_periods[-1]:.3g}s "
                         f"phải bao phủ các mode giữ lại T={np.min(periods):.3g}–{np.max(periods):.3g}s.")
    sa = np.interp(periods, spectrum_periods, spectrum_accelerations)
    omega = modes["omega"][selected]
    phi = modes["modes"][:, selected]
    gamma = modes["participation"][selected]
    modal_u = phi * (gamma * sa / omega**2)[None, :]
    modal_force = model.floor_mass * phi * (gamma * sa)[None, :]
    modal_drift = np.diff(np.vstack([np.zeros((1, count)), modal_u]), axis=0)
    modal_shear = np.cumsum(modal_force[::-1, :], axis=0)[::-1, :]
    combo = str(combination).upper()
    if combo not in {"SRSS", "CQC"}:
        raise ValueError("Chọn tổ hợp SRSS hoặc CQC.")
    rho = _cqc_matrix(omega, damping_ratio)
    floor_u = _combine(modal_u.T, rho, combo)
    drift = _combine(modal_drift.T, rho, combo)
    floor_force = _combine(modal_force.T, rho, combo)
    story_shear = _combine(modal_shear.T, rho, combo)
    mode_table = [{"Mode": int(mode + 1), "T (s)": float(modes["periods"][mode]),
                   "Sa(T) (m/s²)": float(sa[j]), "Sd(T) (m)": float(sa[j] / omega[j]**2),
                   "Khối lượng hữu hiệu (%)": float(ratios[mode] * 100.0)}
                  for j, mode in enumerate(selected)]
    return {"analysis_type": "mrsa", "method_name": "OpenSees 3D · Modal Response Spectrum Analysis (MRSA)",
            "periods": periods, "dominant_period": float(modes["periods"][modes["dominant_mode"]]),
            "mode_sa": sa, "mode_sd": sa / omega**2,
            "effective_mass_ratio": ratios[selected], "effective_mass_total": float(np.sum(ratios[selected])),
            "modes_retained": int(count), "combination": combo,
            "base_shear": float(story_shear[0]), "floor_displacement": floor_u,
            "floor_forces": floor_force, "story_shear": story_shear,
            "story_drift": drift, "story_drift_ratio": drift / model.story_height,
            "spectrum_periods": np.asarray(spectrum_periods),
            "spectrum_accelerations": np.asarray(spectrum_accelerations),
            "mode_table": mode_table}


def _apply_gravity_preload(ops, model: FrameModel, built):
    """Apply floor-mass weight as gravity before nonlinear THA (no frame self-weight)."""
    ops.timeSeries("Linear", 9001)
    ops.pattern("Plain", 9001, 9001)
    for floor in range(1, model.floors + 1):
        tags=built["level_nodes"][floor]
        nodal_weight=model.floor_mass*9.80665/max(len(tags),1)
        for tag in tags:
            ops.load(tag,0.0,0.0,-nodal_weight,0.0,0.0,0.0)
    _setup_analysis(ops)
    ops.algorithm("Newton")
    ops.integrator("LoadControl",1.0)
    ops.analysis("Static")
    if ops.analyze(1)!=0:
        raise RuntimeError("Phân tích trọng lực trước THA phi tuyến không hội tụ; kiểm tra tiết diện và khối lượng sàn.")
    ops.loadConst("-time",0.0)
    ops.wipeAnalysis()


def run_tha(model: FrameModel, ground_acceleration, dt: float, damping_ratio: float = 0.05,
            nonlinear: bool = False):
    ops = _ops()
    ground = np.asarray(ground_acceleration, dtype=float).reshape(-1)
    if len(ground) < 2 or not np.isfinite(ground).all() or dt <= 0:
        raise ValueError("THA cần tối thiểu 2 mẫu gia tốc hữu hạn và dt dương.")
    built = _build_model(ops, model, nonlinear=nonlinear)
    if nonlinear:
        _apply_gravity_preload(ops,model,built)
    modes = _eigen_data(ops, model, built)
    active = np.flatnonzero(modes["effective_mass_ratio"] > 1e-10)
    if not len(active):
        raise RuntimeError("Không tìm được mode để hiệu chỉnh cản Rayleigh.")
    w1 = float(modes["omega"][active[0]])
    w2 = float(modes["omega"][active[1]]) if len(active) > 1 else w1
    alpha_m = 2.0 * damping_ratio * w1 * w2 / (w1 + w2)
    beta_k = 2.0 * damping_ratio / (w1 + w2)
    ops.rayleigh(alpha_m, 0.0, beta_k, 0.0)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="ascii") as stream:
        motion_path = Path(stream.name)
        np.savetxt(stream, ground, fmt="%.12g")
    try:
        ops.timeSeries("Path", 1, "-dt", float(dt), "-filePath", str(motion_path), "-factor", 1.0)
        ops.pattern("UniformExcitation", 1, built["direction_dof"], "-accel", 1)
        _setup_analysis(ops)
        ops.test("NormDispIncr", 1.0e-8 if nonlinear else 1.0e-9, 50 if nonlinear else 30, 0)
        ops.algorithm("Newton" if nonlinear else "Linear")
        ops.integrator("Newmark", 0.5, 0.25)
        ops.analysis("Transient")
        disp = np.zeros((len(ground), model.floors), dtype=float)
        abs_accel = np.zeros_like(disp)
        base_shear = np.zeros(len(ground), dtype=float)
        column_force = np.zeros((len(ground), len(built["column_element_tags"])), dtype=float)
        beam_force = np.zeros((len(ground), len(built["beam_element_tags"])), dtype=float)
        for step, ag in enumerate(ground):
            status=ops.analyze(1,float(dt))
            if status!=0 and nonlinear:
                # Retry the same uncommitted step with standard nonlinear
                # equilibrium algorithms before reporting nonconvergence.
                for algorithm in (("NewtonLineSearch",), ("ModifiedNewton", "-initial"), ("KrylovNewton",)):
                    ops.algorithm(*algorithm)
                    if ops.analyze(1,float(dt))==0:
                        status=0
                        break
                ops.algorithm("Newton")
            if status!=0:
                raise RuntimeError(f"OpenSees THA không hội tụ tại bước {step + 1}/{len(ground)}; thử giảm Δt hoặc kiểm tra dữ liệu và mô hình vật liệu.")
            for floor, tag in enumerate(built["floor_masters"]):
                disp[step, floor] = ops.nodeDisp(tag, built["direction_dof"])
                rel_acc = ops.nodeAccel(tag, built["direction_dof"])
                abs_accel[step, floor] = float(rel_acc) + float(ag)
            ops.reactions()
            # Support reactions oppose the frame's restoring shear. Reverse
            # their sign to report restoring base shear, so positive drift
            # and positive resistance use the conventional positive slope.
            base_shear[step] = -sum(
                float(ops.nodeReaction(tag, built["direction_dof"]))
                for tag in built["base_nodes"]
            )
            for j, tag in enumerate(built["column_element_tags"]):
                try:
                    values = np.asarray(ops.eleResponse(tag, "basicForce"), dtype=float).reshape(-1)
                    if values.size:
                        column_force[step, j] = float(np.max(np.abs(values)))
                except Exception:
                    pass
            for j, tag in enumerate(built["beam_element_tags"]):
                try:
                    values = np.asarray(ops.eleResponse(tag, "basicForce"), dtype=float).reshape(-1)
                    if values.size:
                        beam_force[step, j] = float(np.max(np.abs(values)))
                except Exception:
                    pass
    finally:
        motion_path.unlink(missing_ok=True)
    drift = np.diff(np.column_stack([np.zeros(len(ground)), disp]), axis=1)
    roof_displacement = disp[:, -1]
    story_shear = np.cumsum((model.floor_mass * abs_accel)[:, ::-1], axis=1)[:, ::-1]
    return {"analysis_type": "time_history", "nonlinear": bool(nonlinear),
            "method_name": "OpenSees 3D · Time-History Analysis (THA, phi tuyến fiber RC)" if nonlinear else "OpenSees 3D · Time-History Analysis (THA, đàn hồi)",
            "periods": modes["periods"], "time": np.arange(len(ground)) * float(dt),
            "ground_acceleration": ground, "displacement": disp,
            "floor_displacement": np.max(np.abs(disp), axis=0),
            "roof_displacement": roof_displacement,
            "story_drift": drift, "story_drift_ratio": drift / model.story_height,
            "max_story_drift_ratio": np.max(np.abs(drift), axis=1) / model.story_height,
            "floor_forces": np.zeros(model.floors),
            "story_shear": story_shear, "base_shear_history": base_shear,
            "base_shear": float(np.max(np.abs(base_shear))),
            "absolute_acceleration": abs_accel,
            "base_shear_history": base_shear,
            "column_basic_force_history": column_force,
            "beam_basic_force_history": beam_force,
            "rayleigh_alpha_m": alpha_m, "rayleigh_beta_k": beta_k,
            "material_assumptions": ("Concrete01 + Steel02; section Fiber; nonlinearBeamColumn Lobatto 5 integration points" if nonlinear else "elasticBeamColumn")}

