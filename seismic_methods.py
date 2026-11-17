"""Earthquake procedures for reduced-order shear-building models (SI units)."""
from __future__ import annotations

import numpy as np

TYPE1_GROUND_PARAMETERS = {
    "A": (1.00, 0.15, 0.40, 2.00),
    "B": (1.20, 0.15, 0.50, 2.00),
    "C": (1.15, 0.20, 0.60, 2.00),
    "D": (1.35, 0.20, 0.80, 2.00),
    "E": (1.40, 0.15, 0.50, 2.00),
}


def tcvn9386_type1_elastic_spectrum(periods, reference_ag, ground_type="B",
                                    importance_factor=1.0, damping_ratio=0.05):
    """Type 1 elastic spectrum shape used with TCVN 9386 / EN 1998.

    ``reference_ag`` is the reference rock PGA in m/s²; ground-type factors,
    importance and viscous damping are applied explicitly. This returns the
    elastic spectrum (q=1), not a reduced design spectrum.
    """
    T = np.asarray(periods, dtype=float)
    agR, gamma_i, xi = float(reference_ag), float(importance_factor), float(damping_ratio)
    soil = str(ground_type).upper()
    if soil not in TYPE1_GROUND_PARAMETERS:
        raise ValueError("Phổ mẫu Type 1 hiện hỗ trợ nền A, B, C, D hoặc E.")
    if T.size == 0 or np.any(~np.isfinite(T)) or np.any(T < 0):
        raise ValueError("Chu kỳ phổ phải là các số hữu hạn không âm.")
    if not np.isfinite([agR, gamma_i, xi]).all() or agR < 0 or gamma_i <= 0 or not 0 <= xi < 1:
        raise ValueError("agR không âm, hệ số tầm quan trọng dương và tỷ số cản thuộc [0,1).")
    S, TB, TC, TD = TYPE1_GROUND_PARAMETERS[soil]
    eta = max(0.55, np.sqrt(10.0 / (5.0 + 100.0*xi)))
    ag = gamma_i * agR
    ratio = np.maximum(T, 1e-12)
    shape = np.where(
        T <= TB, 1.0 + (T / TB) * (2.5*eta - 1.0),
        np.where(T <= TC, 2.5*eta,
                 np.where(T <= TD, 2.5*eta*TC/ratio,
                          2.5*eta*TC*TD/(ratio**2)))
    )
    return ag * S * shape


def modal_properties(masses, story_stiffnesses):
    m = np.asarray(masses, dtype=float).reshape(-1)
    ks = np.asarray(story_stiffnesses, dtype=float).reshape(-1)
    if len(m) < 1 or len(ks) != len(m) or np.any(~np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("Cần khối lượng dương và độ cứng dương cho mỗi tầng.")
    if np.any(~np.isfinite(ks)) or np.any(ks <= 0):
        raise ValueError("Độ cứng tầng phải là số dương hữu hạn.")
    n = len(m)
    M = np.diag(m)
    K = np.zeros((n, n))
    for i, ki in enumerate(ks):
        K[i, i] += ki
        if i:
            K[i - 1, i - 1] += ki
            K[i - 1, i] -= ki
            K[i, i - 1] -= ki
    inv_sqrt_m = 1.0 / np.sqrt(m)
    eigenvalues, vectors = np.linalg.eigh(inv_sqrt_m[:, None] * K * inv_sqrt_m[None, :])
    keep = eigenvalues > max(float(np.max(eigenvalues)), 1.0) * 1e-12
    omega = np.sqrt(eigenvalues[keep])
    phi = inv_sqrt_m[:, None] * vectors[:, keep]
    for j in range(phi.shape[1]):
        if phi[-1, j] < 0:
            phi[:, j] *= -1
    modal_mass = np.einsum("in,i,in->n", phi, m, phi)
    gamma = np.einsum("in,i->n", phi, m) / modal_mass
    effective_mass = gamma**2 * modal_mass
    return {"mass_matrix": M, "stiffness_matrix": K, "omega": omega,
            "periods": 2*np.pi/omega, "modes": phi,
            "participation_factor": gamma, "modal_mass": modal_mass,
            "effective_mass": effective_mass,
            "effective_mass_ratio": effective_mass/np.sum(m)}


def equivalent_lateral_force(masses, story_stiffnesses, floor_heights,
                             spectral_acceleration, lambda_factor=1.0):
    """Equivalent-static force distribution Fi ∝ mi zi from entered Sa(T1)."""
    m = np.asarray(masses, dtype=float).reshape(-1)
    z = np.asarray(floor_heights, dtype=float).reshape(-1)
    sa, lam = float(spectral_acceleration), float(lambda_factor)
    if len(z) != len(m) or len(m) == 0 or np.any(~np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("Cần cao độ sàn dương cho từng khối lượng tầng.")
    if not np.isfinite([sa, lam]).all() or sa < 0 or lam <= 0:
        raise ValueError("Sa(T1) phải không âm và hệ số λ phải dương.")
    props = modal_properties(m, story_stiffnesses)
    base_shear = float(sa * np.sum(m) * lam)
    floor_forces = base_shear * m*z / np.sum(m*z)
    story_shear = np.cumsum(floor_forces[::-1])[::-1]
    displacement = np.linalg.solve(props["stiffness_matrix"], floor_forces)
    drift = np.diff(np.r_[0.0, displacement])
    return {"periods": props["periods"], "base_shear": base_shear,
            "floor_forces": floor_forces, "story_shear": story_shear,
            "displacement": displacement, "story_drift": drift,
            "effective_mass_ratio": props["effective_mass_ratio"],
            "spectral_acceleration": sa, "lambda_factor": lam}


def modal_response_spectrum(masses, story_stiffnesses, floor_heights,
                            spectrum_periods, spectrum_accelerations,
                            damping_ratio=0.05, combination="CQC"):
    """Elastic modal response-spectrum analysis with SRSS or equal-damping CQC."""
    m = np.asarray(masses, dtype=float).reshape(-1)
    z = np.asarray(floor_heights, dtype=float).reshape(-1)
    st = np.asarray(spectrum_periods, dtype=float).reshape(-1)
    sa_spec = np.asarray(spectrum_accelerations, dtype=float).reshape(-1)
    xi = float(damping_ratio)
    if len(z) != len(m) or np.any(~np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("Cần cao độ sàn dương cho từng tầng.")
    if (len(st) < 2 or len(st) != len(sa_spec) or np.any(~np.isfinite(st))
            or np.any(~np.isfinite(sa_spec)) or np.any(st <= 0)
            or np.any(sa_spec < 0) or np.any(np.diff(st) <= 0)):
        raise ValueError("Phổ cần ít nhất hai hàng T tăng dần, Sa không âm và hữu hạn.")
    if not 0 <= xi < 1:
        raise ValueError("Tỷ số cản cần thuộc [0, 1).")
    combination = str(combination).upper()
    if combination not in {"SRSS", "CQC"}:
        raise ValueError("Chỉ hỗ trợ tổ hợp SRSS hoặc CQC.")
    props = modal_properties(m, story_stiffnesses)
    cumulative_mass = np.cumsum(props["effective_mass_ratio"])
    n_keep = min(len(cumulative_mass), int(np.searchsorted(cumulative_mass, 0.90)) + 1)
    periods = props["periods"][:n_keep]
    if periods.min() < st[0] or periods.max() > st[-1]:
        raise ValueError(f"Phổ T={st[0]:.3g}–{st[-1]:.3g}s phải bao phủ mọi mode T={periods.min():.3g}–{periods.max():.3g}s; không ngoại suy.")
    M = props["mass_matrix"]
    phi = props["modes"][:, :n_keep]
    gamma = props["participation_factor"][:n_keep]
    omega = props["omega"][:n_keep]
    mode_sa = np.interp(periods, st, sa_spec)
    mode_sd = mode_sa / omega**2
    modal_u = phi * (gamma*mode_sd)[None, :]
    modal_floor_force = M @ phi * (gamma*mode_sa)[None, :]
    modal_story_shear = np.cumsum(modal_floor_force[::-1, :], axis=0)[::-1, :]
    modal_story_drift = np.diff(np.vstack([np.zeros((1, len(periods))), modal_u]), axis=0)
    rho = np.eye(len(periods))
    if combination == "CQC":
        for i in range(len(periods)):
            for j in range(i+1, len(periods)):
                r = min(omega[i], omega[j]) / max(omega[i], omega[j])
                den = (1-r**2)**2 + 4*xi**2*r*(1+r)**2
                rho[i, j] = rho[j, i] = 8*xi**2*(1+r)*r**1.5/den if den else 1.0

    def combine(response_by_mode):
        if combination == "SRSS":
            return np.sqrt(np.sum(response_by_mode**2, axis=1))
        # response_by_mode is shaped (n_modes, n_responses), so contract the
        # two modal axes and retain one result for each floor/response item.
        variance = np.einsum("ij,ik,jk->k", rho, response_by_mode, response_by_mode)
        return np.sqrt(np.maximum(variance, 0.0))

    floor_u = combine(modal_u.T)
    floor_force = combine(modal_floor_force.T)
    story_shear = combine(modal_story_shear.T)
    story_drift = combine(modal_story_drift.T)
    return {"periods": periods, "mode_sa": mode_sa, "mode_sd": mode_sd,
            "effective_mass_ratio": props["effective_mass_ratio"][:n_keep],
            "modes_retained": n_keep,
            "floor_displacement": floor_u, "floor_forces": floor_force,
            "story_shear": story_shear, "story_drift": story_drift,
            "base_shear": float(story_shear[0]), "combination": combination,
            "damping_ratio": xi, "spectrum_periods": st,
            "spectrum_accelerations": sa_spec}
