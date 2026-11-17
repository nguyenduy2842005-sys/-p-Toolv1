"""Linear SDOF dynamics utilities adapted from the supplied OpenSees study.

The routines use SI units and Newmark average-acceleration integration. They
provide a dependency-free counterpart to the study's free-vibration and
response-spectrum workflows; nonlinear OpenSees models are not reproduced.
"""
from __future__ import annotations

import numpy as np


def _validate_system(mass: float, stiffness: float, damping_ratio: float) -> tuple[float, float, float]:
    m, k, zeta = float(mass), float(stiffness), float(damping_ratio)
    if not np.isfinite([m, k, zeta]).all() or m <= 0 or k <= 0 or zeta < 0:
        raise ValueError("Khối lượng và độ cứng phải dương; tỷ số cản không được âm.")
    return m, k, zeta


def free_vibration(mass: float, stiffness: float, damping_ratio: float,
                   duration: float, dt: float, u0: float = 0.0,
                   v0: float = 0.0) -> dict[str, np.ndarray | float]:
    """Exact underdamped/critical/overdamped free response for an SDOF."""
    m, k, zeta = _validate_system(mass, stiffness, damping_ratio)
    if duration <= 0 or dt <= 0 or not np.isfinite([duration, dt, u0, v0]).all():
        raise ValueError("Thời lượng, bước thời gian và điều kiện đầu phải hữu hạn; thời lượng/bước > 0.")
    n = max(2, int(np.ceil(duration / dt)) + 1)
    t = np.linspace(0.0, float(duration), n)
    wn = np.sqrt(k / m)
    wd2 = wn**2 * (1.0 - zeta**2)
    if wd2 > 1e-12:
        wd = np.sqrt(wd2)
        b = (v0 + zeta * wn * u0) / wd
        u = np.exp(-zeta * wn * t) * (u0 * np.cos(wd * t) + b * np.sin(wd * t))
    elif abs(zeta - 1.0) <= 1e-6:
        b = v0 + wn * u0
        u = (u0 + b * t) * np.exp(-wn * t)
    else:
        root = np.sqrt(zeta**2 - 1.0)
        r1, r2 = -wn * (zeta - root), -wn * (zeta + root)
        c1 = (v0 - r2 * u0) / (r1 - r2)
        c2 = u0 - c1
        u = c1 * np.exp(r1 * t) + c2 * np.exp(r2 * t)
    v = np.gradient(u, t)
    a = np.gradient(v, t)
    return {"time": t, "displacement": u, "velocity": v, "acceleration": a,
            "omega_n": float(wn), "period": float(2 * np.pi / wn),
            "damping": float(2 * zeta * wn * m)}


def newmark_response(mass: float, stiffness: float, damping_ratio: float,
                     ground_acceleration: np.ndarray, dt: float) -> dict[str, np.ndarray | float]:
    """Relative response to base acceleration using Newmark beta=1/4, gamma=1/2."""
    m, k, zeta = _validate_system(mass, stiffness, damping_ratio)
    ag = np.asarray(ground_acceleration, dtype=float).reshape(-1)
    if len(ag) < 2 or not np.isfinite(ag).all() or dt <= 0 or not np.isfinite(dt):
        raise ValueError("Cần ít nhất 2 giá trị gia tốc nền hữu hạn và dt > 0.")
    c = 2.0 * zeta * np.sqrt(k * m)
    beta, gamma = 0.25, 0.5
    u = np.zeros(len(ag)); v = np.zeros(len(ag)); a = np.zeros(len(ag))
    a[0] = -ag[0]
    a0 = 1.0 / (beta * dt**2); a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt); a3 = 1.0 / (2 * beta) - 1
    a4 = gamma / beta - 1; a5 = dt * (gamma / (2 * beta) - 1)
    keff = k + a0 * m + a1 * c
    for i in range(len(ag) - 1):
        pnext = -m * ag[i + 1]
        peff = pnext + m * (a0 * u[i] + a2 * v[i] + a3 * a[i]) + c * (a1 * u[i] + a4 * v[i] + a5 * a[i])
        u[i + 1] = peff / keff
        a[i + 1] = a0 * (u[i + 1] - u[i]) - a2 * v[i] - a3 * a[i]
        v[i + 1] = v[i] + dt * ((1 - gamma) * a[i] + gamma * a[i + 1])
    return {"time": np.arange(len(ag)) * dt, "ground_acceleration": ag,
            "displacement": u, "velocity": v, "acceleration": a,
            "absolute_acceleration": a + ag, "base_shear": -m * (a + ag),
            "omega_n": float(np.sqrt(k / m)), "period": float(2 * np.pi * np.sqrt(m / k)),
            "damping": float(c)}


def response_spectrum(periods: np.ndarray, ground_acceleration: np.ndarray,
                      dt: float, damping_ratio: float = 0.05) -> dict[str, np.ndarray]:
    """Elastic pseudo-acceleration/displacement spectrum via Newmark analysis."""
    ts = np.asarray(periods, dtype=float).reshape(-1)
    if not len(ts) or np.any(~np.isfinite(ts)) or np.any(ts <= 0):
        raise ValueError("Các chu kỳ phải là số dương hữu hạn.")
    ag = np.asarray(ground_acceleration, dtype=float).reshape(-1)
    sd = np.empty(len(ts)); sa = np.empty(len(ts))
    for i, period in enumerate(ts):
        result = newmark_response(1.0, (2 * np.pi / period)**2, damping_ratio, ag, dt)
        sd[i] = np.max(np.abs(result["displacement"]))
        sa[i] = (2 * np.pi / period)**2 * sd[i]
    return {"period": ts, "Sd": sd, "Sa": sa, "PSa_g": sa / 9.80665}


def mdof_response(masses: np.ndarray, story_stiffnesses: np.ndarray,
                  damping_ratio: float, ground_acceleration: np.ndarray,
                  dt: float) -> dict[str, np.ndarray]:
    """Shear-building response with one lateral DOF per floor.

    ``story_stiffnesses[i]`` is the lateral stiffness of story i (ground to
    floor 1, then floor to floor). Proportional damping is C=aM+bK fitted to
    the requested ratio at the first two modal frequencies.
    """
    m = np.asarray(masses, dtype=float).reshape(-1)
    ks = np.asarray(story_stiffnesses, dtype=float).reshape(-1)
    ag = np.asarray(ground_acceleration, dtype=float).reshape(-1)
    if len(m) < 1 or len(ks) != len(m) or np.any(~np.isfinite(m)) or np.any(m <= 0):
        raise ValueError("Cần khối lượng dương và độ cứng cho mỗi tầng.")
    if np.any(~np.isfinite(ks)) or np.any(ks <= 0):
        raise ValueError("Độ cứng tầng phải là số dương hữu hạn.")
    if len(ag) < 2 or np.any(~np.isfinite(ag)) or dt <= 0 or not np.isfinite(dt):
        raise ValueError("Cần ít nhất 2 mẫu gia tốc nền hữu hạn và dt > 0.")
    if damping_ratio < 0 or not np.isfinite(damping_ratio):
        raise ValueError("Tỷ số cản không được âm.")
    n = len(m)
    M = np.diag(m)
    K = np.zeros((n, n))
    for i, ki in enumerate(ks):
        K[i, i] += ki
        if i > 0:
            K[i - 1, i - 1] += ki
            K[i, i - 1] -= ki
            K[i - 1, i] -= ki
    eig = np.linalg.eigvals(np.linalg.solve(M, K)).real
    omega = np.sqrt(np.sort(eig[eig > 0]))
    if len(omega) >= 2:
        A = np.array([[1 / (2 * omega[0]), omega[0] / 2],
                      [1 / (2 * omega[1]), omega[1] / 2]])
        alpha, beta_r = np.linalg.solve(A, np.full(2, damping_ratio))
    else:
        alpha, beta_r = 2 * damping_ratio * omega[0], 0.0
    C = alpha * M + beta_r * K
    beta, gamma = 0.25, 0.5
    nstep = len(ag)
    U = np.zeros((nstep, n)); V = np.zeros_like(U); Acc = np.zeros_like(U)
    influence = np.ones(n)
    Acc[0] = np.linalg.solve(M, -M @ influence * ag[0])
    a0 = 1 / (beta * dt**2); a1 = gamma / (beta * dt)
    a2 = 1 / (beta * dt); a3 = 1 / (2 * beta) - 1
    a4 = gamma / beta - 1; a5 = dt * (gamma / (2 * beta) - 1)
    Keff = K + a0 * M + a1 * C
    for j in range(nstep - 1):
        pnext = -M @ influence * ag[j + 1]
        peff = pnext + M @ (a0 * U[j] + a2 * V[j] + a3 * Acc[j]) + C @ (a1 * U[j] + a4 * V[j] + a5 * Acc[j])
        U[j + 1] = np.linalg.solve(Keff, peff)
        Acc[j + 1] = a0 * (U[j + 1] - U[j]) - a2 * V[j] - a3 * Acc[j]
        V[j + 1] = V[j] + dt * ((1 - gamma) * Acc[j] + gamma * Acc[j + 1])
    return {"time": np.arange(nstep) * dt, "ground_acceleration": ag,
            "displacement": U, "velocity": V, "acceleration": Acc,
            "absolute_acceleration": Acc + ag[:, None],
            "periods": 2 * np.pi / omega, "damping_matrix": C}


def inelastic_mdof_response(masses: np.ndarray, story_stiffnesses: np.ndarray,
                            damping_ratio: float, ground_acceleration: np.ndarray,
                            dt: float, yield_drifts: np.ndarray,
                            hardening_ratio: float = 0.02,
                            failure_drifts: np.ndarray | None = None,
                            yield_forces: np.ndarray | None = None) -> dict[str, np.ndarray | float | None]:
    """Bilinear kinematic-hysteretic shear building, Newmark + Newton iteration."""
    m = np.asarray(masses, dtype=float).reshape(-1)
    ks = np.asarray(story_stiffnesses, dtype=float).reshape(-1)
    dy = np.asarray(yield_drifts, dtype=float).reshape(-1)
    dfail = np.asarray(failure_drifts, dtype=float).reshape(-1) if failure_drifts is not None else np.full_like(dy, np.inf)
    ag = np.asarray(ground_acceleration, dtype=float).reshape(-1)
    n = len(m)
    if n < 1 or len(ks) != n or len(dy) != n or len(dfail) != n or np.any(m <= 0) or np.any(ks <= 0) or np.any(dy <= 0) or np.any(dfail <= dy):
        raise ValueError("Cần m, k tầng và chuyển vị chảy dương cho mỗi tầng.")
    if len(ag) < 2 or np.any(~np.isfinite(ag)) or dt <= 0:
        raise ValueError("Gia tốc nền cần ít nhất hai mẫu hữu hạn và dt > 0.")
    if not 0 <= hardening_ratio < 1:
        raise ValueError("Tỷ lệ hóa bền cần nằm trong [0,1).")
    B = np.zeros((n, n))
    for i in range(n):
        B[i, i] = 1.0
        if i: B[i, i-1] = -1.0
    M = np.diag(m)
    K = B.T @ np.diag(ks) @ B
    eig = np.sqrt(np.sort(np.linalg.eigvals(np.linalg.solve(M, K)).real))
    omega = eig[eig > 1e-8]
    if len(omega) > 1:
        A = np.array([[1/(2*omega[0]), omega[0]/2],[1/(2*omega[1]),omega[1]/2]])
        alpha, beta_r = np.linalg.solve(A, [damping_ratio,damping_ratio])
    else: alpha,beta_r=2*damping_ratio*omega[0],0.
    C=alpha*M+beta_r*K
    fy=ks*dy if yield_forces is None else np.asarray(yield_forces,dtype=float).reshape(-1)
    if len(fy)!=n or np.any(~np.isfinite(fy)) or np.any(fy<=0):
        raise ValueError("Sức kháng chảy phải dương cho mỗi tầng.")
    dy=fy/ks
    H=hardening_ratio*ks/max(1-hardening_ratio,1e-12)
    beta,gamma=.25,.5; a0=1/(beta*dt**2); a1=gamma/(beta*dt)
    U=np.zeros((len(ag),n)); V=np.zeros_like(U); Acc=np.zeros_like(U)
    drifts=np.zeros((len(ag),n)); shear=np.zeros_like(drifts); plastic=np.zeros_like(drifts)
    Acc[0]=np.linalg.solve(M,-M@np.ones(n)*ag[0])
    p=np.zeros(n); q=np.zeros(n)
    failure_time=None
    for step in range(len(ag)-1):
        un=U[step].copy(); vn=V[step].copy(); an=Acc[step].copy()
        trial_u=un.copy(); p_trial=p.copy(); q_trial=q.copy()
        for _ in range(30):
            aa=a0*(trial_u-un)-1/(beta*dt)*vn-(1/(2*beta)-1)*an
            vv=vn+dt*((1-gamma)*an+gamma*aa)
            d=B@trial_u; dp=B@p
            elastic=ks*(d-dp); xi=elastic-q
            ftrial=np.abs(xi)-fy
            yielded=ftrial>1e-8
            dgamma=np.zeros(n); dgamma[yielded]=ftrial[yielded]/(ks[yielded]+H[yielded])
            sgn=np.sign(xi)
            p_trial=p+sgn*dgamma
            q_trial=q+H*sgn*dgamma
            fs=ks*(d-B@p_trial)
            kt=ks.copy(); kt[yielded]=ks[yielded]*H[yielded]/(ks[yielded]+H[yielded])
            restoring=B.T@fs
            residual=-M@aa-C@vv-restoring-M@np.ones(n)*ag[step+1]
            Kt=B.T@np.diag(kt)@B+a0*M+a1*C
            if np.linalg.norm(residual) <= 1e-8*max(1.,np.linalg.norm(M@np.ones(n)*ag[step+1])):
                break
            try: trial_u += np.linalg.solve(Kt,residual)
            except np.linalg.LinAlgError: raise ValueError("Ma trận tiếp tuyến suy biến khi phân tích phi tuyến.")
        else:
            raise ValueError(f"Không hội tụ tại bước thời gian {step+1}; hãy giảm bước dt hoặc kiểm tra mô hình.")
        U[step+1]=trial_u; Acc[step+1]=aa; V[step+1]=vv
        p=p_trial; q=q_trial
        drifts[step+1]=B@trial_u; shear[step+1]=fs; plastic[step+1]=p
        if failure_time is None and np.any(np.abs(drifts[step+1]) >= dfail):
            failure_time=(step+1)*dt
    return {"time":np.arange(len(ag))*dt,"ground_acceleration":ag,"displacement":U,
            "velocity":V,"acceleration":Acc,"absolute_acceleration":Acc+ag[:,None],
            "story_drift":drifts,"story_shear":shear,"plastic_story_drift":plastic,
            "periods":2*np.pi/omega,"yield_story_displacement":dy,"yield_story_force":fy,
            "failure_time":failure_time,"failure_story_drift":dfail,"damping_matrix":C}


def rc_section_properties(width: float, depth: float, concrete_E: float,
                          steel_E: float, bar_diameter: float,
                          bars_top: int, bars_bottom: int,
                          cover: float, steel_yield: float,
                          cracked_inertia_ratio: float = 0.35) -> dict[str, float]:
    """Transformed RC section stiffness and approximate first-yield moment.

    Bar layers are represented at their centroidal cover locations. The
    cracked-concrete multiplier is an explicit modeling choice, not a TCVN
    prescribed coefficient. Flexural capacity is an equilibrium estimate and
    must not replace full section interaction/strain-compatibility design.
    """
    vals=(width,depth,concrete_E,steel_E,bar_diameter,cover,steel_yield,cracked_inertia_ratio)
    if any(not np.isfinite(v) or v<=0 for v in vals) or cracked_inertia_ratio>1:
        raise ValueError("Kích thước tiết diện và đặc trưng vật liệu phải dương; tỷ lệ nứt ≤ 1.")
    if bars_top<1 or bars_bottom<1:
        raise ValueError("Cần ít nhất một thanh thép ở mỗi mép chịu uốn.")
    if 2*(cover+bar_diameter/2)>=depth:
        raise ValueError("Lớp bảo vệ và cốt thép không vừa với chiều cao tiết diện.")
    as_bar=np.pi*bar_diameter**2/4
    ast=bars_top*as_bar; asb=bars_bottom*as_bar
    d=depth-cover-bar_diameter/2
    n=steel_E/concrete_E
    ig=width*depth**3/12
    ieff=cracked_inertia_ratio*ig+(n-1)*(ast+asb)*d**2
    # Reversing bending: take the weaker tension-side reinforcement.
    my=min(ast,asb)*steel_yield*0.8*depth
    return {"gross_I":float(ig),"effective_I":float(ieff),"steel_area_top":float(ast),
            "steel_area_bottom":float(asb),"yield_moment":float(my),"lever_arm":float(0.8*depth)}


def rc_circular_column_properties(diameter: float, concrete_E: float,
                                  steel_E: float, bar_diameter: float,
                                  bar_count: int, cover: float,
                                  steel_yield: float,
                                  cracked_inertia_ratio: float = 0.35) -> dict[str,float]:
    vals=(diameter,concrete_E,steel_E,bar_diameter,cover,steel_yield,cracked_inertia_ratio)
    if any(not np.isfinite(v) or v<=0 for v in vals) or cracked_inertia_ratio>1 or bar_count<4:
        raise ValueError("Thông số cột tròn không hợp lệ.")
    radius=diameter/2-cover-bar_diameter/2
    if radius<=0: raise ValueError("Cốt thép không vừa trong tiết diện tròn.")
    asbar=np.pi*bar_diameter**2/4; ast=bar_count*asbar; n=steel_E/concrete_E
    ig=np.pi*diameter**4/64
    ieff=cracked_inertia_ratio*ig+(n-1)*ast*radius**2/2
    my=(ast/2)*steel_yield*0.8*diameter
    return {"gross_I":float(ig),"effective_I":float(ieff),"steel_area_top":float(ast/2),
            "steel_area_bottom":float(ast/2),"yield_moment":float(my),"lever_arm":float(.8*diameter)}


def frame_story_stiffness(floors: int, bays_x: int, bays_y: int,
                          bay_x: float, bay_y: float, story_height: float,
                          E: float, column_I: float, beam_I: float,
                          footprint: str = "rect") -> np.ndarray:
    """Condensed lateral story stiffness from a 3D elastic frame FEM model.

    3D Euler-Bernoulli beam-column elements use 6 DOF/node. Floor translations
    in X and Y and floor rotations are condensed to obtain the X sway stiffness.
    This is a rigid-diaphragm shear-building idealization of the entered frame.
    """
    ints = (floors, bays_x, bays_y)
    if any(int(v) != v or v < 1 for v in ints) or floors > 20 or bays_x > 8 or bays_y > 8:
        raise ValueError("Số tầng phải 1–20; số nhịp X/Y phải 1–8.")
    if footprint not in {"rect", "L", "T"}:
        raise ValueError("Mặt bằng phải là rect, L hoặc T.")
    vals = [bay_x, bay_y, story_height, E, column_I, beam_I]
    if any(not np.isfinite(v) or v <= 0 for v in vals):
        raise ValueError("Kích thước, môđun đàn hồi và quán moment tiết diện phải dương.")
    nx, ny, nz = int(bays_x) + 1, int(bays_y) + 1, int(floors) + 1
    def active_bay(i, j):
        if footprint == "L":
            return not (i == int(bays_x)-1 and j == int(bays_y)-1)
        if footprint == "T":
            return j == 0 or (i == (int(bays_x)-1)//2 and j > 0)
        return True
    node_id = lambda i, j, k: (k * ny + j) * nx + i
    ndof = nx * ny * nz * 6
    K = np.zeros((ndof, ndof))

    def add_member(ni, nj, A, I1, I2, J):
        pi = np.array([ni % nx * bay_x, (ni // nx) % ny * bay_y, ni // (nx * ny) * story_height], dtype=float)
        pj = np.array([nj % nx * bay_x, (nj // nx) % ny * bay_y, nj // (nx * ny) * story_height], dtype=float)
        delta = pj - pi; L = np.linalg.norm(delta); ex = delta / L
        ref = np.array([0., 1., 0.]) if abs(ex[1]) < 0.9 else np.array([1., 0., 0.])
        ez = np.cross(ex, ref); ez /= np.linalg.norm(ez); ey = np.cross(ez, ex)
        R = np.vstack([ex, ey, ez]); T = np.zeros((12, 12))
        for q in range(4): T[q*3:q*3+3, q*3:q*3+3] = R
        k = np.zeros((12, 12)); EA=E*A/L; G=E/(2*1.3); GJ=G*J/L
        k[0,0]=k[6,6]=EA; k[0,6]=k[6,0]=-EA
        # torsion
        for a,b,c,d,v in [(3,3,9,9,GJ)]:
            k[a,a]=k[b,b]=k[c,c]=k[d,d]=v; k[a,c]=k[c,a]=-v
        # bending local y / rz
        kb=E*I2/L**3*np.array([[12,6*L,-12,6*L],[6*L,4*L**2,-6*L,2*L**2],[-12,-6*L,12,-6*L],[6*L,2*L**2,-6*L,4*L**2]])
        ids=[1,5,7,11]
        for a in range(4):
            for b in range(4): k[ids[a],ids[b]] += kb[a,b]
        # bending local z / ry, with sign convention on rotations
        kb2=E*I1/L**3*np.array([[12,-6*L,-12,-6*L],[-6*L,4*L**2,6*L,2*L**2],[-12,6*L,12,6*L],[-6*L,2*L**2,6*L,4*L**2]])
        ids=[2,4,8,10]
        for a in range(4):
            for b in range(4): k[ids[a],ids[b]] += kb2[a,b]
        kg=T.T @ k @ T
        dofs=np.r_[np.arange(ni*6,ni*6+6),np.arange(nj*6,nj*6+6)]
        K[np.ix_(dofs,dofs)] += kg

    Acol=max(column_I/(0.3**2), 1e-4); Abeam=max(beam_I/(0.3**2),1e-4)
    Jcol=2*column_I; Jbeam=2*beam_I
    for k in range(floors):
        for j in range(ny):
            for i in range(nx): add_member(node_id(i,j,k),node_id(i,j,k+1),Acol,column_I,column_I,Jcol)
    for k in range(1,nz):
        for j in range(ny):
            for i in range(int(bays_x)):
                if j < int(bays_y) and active_bay(i,j):
                    add_member(node_id(i,j,k),node_id(i+1,j,k),Abeam,beam_I,beam_I,Jbeam)
        for i in range(nx):
            for j in range(int(bays_y)):
                if i < int(bays_x) and active_bay(i,j):
                    add_member(node_id(i,j,k),node_id(i,j+1,k),Abeam,beam_I,beam_I,Jbeam)
    # Rigid floor diaphragms: equal X translation at all floor nodes. Other
    # floor DOFs remain independent; constrain floor Y translation softly to
    # stabilize the X-sway-only equivalent frame model.
    constrained=[]
    for j in range(ny):
        for i in range(nx): constrained.extend(range(node_id(i,j,0)*6,node_id(i,j,0)*6+6))
    free=np.ones(ndof,dtype=bool); free[constrained]=False
    floor_groups=[]
    for k in range(1,nz):
        group=[node_id(i,j,k)*6 for j in range(ny) for i in range(nx)]
        floor_groups.append(group)
    # transform q to physical DOFs: one common X translation per floor;
    # retain all non-X DOFs, then statically condense them.
    x_dofs=set(d for group in floor_groups for d in group)
    base=np.flatnonzero(free)
    q_dofs=[d for d in base if d not in x_dofs]
    Tred=np.zeros((len(base),floors+len(q_dofs)))
    map_base={d:i for i,d in enumerate(base)}
    for k,group in enumerate(floor_groups):
        for d in group: Tred[map_base[d],k]=1.0
    for q,d in enumerate(q_dofs): Tred[map_base[d],floors+q]=1.0
    Kr=Tred.T @ K[np.ix_(base,base)] @ Tred
    Kxx=Kr[:floors,:floors]; Kxr=Kr[:floors,floors:]; Krr=Kr[floors:,floors:]
    Kcond=Kxx-Kxr @ np.linalg.pinv(Krr,rcond=1e-10) @ Kxr.T if Krr.size else Kxx
    # Standard shear-building story stiffnesses reproduce condensed floor
    # stiffness matrix by matching its off-diagonal terms.
    story=np.empty(floors)
    for i in range(floors-1): story[i+1]=max(1.0,-Kcond[i,i+1])
    story[0]=max(1.0,Kcond[0,0]-story[1] if floors>1 else Kcond[0,0])
    return story
