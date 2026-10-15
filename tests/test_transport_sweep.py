"""Tests of the batched LD discrete-ordinates sweep (citrus/scatter_solver/transport_sweep.py): 1-D slab oracle,
particle balance, dense Galerkin assembly, Beer's law, chunking, and the fused Triton plane kernel."""

from __future__ import annotations

import math

import pytest
import torch

from citrus.scatter_solver.angular_quadrature import Quadrature, moments_from_ordinates, octant_of, product_quadrature
from citrus.scatter_solver.geometry import Grid
from citrus.scatter_solver.ld_cell import ld_cell_matrix
from citrus.scatter_solver.transport_sweep import make_sweep_context, sweep
from oracles.ld_slab_1d import ld_slab_sweep

F64 = torch.float64
F32 = torch.float32
OCT_SIGNS = torch.tensor(
    [[1 if not o & 4 else -1, 1 if not o & 2 else -1, 1 if not o & 1 else -1] for o in range(8)], dtype=F64
)
F32_REL = 1e-5  # float32 tolerance of the fused (Triton) path, relative to the field maximum


def _triton_usable() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "no CUDA"
    from citrus.scatter_solver.kernels.sweep_kernels import kernels_usable

    return kernels_usable(torch.device("cuda"))


class Backend:
    """Backend configuration of the oracle tests: the PyTorch path in float64 on the CPU with the original
    tolerances, or the fused Triton path in float32 on CUDA with a relative float32 tolerance."""

    def __init__(self, name: str):
        self.name = name
        self.f64 = name == "torch"
        self.dtype = F64 if self.f64 else F32
        self.device = torch.device("cpu" if self.f64 else "cuda")

    def ctx(self, grid, q, L):
        return make_sweep_context(grid, q, L, device=self.device, dtype=self.dtype, backend=self.name)

    def to(self, t):
        """float64 CPU input -> this backend's device / dtype."""
        return t.to(device=self.device, dtype=self.dtype)

    def tol(self, f64_tol: float, scale: float = 1.0) -> float:
        """The original float64 tolerance, or F32_REL * max(1, scale) for the float32 path."""
        return f64_tol if self.f64 else F32_REL * max(1.0, scale)


@pytest.fixture(params=["torch", "triton"])
def bk(request) -> Backend:
    if request.param == "triton":
        ok, why = _triton_usable()
        if not ok:
            pytest.skip(f"Triton sweep kernel unusable: {why}")
    return Backend(request.param)


def generic_octant_quadrature(base=(0.3, 0.5, 0.81), replace: dict[int, tuple[float, float, float]] | None = None,
                              dtype=F64) -> Quadrature:
    """One direction per octant (D8 = 1): |base| reflected into every octant, optionally replaced."""
    b = torch.tensor(base, dtype=F64)
    b = b / b.norm()
    om = b[None] * OCT_SIGNS
    for o, v in (replace or {}).items():
        om[o] = torch.tensor(v, dtype=F64)
    return Quadrature.from_directions(om.to(dtype), torch.full((8,), 4 * math.pi / 8, dtype=dtype))


def const_source_fn(src):
    """Direction-independent source src [G, 4, R, C, S] -> source_fn."""
    return lambda idx: src[None].expand(idx.numel(), *src.shape)


def dir_source_fn(src):
    """Direction-dependent source src [M, G, 4, R, C, S] (reordered order) -> source_fn."""
    return lambda idx: src.index_select(0, idx)


# ------------------------------------------------------------------------------------------------
# (a) 1D slab oracle
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [1, -1])
def test_slab_oracle(axis, sign, bk):
    torch.manual_seed(10 + axis)
    K = 9
    G = 2
    deltas = (0.7, 1.3, 0.9)  # (dx, dy, dz)
    shape = [3, 3, 3]  # (rows, cols, slices)
    vol_dim = {0: 1, 1: 0, 2: 2}[axis]  # x <-> cols, y <-> rows, z <-> slices
    shape[vol_dim] = K
    grid = Grid(tuple(shape), deltas[0], deltas[1] / deltas[0], deltas[2] / deltas[0])
    axis_dir = [0.0, 0.0, 0.0]
    axis_dir[axis] = float(sign)
    oct_code = int(octant_of(torch.tensor(axis_dir)))
    q = generic_octant_quadrature(replace={oct_code: tuple(axis_dir)})
    ctx = bk.ctx(grid, q, 1)
    m = oct_code  # D8 = 1: reordered index == octant code
    assert torch.equal(ctx.quad.omega[m].cpu().double(), torch.tensor(axis_dir, dtype=F64))
    # the sweep's sign for zero components is +1, consistent with the octant code
    assert torch.equal(ctx.s[oct_code, 0].cpu().double(), OCT_SIGNS[oct_code])

    mu1 = torch.rand(G, K, dtype=F64) * 1.5 + 0.05
    S0 = torch.rand(G, K, dtype=F64)
    S1 = torch.randn(G, K, dtype=F64) * 0.3

    def along(v):  # [G, K] -> [G, R, C, S] constant across the transverse axes
        view = [G, 1, 1, 1]
        view[1 + vol_dim] = K
        return v.reshape(view).expand(G, *shape).contiguous()

    mu = along(mu1)
    src = torch.zeros(G, 4, *shape, dtype=F64)
    src[:, 0] = along(S0)
    src[:, 1 + axis] = along(S1)
    _, psi = sweep(ctx, bk.to(mu), const_source_fn(bk.to(src)), return_psi=True)
    psi = psi.cpu().double()
    for g in range(G):
        p0, p1 = ld_slab_sweep(mu1[g].numpy(), S0[g].numpy(), S1[g].numpy(), deltas[axis], float(sign))
        ref0 = along(torch.as_tensor(p0)[None].expand(G, K))[g]
        ref1 = along(torch.as_tensor(p1)[None].expand(G, K))[g]
        tol = bk.tol(1e-12, float(ref0.abs().max()))
        assert torch.allclose(psi[m, g, 0], ref0, rtol=0, atol=tol)
        assert torch.allclose(psi[m, g, 1 + axis], ref1, rtol=0, atol=tol)
        for b in range(3):
            if b != axis:
                assert psi[m, g, 1 + b].abs().max() < tol


# ------------------------------------------------------------------------------------------------
# (b) exact particle balance for all 8 octants
# ------------------------------------------------------------------------------------------------

def _outflow(psi_m, omega, deltas, V):
    """sum over outflow boundary faces of |Omega_a| A_a (psi0 + s_a psi_a); psi_m [G, 4, R, C, S] -> [G]."""
    out = torch.zeros(psi_m.shape[0], dtype=psi_m.dtype)
    vol_dim = {0: 2, 1: 1, 2: 3}  # axis -> dim of [G, R, C, S]
    for a in range(3):
        s = 1.0 if omega[a] >= 0 else -1.0
        face = psi_m[:, 0] + s * psi_m[:, 1 + a]  # [G, R, C, S]
        idx = -1 if s > 0 else 0
        face_b = face.select(vol_dim[a], idx)
        out = out + abs(float(omega[a])) * (V / deltas[a]) * face_b.sum(dim=(1, 2))
    return out


def test_particle_balance_all_octants(bk):
    torch.manual_seed(1)
    shape = (4, 5, 3)
    grid = Grid(shape, 0.8, 1.25, 0.7)
    deltas = grid.deltas
    V = grid.voxel_volume
    G = 2
    q = generic_octant_quadrature(base=(0.62, 0.21, 0.45))
    ctx = bk.ctx(grid, q, 1)
    mu = bk.to(torch.rand(G, *shape, dtype=F64) * 2.0 + 0.05)
    src = torch.randn(8, G, 4, *shape, dtype=F64)
    src[:, :, 0] = src[:, :, 0].abs() + 0.1
    src = bk.to(src)
    _, psi = sweep(ctx, mu, dir_source_fn(src), return_psi=True)
    # balance evaluated in float64 from the (possibly float32) inputs and solution
    psi, mu, src = psi.cpu().double(), mu.cpu().double(), src.cpu().double()
    for m in range(8):
        omega = ctx.quad.omega[m].cpu().double()
        lhs = _outflow(psi[m], omega, deltas, V) + V * (mu * psi[m, :, 0]).sum(dim=(1, 2, 3))
        rhs = V * src[m, :, 0].sum(dim=(1, 2, 3))
        assert torch.allclose(lhs, rhs, rtol=bk.tol(1e-10), atol=0), (m, lhs, rhs)


# ------------------------------------------------------------------------------------------------
# (c) dense global LD assembly
# ------------------------------------------------------------------------------------------------

def _dense_ld_solve(omega, deltas, mu, src):
    """Assemble and solve the global upwind LD system for one direction and group.

    omega [3]; deltas (dx, dy, dz); mu [R, C, S]; src [4, R, C, S]. Returns psi [4, R, C, S].
    Unknown ordering: 4 * flat_cell + k, flat_cell = (i * C + j) * S + k_slice.
    """
    R, C, S = mu.shape
    N = R * C * S
    s = torch.where(omega < 0, -1.0, 1.0).to(F64)
    c = omega.abs() / torch.tensor(deltas, dtype=F64)
    A = torch.zeros(4 * N, 4 * N, dtype=F64)
    b = src.reshape(4, N).t().reshape(-1).clone()

    def flat(i, j, k):
        return (i * C + j) * S + k

    for i in range(R):
        for j in range(C):
            for k in range(S):
                n = flat(i, j, k)
                A[4 * n:4 * n + 4, 4 * n:4 * n + 4] = ld_cell_matrix(c, s, float(mu[i, j, k]))
                # upstream neighbor along each axis (x <-> j, y <-> i, z <-> k)
                ups = []
                for a in range(3):
                    ii, jj, kk = i, j, k
                    if a == 0:
                        jj = j - int(s[0])
                    elif a == 1:
                        ii = i - int(s[1])
                    else:
                        kk = k - int(s[2])
                    inside = 0 <= ii < R and 0 <= jj < C and 0 <= kk < S
                    ups.append(flat(ii, jj, kk) if inside else None)
                for a in range(3):
                    na = ups[a]
                    if na is None:
                        continue
                    # R_0 += c_a (psi0[N_a] + s_a psi_a[N_a])
                    A[4 * n, 4 * na] -= c[a]
                    A[4 * n, 4 * na + 1 + a] -= c[a] * s[a]
                    # R_a -= 3 s_a c_a (psi0[N_a] + s_a psi_a[N_a])
                    A[4 * n + 1 + a, 4 * na] += 3 * s[a] * c[a]
                    A[4 * n + 1 + a, 4 * na + 1 + a] += 3 * c[a]
                    # R_b += c_a psi_b[N_a] for b != a (transverse coupling)
                    for bb in range(3):
                        if bb != a:
                            A[4 * n + 1 + bb, 4 * na + 1 + bb] -= c[a]
    x = torch.linalg.solve(A, b)
    return x.reshape(N, 4).t().reshape(4, R, C, S)


@pytest.mark.parametrize("n", [2, 3])
def test_dense_global_assembly(n, bk):
    torch.manual_seed(2 + n)
    shape = (n, n + 1, n) if n == 2 else (n, n, n)
    grid = Grid(shape, 0.9, 1.2, 0.75)
    G = 2
    q = generic_octant_quadrature(base=(0.55, 0.38, 0.74))
    ctx = bk.ctx(grid, q, 1)
    mu = bk.to(torch.rand(G, *shape, dtype=F64) + 0.1)
    src = bk.to(torch.randn(8, G, 4, *shape, dtype=F64))
    _, psi = sweep(ctx, mu, dir_source_fn(src), return_psi=True)
    psi, mu, src = psi.cpu().double(), mu.cpu().double(), src.cpu().double()
    for m in range(8):
        for g in range(G):
            ref = _dense_ld_solve(ctx.quad.omega[m].cpu().double(), grid.deltas, mu[g], src[m, g])
            scale = float(ref.abs().max())
            assert torch.allclose(psi[m, g], ref, rtol=0, atol=bk.tol(1e-12 * max(1.0, scale), scale))


# ------------------------------------------------------------------------------------------------
# (d) pure absorber, localized source: exponential decay along the direction, refinement
# ------------------------------------------------------------------------------------------------

def test_point_source_absorber_refinement():
    """Source cube [-a/2, a/2]^3 in a pure absorber [-H, H]^3; per direction the fluence integrated over
    slabs z in [z_k, z_k+1] downstream of the source equals (analytically, the beam leaves through the
    z faces only) a^2 int dz0 (e^{-mu (z_k - z0)/|Wz|} - e^{-mu (z_k+1 - z0)/|Wz|}) / mu. The error
    must fall by >= 3 per 2x refinement."""
    H, a, mu_val = 2.0, 0.5, 0.5
    q = generic_octant_quadrature(base=(0.3, 0.5, 0.81))
    slabs = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    errors = []
    for n in (8, 16, 32):
        h = 2 * H / n
        grid = Grid((n, n, n), h)
        ctx = make_sweep_context(grid, q, 0, dtype=F64)
        kk = int(round(a / h))
        c0 = n // 2 - kk // 2
        src = torch.zeros(1, 4, n, n, n, dtype=F64)
        src[0, 0, c0:c0 + kk, c0:c0 + kk, c0:c0 + kk] = 1.0
        _, psi = sweep(ctx, torch.full((1, n, n, n), mu_val, dtype=F64), const_source_fn(src), return_psi=True)
        z_edges = -H + h * torch.arange(n + 1, dtype=F64)
        err = 0.0
        for m in range(8):
            w = ctx.quad.omega[m]
            wz = abs(float(w[2]))
            # the beam from every source point exits through a z face (needed for the analytic slab values)
            assert (H - a / 2) / wz < (H - a / 2) / max(abs(float(w[0])), abs(float(w[1])))
            for z0, z1 in slabs:
                lo, hi = (z0, z1) if w[2] > 0 else (-z1, -z0)
                sel = (z_edges[:-1] >= lo - 1e-12) & (z_edges[1:] <= hi + 1e-12)
                num = float(psi[m, 0, 0][:, :, sel].sum()) * h ** 3
                d0, d1 = z0, z1  # distances from the source mid-plane along z (mirrored for wz < 0)

                def edge(d):
                    return wz / mu_val * math.exp(-mu_val * d / wz) * 2 * math.sinh(mu_val * a / (2 * wz))

                exact = a * a * (edge(d0) - edge(d1)) / mu_val
                err = max(err, abs(num - exact) / exact)
        errors.append(err)
    assert errors[0] / errors[1] >= 3.0 and errors[1] / errors[2] >= 3.0, errors


# ------------------------------------------------------------------------------------------------
# (e) rotational (reflection) consistency
# ------------------------------------------------------------------------------------------------

def test_reflection_symmetry_isotropic_source():
    n = 5
    grid = Grid((n, n, n), 1.0)
    q = product_quadrature(8, 16, dtype=F64)
    ctx = make_sweep_context(grid, q, 3, dtype=F64)
    src = torch.zeros(1, 4, n, n, n, dtype=F64)
    src[0, 0, 1:4, 1:4, 1:4] = 0.5
    src[0, 0, 2, 2, 2] = 2.0
    mu = torch.full((1, n, n, n), 0.4, dtype=F64)
    phi = sweep(ctx, mu, const_source_fn(src))
    p0 = phi[0, 0]  # [4, R, C, S] l = 0 moment
    scale = float(p0[0].abs().max())
    tol = 1e-12 * scale
    dims = {0: 2, 1: 1, 2: 3}  # axis a -> dim of [4, R, C, S] (x <-> cols, y <-> rows, z <-> slices)
    for a in range(3):
        f = torch.flip(p0, dims=(dims[a],))
        assert torch.allclose(f[0], p0[0], atol=tol, rtol=0)
        for b in range(3):
            sgn = -1.0 if b == a else 1.0
            assert torch.allclose(f[1 + b], sgn * p0[1 + b], atol=tol, rtol=0)
    # l = 1 moments of the cell average vanish at the center cell
    assert phi[0, 1:4, 0, 2, 2, 2].abs().max() < tol
    assert float(phi[0, 0, 0, 2, 2, 2]) > 0


# ------------------------------------------------------------------------------------------------
# (f) on-the-fly moments == moments_from_ordinates(psi); chunking; devices
# ------------------------------------------------------------------------------------------------

def _random_problem(shape, G, M, dtype=F64, device="cpu", seed=5):
    gen = torch.Generator().manual_seed(seed)
    mu = (torch.rand(G, *shape, generator=gen, dtype=F64) + 0.2).to(device=device, dtype=dtype)
    src = torch.randn(M, G, 4, *shape, generator=gen, dtype=F64)
    src[:, :, 0] = src[:, :, 0].abs()
    return mu, src.to(device=device, dtype=dtype)


def test_moments_match_ordinates(bk):
    grid = Grid((3, 4, 5), 0.6, 1.1, 0.9)
    q = product_quadrature(4, 8, dtype=F64)
    ctx = bk.ctx(grid, q, 3)
    mu, src = _random_problem(grid.shape, 2, ctx.num_directions, dtype=bk.dtype, device=bk.device)
    phi, psi = sweep(ctx, mu, dir_source_fn(src), return_psi=True)
    ref = moments_from_ordinates(psi.movedim(0, -1), ctx.basis).movedim(-1, 1)  # [G, N, 4, R, C, S]
    assert phi.shape == (2, 16, 4, 3, 4, 5)
    if bk.f64:
        assert torch.allclose(phi, ref, rtol=1e-12, atol=1e-13)
    else:
        assert torch.allclose(phi, ref, rtol=F32_REL, atol=F32_REL * float(ref.abs().max()))


def test_direction_chunks_and_device(device):
    grid = Grid((4, 3, 5), 0.5)
    q = product_quadrature(4, 8, dtype=F64)  # D8 = 4
    ctx64 = make_sweep_context(grid, q, 2, dtype=F64)
    mu, src = _random_problem(grid.shape, 3, ctx64.num_directions)
    ref = sweep(ctx64, mu, dir_source_fn(src))
    for chunk in (1, 3):
        assert torch.allclose(sweep(ctx64, mu, dir_source_fn(src), direction_chunk=chunk), ref, rtol=1e-12, atol=1e-13)
    ctx = make_sweep_context(grid, q, 2, device=device, dtype=torch.float32)
    mu32, src32 = mu.to(device, torch.float32), src.to(device, torch.float32)
    out = sweep(ctx, mu32, dir_source_fn(src32), direction_chunk=2)
    assert out.device.type == device.type and out.dtype == torch.float32
    assert torch.allclose(out.cpu().double(), ref, rtol=1e-4, atol=1e-5 * float(ref.abs().max()))



# ------------------------------------------------------------------------------------------------
# (g) fused Triton plane kernel vs the PyTorch plane step; backend selection
# ------------------------------------------------------------------------------------------------

def _require_triton():
    ok, why = _triton_usable()
    if not ok:
        pytest.skip(f"Triton sweep kernel unusable: {why}")


def _heterogeneous_problem(shape=(12, 10, 14), G=6, seed=21):
    """Random heterogeneous float32 CUDA problem: mu spanning thin to thick cells, signed sources."""
    dev = torch.device("cuda")
    gen = torch.Generator().manual_seed(seed)
    mu = torch.exp(torch.empty(G, *shape, dtype=F64).uniform_(-3.0, 1.5, generator=gen))  # 0.05 .. 4.5 / cm
    grid = Grid(shape, 0.8, 1.15, 0.9)
    ctx = make_sweep_context(grid, product_quadrature(8, 16, dtype=F64), 3, device=dev, dtype=F32)
    src = torch.randn(ctx.num_directions, G, 4, *shape, generator=gen, dtype=F64)
    src[:, :, 0] = src[:, :, 0].abs() * (torch.rand(G, *shape, generator=gen, dtype=F64) < 0.3)  # sparse sources
    return ctx, mu.to(dev, F32), src.to(dev, F32)


def _rel(a, b):
    return float((a - b).abs().max()) / float(b.abs().max())


@pytest.mark.parametrize("chunk", [None, 5])
def test_triton_matches_torch_random_heterogeneous(chunk):
    _require_triton()
    ctx, mu, src = _heterogeneous_problem()
    assert ctx.num_directions == 128
    kw = dict(direction_chunk=chunk, return_psi=True)
    phi_t, psi_t = sweep(ctx, mu, dir_source_fn(src), backend="torch", **kw)
    phi_k, psi_k = sweep(ctx, mu, dir_source_fn(src), backend="triton", **kw)
    assert _rel(psi_k, psi_t) < F32_REL and _rel(phi_k, phi_t) < F32_REL
    assert float(psi_k[:, :, 0].min()) < 0.0  # the problem does exercise negative cell means


def test_backend_selection():
    grid = Grid((3, 3, 4), 0.5)
    q = product_quadrature(4, 8, dtype=F64)
    mu, src = _random_problem(grid.shape, 2, 32)
    with pytest.raises(ValueError):
        make_sweep_context(grid, q, 1, dtype=F64, backend="bogus")
    with pytest.raises(ValueError):
        make_sweep_context(grid, q, 1, dtype=F64, backend="triton")  # CPU context
    ctx = make_sweep_context(grid, q, 1, dtype=F64)  # "auto" on the CPU: PyTorch path
    assert ctx.backend == "auto" and ctx.flat is None
    ref = sweep(ctx, mu, dir_source_fn(src), backend="torch")
    assert torch.equal(sweep(ctx, mu, dir_source_fn(src)), ref)
    with pytest.raises(ValueError):
        sweep(ctx, mu, dir_source_fn(src), backend="triton")
    with pytest.raises(ValueError):
        sweep(ctx, mu, dir_source_fn(src), backend="bogus")
    if torch.cuda.is_available():
        ctx_c = make_sweep_context(grid, q, 1, device="cuda", dtype=F64)
        out = sweep(ctx_c, mu.cuda(), dir_source_fn(src.cuda()))  # float64 on CUDA, "auto": PyTorch path
        assert torch.allclose(out.cpu(), ref, rtol=1e-12, atol=1e-13)
        with pytest.raises(ValueError):
            sweep(ctx_c, mu.cuda(), dir_source_fn(src.cuda()), backend="triton")  # float64 is not fused


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_triton_kill_switch():
    import os
    import subprocess
    import sys

    code = ("import torch; from citrus.scatter_solver.kernels import sweep_kernels as st; "
            "ok, why = st.kernels_usable(torch.device('cuda')); assert not ok and 'CITRUS_DISABLE_TRITON' in why, why; "
            "from citrus.scatter_solver.transport_sweep import _use_triton, make_sweep_context; "
            "from citrus.scatter_solver.angular_quadrature import product_quadrature; from citrus.scatter_solver.geometry import Grid; "
            "ctx = make_sweep_context(Grid((2, 2, 2), 1.0), product_quadrature(2, 4, dtype=torch.float64), 0, "
            "device='cuda'); assert not _use_triton(ctx, None, torch.device('cuda'), torch.float32)")
    env = dict(os.environ, CITRUS_DISABLE_TRITON="1")
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]


# ------------------------------------------------------------------------------------------------
# (h) source evaluation: one source_fn call per chunk (all octants), `out=` lending
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("chunk", [None, 3])
def test_source_called_once_per_chunk(device, chunk):
    grid = Grid((3, 4, 3), 0.7)
    q = product_quadrature(4, 8, dtype=F64)  # D8 = 4
    ctx = make_sweep_context(grid, q, 2, device=device, dtype=F64)
    mu, src = _random_problem(grid.shape, 2, ctx.num_directions, device=device)
    calls, calls_out = [], []

    def plain(idx):
        calls.append(idx.clone())
        return src.index_select(0, idx)

    def with_out(idx, out=None):
        calls_out.append((idx.clone(), out is not None))
        assert out is not None and out.is_contiguous() and out.shape == (idx.numel(),) + tuple(src.shape[1:])
        out.copy_(src.index_select(0, idx))
        return out

    ref = sweep(ctx, mu, dir_source_fn(src), direction_chunk=chunk)
    out_plain = sweep(ctx, mu, plain, direction_chunk=chunk)
    out_lend = sweep(ctx, mu, with_out, direction_chunk=chunk)
    assert torch.allclose(out_plain, ref, rtol=1e-13, atol=1e-14)
    assert torch.allclose(out_lend, ref, rtol=1e-13, atol=1e-14)
    dc = 4 if chunk is None else 3
    starts = list(range(0, 4, dc))
    assert len(calls) == len(starts) and len(calls_out) == len(starts)
    for d0, idx, (idx_o, lent) in zip(starts, calls, calls_out):
        expect = ctx.direction_indices(d0, min(d0 + dc, 4))
        assert torch.equal(idx, expect) and torch.equal(idx_o, expect) and lent
