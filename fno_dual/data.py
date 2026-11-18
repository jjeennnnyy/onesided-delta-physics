import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def generate_wave_dataset(
    n_samples=256,
    n_grid=128,
    c=1.0,
    T=0.25,
    dt=1e-3,
    seed=0,
    device="cpu",
):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_grid, endpoint=False)

    def random_field(max_mode=8, scale=1.0):
        amps = rng.normal(0, 1, size=max_mode) * scale / (np.arange(1, max_mode + 1))
        phase = rng.uniform(0, 2 * np.pi, size=max_mode)
        out = np.zeros_like(x)
        for k in range(1, max_mode + 1):
            out += amps[k - 1] * np.sin(2 * np.pi * k * x + phase[k - 1])
        return out

    n_steps = int(T / dt)
    dx = 1.0 / n_grid
    if c * dt / dx > 1.0:
        raise ValueError(f"CFL too large: c*dt/dx = {c * dt / dx:.3f}")

    def lap(u):
        return (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / (dx * dx)

    inputs, targets = [], []
    for _ in range(n_samples):
        u0 = random_field(max_mode=8, scale=1.0)
        v0 = random_field(max_mode=8, scale=0.5)
        u_prev = u0.copy()
        u_curr = u0 + dt * v0 + 0.5 * (dt * dt) * (c * c) * lap(u0)
        for _ in range(1, n_steps):
            u_next = 2 * u_curr - u_prev + (dt * dt) * (c * c) * lap(u_curr)
            u_prev, u_curr = u_curr, u_next
        inputs.append(np.stack([u0, v0], axis=-1).astype(np.float32))
        targets.append(u_curr.astype(np.float32))

    A = torch.tensor(np.stack(inputs, axis=0), dtype=torch.float32, device=device)
    U = torch.tensor(np.stack(targets, axis=0), dtype=torch.float32, device=device)
    return A, U


def add_grid_channel_1d(A):
    batch, n_grid, _ = A.shape
    grid = torch.linspace(0, 1, n_grid, device=A.device, dtype=A.dtype).view(1, n_grid, 1)
    return torch.cat([A, grid.repeat(batch, 1, 1)], dim=-1)


def generate_random_coefficient_field(n_grid, rng, max_mode=4, scale=0.8):
    x = np.linspace(0, 1, n_grid)
    y = np.linspace(0, 1, n_grid)
    X, Y = np.meshgrid(x, y, indexing="ij")
    log_a = np.zeros((n_grid, n_grid), dtype=np.float64)
    for kx in range(1, max_mode + 1):
        for ky in range(1, max_mode + 1):
            amp = rng.normal(0, scale) / ((kx * kx + ky * ky) ** 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            log_a += amp * np.sin(2 * np.pi * (kx * X + ky * Y) + phase)
    return np.exp(log_a).astype(np.float32)


def solve_darcy_fd(a):
    n = a.shape[0]
    h = 1.0 / (n - 1)
    interior = n - 2
    size = interior * interior
    mat = lil_matrix((size, size), dtype=np.float64)
    rhs = np.ones(size, dtype=np.float64)

    def idx(i, j):
        return (i - 1) * interior + (j - 1)

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            row = idx(i, j)
            ae = 0.5 * (a[i, j] + a[i + 1, j])
            aw = 0.5 * (a[i, j] + a[i - 1, j])
            an = 0.5 * (a[i, j] + a[i, j + 1])
            ass = 0.5 * (a[i, j] + a[i, j - 1])
            mat[row, row] = (ae + aw + an + ass) / (h * h)
            if i + 1 <= n - 2:
                mat[row, idx(i + 1, j)] = -ae / (h * h)
            if i - 1 >= 1:
                mat[row, idx(i - 1, j)] = -aw / (h * h)
            if j + 1 <= n - 2:
                mat[row, idx(i, j + 1)] = -an / (h * h)
            if j - 1 >= 1:
                mat[row, idx(i, j - 1)] = -ass / (h * h)

    sol = spsolve(mat.tocsr(), rhs)
    u = np.zeros((n, n), dtype=np.float32)
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            u[i, j] = sol[idx(i, j)]
    return u


def generate_darcy_dataset(n_samples=128, n_grid=32, seed=0, device="cpu"):
    rng = np.random.default_rng(seed)
    coeffs, sols = [], []
    for _ in range(n_samples):
        a = generate_random_coefficient_field(n_grid, rng)
        u = solve_darcy_fd(a)
        coeffs.append(a[..., None])
        sols.append(u)
    A = torch.tensor(np.stack(coeffs, axis=0), dtype=torch.float32, device=device)
    U = torch.tensor(np.stack(sols, axis=0), dtype=torch.float32, device=device)
    return A, U


def add_grid_channels_2d(A):
    batch, h, w, _ = A.shape
    x = torch.linspace(0, 1, h, device=A.device, dtype=A.dtype)
    y = torch.linspace(0, 1, w, device=A.device, dtype=A.dtype)
    X, Y = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([X, Y], dim=-1).unsqueeze(0).repeat(batch, 1, 1, 1)
    return torch.cat([A, grid], dim=-1)


def _periodic_random_field_1d(x, rng, max_mode=6, scale=0.8):
    field = np.zeros_like(x, dtype=np.float64)
    for mode in range(1, max_mode + 1):
        amplitude = rng.normal() * scale / (mode**1.5)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field += amplitude * np.sin(2.0 * np.pi * mode * x + phase)
    return field


def generate_burgers_dataset(
    n_samples=128,
    n_grid=64,
    viscosity=0.01,
    T=0.5,
    dt=2e-3,
    seed=0,
    device="cpu",
):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    dealias = np.abs(np.fft.fftfreq(n_grid) * n_grid) <= n_grid / 3

    def rhs(u):
        u_hat = np.fft.fft(u)
        ux = np.fft.ifft(1j * wave_number * u_hat).real
        nonlinear_hat = np.fft.fft(u * ux) * dealias
        return -np.fft.ifft(nonlinear_hat).real + viscosity * np.fft.ifft(
            -(wave_number**2) * u_hat
        ).real

    n_steps = int(round(T / dt))
    inputs, targets = [], []
    for _ in range(n_samples):
        u0 = _periodic_random_field_1d(x, rng)
        u0 *= 0.8 / max(np.max(np.abs(u0)), 1e-8)
        u = u0.copy()
        for _ in range(n_steps):
            k1 = rhs(u)
            k2 = rhs(u + 0.5 * dt * k1)
            k3 = rhs(u + 0.5 * dt * k2)
            k4 = rhs(u + dt * k3)
            u += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        inputs.append(u0[:, None].astype(np.float32))
        targets.append(u.astype(np.float32))
    return (
        torch.tensor(np.stack(inputs), dtype=torch.float32, device=device),
        torch.tensor(np.stack(targets), dtype=torch.float32, device=device),
    )


def generate_burgers_trajectories(
    n_trajectories=32,
    n_grid=64,
    viscosity=0.01,
    n_steps=50,
    step_dt=0.02,
    solver_dt=2e-3,
    seed=0,
    device="cpu",
):
    """Generate periodic viscous Burgers trajectories with spectral RK4."""
    substeps = int(round(step_dt / solver_dt))
    if substeps < 1 or not np.isclose(substeps * solver_dt, step_dt):
        raise ValueError("step_dt must be a positive integer multiple of solver_dt")
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    wave_number = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    dealias = np.abs(np.fft.fftfreq(n_grid) * n_grid) <= n_grid / 3

    def rhs(u):
        u_hat = np.fft.fft(u)
        ux = np.fft.ifft(1j * wave_number * u_hat).real
        nonlinear_hat = np.fft.fft(u * ux) * dealias
        return -np.fft.ifft(nonlinear_hat).real + viscosity * np.fft.ifft(
            -(wave_number**2) * u_hat
        ).real

    trajectories = []
    for _ in range(n_trajectories):
        u = _periodic_random_field_1d(x, rng)
        u *= 0.8 / max(np.max(np.abs(u)), 1e-8)
        frames = [u.astype(np.float32).copy()]
        for _ in range(n_steps):
            for _ in range(substeps):
                k1 = rhs(u)
                k2 = rhs(u + 0.5 * solver_dt * k1)
                k3 = rhs(u + 0.5 * solver_dt * k2)
                k4 = rhs(u + solver_dt * k3)
                u += (solver_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            frames.append(u.astype(np.float32).copy())
        trajectories.append(np.stack(frames))
    return torch.tensor(np.stack(trajectories), dtype=torch.float32, device=device)


def _periodic_random_field_2d(n_grid, rng, max_mode=4, scale=1.0):
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    field = np.zeros((n_grid, n_grid), dtype=np.float64)
    for kx in range(0, max_mode + 1):
        for ky in range(0, max_mode + 1):
            if kx == 0 and ky == 0:
                continue
            radius = max(1.0, np.sqrt(kx * kx + ky * ky))
            amplitude = rng.normal() * scale / (radius**2.0)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            field += amplitude * np.cos(2.0 * np.pi * (kx * X + ky * Y) + phase)
    field -= field.mean()
    return field


def generate_allen_cahn_dataset(
    n_samples=128,
    n_grid=24,
    epsilon=0.04,
    T=0.2,
    dt=5e-3,
    seed=0,
    device="cpu",
):
    rng = np.random.default_rng(seed)
    frequency = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    KX, KY = np.meshgrid(frequency, frequency, indexing="ij")
    diffusion_denom = 1.0 + dt * epsilon**2 * (KX**2 + KY**2)
    n_steps = int(round(T / dt))
    inputs, targets = [], []
    for _ in range(n_samples):
        u0 = _periodic_random_field_2d(n_grid, rng, max_mode=4, scale=1.0)
        u0 *= 0.75 / max(np.max(np.abs(u0)), 1e-8)
        u = u0.copy()
        for _ in range(n_steps):
            reaction = u - u**3
            u = np.fft.ifft2(np.fft.fft2(u + dt * reaction) / diffusion_denom).real
        inputs.append(u0[..., None].astype(np.float32))
        targets.append(u.astype(np.float32))
    return (
        torch.tensor(np.stack(inputs), dtype=torch.float32, device=device),
        torch.tensor(np.stack(targets), dtype=torch.float32, device=device),
    )


def generate_navier_stokes_dataset(
    n_samples=128,
    n_grid=24,
    viscosity=1e-3,
    T=0.2,
    dt=5e-3,
    seed=0,
    device="cpu",
):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    frequency = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    KX, KY = np.meshgrid(frequency, frequency, indexing="ij")
    k_square = KX**2 + KY**2
    safe_k_square = k_square.copy()
    safe_k_square[0, 0] = 1.0
    integer_frequency = np.fft.fftfreq(n_grid) * n_grid
    IX, IY = np.meshgrid(integer_frequency, integer_frequency, indexing="ij")
    dealias = (np.abs(IX) <= n_grid / 3) & (np.abs(IY) <= n_grid / 3)
    forcing = 0.1 * (
        np.sin(2.0 * np.pi * (X + Y)) + np.cos(2.0 * np.pi * (X - Y))
    )

    def rhs(vorticity):
        omega_hat = np.fft.fft2(vorticity)
        psi_hat = omega_hat / safe_k_square
        psi_hat[0, 0] = 0.0
        velocity_x = np.fft.ifft2(1j * KY * psi_hat).real
        velocity_y = np.fft.ifft2(-1j * KX * psi_hat).real
        omega_x = np.fft.ifft2(1j * KX * omega_hat).real
        omega_y = np.fft.ifft2(1j * KY * omega_hat).real
        advection_hat = np.fft.fft2(velocity_x * omega_x + velocity_y * omega_y) * dealias
        advection = np.fft.ifft2(advection_hat).real
        diffusion = np.fft.ifft2(-viscosity * k_square * omega_hat).real
        return -advection + diffusion + forcing

    n_steps = int(round(T / dt))
    inputs, targets = [], []
    for _ in range(n_samples):
        omega0 = _periodic_random_field_2d(n_grid, rng, max_mode=4, scale=1.0)
        omega0 /= max(omega0.std(), 1e-8)
        omega = omega0.copy()
        for _ in range(n_steps):
            k1 = rhs(omega)
            k2 = rhs(omega + 0.5 * dt * k1)
            k3 = rhs(omega + 0.5 * dt * k2)
            k4 = rhs(omega + dt * k3)
            omega += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        inputs.append(omega0[..., None].astype(np.float32))
        targets.append(omega.astype(np.float32))
    return (
        torch.tensor(np.stack(inputs), dtype=torch.float32, device=device),
        torch.tensor(np.stack(targets), dtype=torch.float32, device=device),
    )


def generate_navier_stokes_trajectories(
    n_trajectories=32,
    n_grid=32,
    viscosity=1e-3,
    n_steps=20,
    step_dt=0.05,
    solver_dt=5e-3,
    seed=0,
    device="cpu",
):
    """Generate forced 2-D vorticity trajectories for autoregressive tests.

    Consecutive returned frames are separated by ``step_dt``.  The same seed
    therefore gives identical initial conditions across viscosity shifts.
    """
    substeps = int(round(step_dt / solver_dt))
    if substeps < 1 or not np.isclose(substeps * solver_dt, step_dt):
        raise ValueError("step_dt must be a positive integer multiple of solver_dt")

    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_grid, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    frequency = 2.0 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    KX, KY = np.meshgrid(frequency, frequency, indexing="ij")
    k_square = KX**2 + KY**2
    safe_k_square = k_square.copy()
    safe_k_square[0, 0] = 1.0
    integer_frequency = np.fft.fftfreq(n_grid) * n_grid
    IX, IY = np.meshgrid(integer_frequency, integer_frequency, indexing="ij")
    dealias = (np.abs(IX) <= n_grid / 3) & (np.abs(IY) <= n_grid / 3)
    forcing = 0.1 * (
        np.sin(2.0 * np.pi * (X + Y)) + np.cos(2.0 * np.pi * (X - Y))
    )

    def rhs(vorticity):
        omega_hat = np.fft.fft2(vorticity)
        psi_hat = omega_hat / safe_k_square
        psi_hat[0, 0] = 0.0
        velocity_x = np.fft.ifft2(1j * KY * psi_hat).real
        velocity_y = np.fft.ifft2(-1j * KX * psi_hat).real
        omega_x = np.fft.ifft2(1j * KX * omega_hat).real
        omega_y = np.fft.ifft2(1j * KY * omega_hat).real
        advection_hat = np.fft.fft2(velocity_x * omega_x + velocity_y * omega_y) * dealias
        advection = np.fft.ifft2(advection_hat).real
        diffusion = np.fft.ifft2(-viscosity * k_square * omega_hat).real
        return -advection + diffusion + forcing

    trajectories = []
    for _ in range(n_trajectories):
        omega = _periodic_random_field_2d(n_grid, rng, max_mode=4, scale=1.0)
        omega /= max(omega.std(), 1e-8)
        frames = [omega.astype(np.float32).copy()]
        for _ in range(n_steps):
            for _ in range(substeps):
                k1 = rhs(omega)
                k2 = rhs(omega + 0.5 * solver_dt * k1)
                k3 = rhs(omega + 0.5 * solver_dt * k2)
                k4 = rhs(omega + solver_dt * k3)
                omega += (solver_dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            frames.append(omega.astype(np.float32).copy())
        trajectories.append(np.stack(frames))
    return torch.tensor(np.stack(trajectories), dtype=torch.float32, device=device)


def trajectory_pairs(trajectories):
    """Flatten [trajectory, time] into one-step operator pairs."""
    inputs = trajectories[:, :-1].reshape(-1, *trajectories.shape[2:])
    targets = trajectories[:, 1:].reshape(-1, *trajectories.shape[2:])
    return inputs.unsqueeze(-1), targets
