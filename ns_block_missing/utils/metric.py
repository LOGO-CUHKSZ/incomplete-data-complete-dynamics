import torch

from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

# from pde_generator.advection import simulate_2d_advection

__all__ = [
    'pde_loss'
]

def pde_loss(data: Tensor, variables: Tensor, dataset: Dataset) -> Tensor:
    if dataset.pde.startswith('2d_sw'):
        return sw_pde_loss(
            data, dataset.dx, dataset.dx, dataset.dt,
            variables, dataset.coriolis_param
        )
    elif dataset.pde.startswith('2d_adv'):
        return adv_2d_loss(
            data, variables[:, 0], variables[:, 1], dataset.times, return_sol=False
        )
    elif dataset.pde.startswith('2d_heat'):
        return heat_2d_loss(
            data, variables, dataset.times, return_sol=False
        )
    raise NotImplementedError()
    return 


def sw_pde_loss(
        data: Tensor,
        dx: float, dy: float, dt: float,
        gd: Tensor, coriolis: Tensor
    ) -> Tensor:
    # gravity.shape, depth.shape == (batch_size, )
    gravity, depth = gd.T
    gravity = gravity.view(-1, 1, 1, 1)
    depth = depth.view(-1, 1, 1, 1)

    # data.shape == (batch_size, n_channel, n_time, img_size, img_size)
    # h.shape, u.shape, v.shape == (batch_size, n_time, img_size, img_size)
    h, u, v = data.transpose(1, 0)
    h = h + depth

    v_avg = 0.25 * (v[:, :, 1:-1, 1:-1] + v[:, :, :-2, 1:-1] + v[:, :, 1:-1, 2:] + v[:, :, :-2, 2:])
    u_avg = 0.25 * (u[:, :, 1:-1, 1:-1] + u[:, :, 1:-1, :-2] + u[:, :, 2:, 1:-1] + u[:, :, 2:, :-2])

    dudt = torch.diff(u, dim=1)[:, :, 1:-1, 1:-1] / dt
    dvdt = torch.diff(v, dim=1)[:, :, 1:-1, 1:-1] / dt
    dhdt = torch.diff(h, dim=1)[:, :, 1:-1, 1:-1] / dt

    dhdx = (h[:, :, 1:-1, 2:] - h[:, :, 1:-1, 1:-1]) / dx
    dhdy = (h[:, :, 2:, 1:-1] - h[:, :, 1:-1, 1:-1]) / dy
    dudx = (u[:, :, 1:-1, 1:-1] - u[:, :, 1:-1, :-2]) / dx
    dvdy = (v[:, :, 1:-1, 1:-1] - v[:, :, :-2, 1:-1]) / dy

    loss1 = dudt - (coriolis * v_avg - gravity * dhdx)[:, :-1]
    loss2 = dvdt + coriolis * u_avg[:, 1:] + gravity * dhdy[:, :-1]
    loss3 = dhdt + depth * (dudx + dvdy)[:, 1:]
    loss = torch.stack([loss1, loss2, loss3], dim=1).square()
    return 1e8 * loss.mean()


def adv_2d_loss(u: Tensor, ax: Tensor, ay: Tensor, t: Tensor, return_sol: bool = False) -> Tensor | tuple[Tensor, Tensor]:
    # MSE loss of numerical simulation and samples
    # u.shape == (batch_size, n_t, n_x, n_y)
    # ax.shape, ay.shape == (batch_size, )
    sol = torch.zeros_like(u, device=u.device)
    batch_size, _, nt, _, _ = u.shape
    total_time = t[-1]
    for i in range(batch_size):
        sol[i] = simulate_2d_advection(u[i, 0, 0], ax[i], ay[i], total_time, nt)
    pde_loss = (sol - u) ** 2
    if return_sol:
        return pde_loss.mean(), sol
    else:
        return pde_loss.mean()


def simulate_2d_advection(u0: Tensor, ax: Tensor, ay: Tensor, total_time: float, nt: float) -> Tensor:
    nx, ny = u0.shape
    device = u0.device

    rows = torch.linspace(-1, 1, nx, device=device)
    cols = torch.linspace(-1, 1, ny, device=device)
    X, Y = torch.meshgrid(rows, cols, indexing='xy')

    times = torch.linspace(0, total_time, nt, device=device)
    dt = times[-1] - times[-2]

    u = torch.empty((nt, nx, ny), device=device)
    u[0] = u0.clone()
    u0_batched = u0.unsqueeze(0).unsqueeze(0)
    
    for n in range(1, nt):
        t = dt * n

        X_new = X + ax * t
        Y_new = Y + ay * t
        
        X_new = torch.remainder(X_new + 1, 2) - 1
        Y_new = torch.remainder(Y_new + 1, 2) - 1
        
        grid = torch.stack([X_new, Y_new], dim=-1).unsqueeze(0)  # (1, nx, ny, 2)
        
        new_u = F.grid_sample(
            u0_batched, grid,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )
        
        u[n] = new_u.squeeze()
    
    return u


def heat_2d_loss(u, nu, t, return_sol = False):
    # MSE loss of numerical simulation and samples
    # u.shape == (batch_size, n_t, n_x, n_y)
    # nu.shape = (batch_size,)
    sol = torch.zeros_like(u, device=u.device)
    batch_size = u.shape[0]
    for i in tqdm(range(batch_size)):
        sol[i] = simulate_heat(u[i, 0, 0], nu[i], t)
    pde_loss = ((sol - u)**2).mean()

    if return_sol:
        return pde_loss, sol
    else:
        return pde_loss


def simulate_heat(u0, nu, times, dx=2/63, dt=0.0001, steps=100):
    nx, ny = u0.shape
    total_time = times[-1] + (times[-1] - times[-2])
    nt = int(total_time/dt)
    save_every = int(nt/steps)

    if(nu*dt/(dx**2) >= 0.5):
        raise ValueError("Unstable Simulation.")

    all_us = torch.empty(len(times), nx, ny)

    # Save initial condition
    all_us[0] = u0
    u = u0.clone()

    for n in range(nt-1):
        if (n+1) // save_every == len(times):
            break
        un = u.clone()

        # Calculate finite differences for diffusion term
        diff_ux = (torch.roll(un, shifts=(1), dims=(1)) + torch.roll(un, shifts=(-1), dims=(1)) - 2*un)
        diff_uy = (torch.roll(un, shifts=(1), dims=(0)) + torch.roll(un, shifts=(-1), dims=(0)) - 2*un)
        diff_u = diff_ux + diff_uy

        # Calculate updateb
        u = nu*dt*diff_u/dx**2 + u

        if((n+1)%save_every == 0):
            all_us[(n+1) // save_every] = u

    return all_us

