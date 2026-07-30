import numpy as np
from symgene.optimization.base import Optimizer, OptimResult


class PSOOptimizer(Optimizer):
    def __init__(
        self,
        n_particles: int = 30,
        n_iter: int = 100,
        w: float = 0.7,       # inertia weight
        c1: float = 1.5,      # cognitive coefficient
        c2: float = 1.5,      # social coefficient
        verbose: int = 0,
        seed: int | None = None,
    ):
        self.n_particles = n_particles
        self.n_iter = n_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.verbose = verbose
        self.seed = seed

    def optimize(self, fn, bounds, seed=None):
        rng = np.random.default_rng(seed)
        n_dims = len(bounds)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        # Initialize positions and velocities
        pos = rng.uniform(lo, hi, (self.n_particles, n_dims))
        vel = rng.uniform(-(hi - lo), (hi - lo), (self.n_particles, n_dims))

        # Evaluate initial positions
        f_vals = np.array([fn(pos[i]) for i in range(self.n_particles)])
        n_eval = self.n_particles

        pbest_pos = pos.copy()
        pbest_f = f_vals.copy()

        gbest_idx = np.argmin(pbest_f)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_f = float(pbest_f[gbest_idx])

        history = [gbest_f]

        for it in range(self.n_iter):
            r1 = rng.uniform(0, 1, (self.n_particles, n_dims))
            r2 = rng.uniform(0, 1, (self.n_particles, n_dims))

            vel = (
                self.w * vel
                + self.c1 * r1 * (pbest_pos - pos)
                + self.c2 * r2 * (gbest_pos - pos)
            )
            pos = np.clip(pos + vel, lo, hi)

            f_vals = np.array([fn(pos[i]) for i in range(self.n_particles)])
            n_eval += self.n_particles

            improved = f_vals < pbest_f
            pbest_pos[improved] = pos[improved]
            pbest_f[improved] = f_vals[improved]

            gbest_idx = np.argmin(pbest_f)
            if pbest_f[gbest_idx] < gbest_f:
                gbest_f = float(pbest_f[gbest_idx])
                gbest_pos = pbest_pos[gbest_idx].copy()

            history.append(gbest_f)

            if self.verbose and (it + 1) % 10 == 0:
                print(f"  PSO iter {it+1}/{self.n_iter}  best={gbest_f:.6f}")

        return OptimResult(
            x_best=gbest_pos,
            f_best=gbest_f,
            history=history,
            n_eval=n_eval,
        )
