"""
Optimization benchmark functions with known global optima.
Used to validate surrogate-assisted optimization workflows.
"""
from typing import NamedTuple, Callable
import numpy as np


class OptimBenchmark(NamedTuple):
    fn: Callable[[np.ndarray], float]   # objective (to minimize)
    bounds: list[tuple[float, float]]   # search domain per dimension
    x_opt: np.ndarray                   # known global optimum location
    f_opt: float                        # known global optimum value
    name: str
    formula: str                        # human-readable formula string
    n_inputs: int


def forrester_1d(noise: float = 0.0, seed: int = 0) -> OptimBenchmark:
    """
    Forrester et al. (2008) 1D benchmark.
    f(x) = (6x - 2)^2 * sin(12x - 4),  x in [0, 1]
    Global minimum at x ≈ 0.7572, f ≈ -6.0207
    Classic surrogate modeling benchmark.
    """
    rng = np.random.default_rng(seed)

    def fn(x: np.ndarray) -> float:
        xv = float(x[0])
        val = (6 * xv - 2) ** 2 * np.sin(12 * xv - 4)
        if noise > 0:
            val += rng.normal(0, noise)
        return float(val)

    return OptimBenchmark(
        fn=fn,
        bounds=[(0.0, 1.0)],
        x_opt=np.array([0.7572]),
        f_opt=-6.0207,
        name="Forrester1D",
        formula="(6x - 2)^2 * sin(12x - 4)",
        n_inputs=1,
    )


def himmelblau_2d() -> OptimBenchmark:
    """
    Himmelblau's function.
    f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2,  x,y in [-5, 5]
    Four global minima, all at f = 0:
      (3, 2), (-2.805, 3.131), (-3.779, -3.283), (3.584, -1.848)
    """
    def fn(x: np.ndarray) -> float:
        return float((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)

    return OptimBenchmark(
        fn=fn,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        x_opt=np.array([3.0, 2.0]),
        f_opt=0.0,
        name="Himmelblau2D",
        formula="(x^2 + y - 11)^2 + (x + y^2 - 7)^2",
        n_inputs=2,
    )


def schwefel_2d() -> OptimBenchmark:
    """
    Schwefel function (2D).
    f(x, y) = 418.9829*2 - x*sin(sqrt(|x|)) - y*sin(sqrt(|y|))
    Global minimum at (420.9687, 420.9687), f ≈ 0.
    Highly multimodal — many deceptive local minima far from the global optimum.
    Domain: x, y in [-500, 500]
    """
    def fn(x: np.ndarray) -> float:
        return float(418.9829 * 2 - x[0] * np.sin(np.sqrt(abs(x[0])))
                     - x[1] * np.sin(np.sqrt(abs(x[1]))))

    return OptimBenchmark(
        fn=fn,
        bounds=[(-500.0, 500.0), (-500.0, 500.0)],
        x_opt=np.array([420.9687, 420.9687]),
        f_opt=0.0,
        name="Schwefel2D",
        formula="418.9829*2 - x*sin(sqrt(|x|)) - y*sin(sqrt(|y|))",
        n_inputs=2,
    )


def ackley_2d() -> OptimBenchmark:
    """
    Ackley function (2D).
    Global minimum at (0, 0), f = 0.
    Many local minima — challenging for gradient-free optimizers.
    x, y in [-5, 5]
    """
    def fn(x: np.ndarray) -> float:
        a, b, c = 20.0, 0.2, 2 * np.pi
        d = len(x)
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(c * x))
        return float(
            -a * np.exp(-b * np.sqrt(sum_sq / d))
            - np.exp(sum_cos / d)
            + a + np.e
        )

    return OptimBenchmark(
        fn=fn,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        x_opt=np.array([0.0, 0.0]),
        f_opt=0.0,
        name="Ackley2D",
        formula="-20*exp(-0.2*sqrt(0.5*(x^2+y^2))) - exp(0.5*(cos(2pi*x)+cos(2pi*y))) + 20 + e",
        n_inputs=2,
    )
