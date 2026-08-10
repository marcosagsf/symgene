"""
LLM-Assisted Symbolic Regression — Phase 2: Interpretation & Concept Library
=============================================================================
Demonstrates how to extract physical meaning from an evolved expression using
LLMContext and the interpret() method, and how to build and evolve a concept
library that accumulates knowledge about successful mathematical structures.

Benchmark: Forrester 1D — (6x − 2)²·sin(12x − 4)
Domain description: aerodynamic drag coefficient vs. Mach number

Library features showcased:
- PopulationResult.interpret()  — LLM interprets each gene term physically
- LLMContext.from_result()      — extracts concepts from best/worst expressions
- LLMContext.evolve()           — refines the concept library with the LLM
- ctx.concepts                  — inspect accumulated mathematical patterns

Requirements:
    pip install symgene[llm]
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY

Run:
    python examples/07_llm_interpret_and_concepts.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.benchmarks import forrester_1d
from symgene.callbacks import GenerationLogger
from symgene.fitness import FitnessEvaluator
from symgene.metrics import rmse
from symgene.metrics.regression import mse
from symgene.metrics.complexity import complexity_penalty
from symgene.primitives import STANDARD
from symgene.selection import TournamentSelection


DESCRIPTION = (
    "Aerodynamic drag coefficient as a function of Mach number for a "
    "transonic airfoil. The coefficient exhibits nonlinear oscillatory "
    "behavior and a sharp local peak near the transonic regime, with "
    "underlying polynomial growth modulated by periodic components."
)
TARGET_NAME = "drag_coefficient"


def _check_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY before running.")
        sys.exit(1)
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"


def main():
    provider = _check_api_key()

    from symgene.llm import LLMClient, LLMContext

    client = LLMClient(
        provider=provider,
        model="claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini",
    )

    # ── Data — Forrester 1D ───────────────────────────────────────────────────
    bench = forrester_1d()
    rng = np.random.default_rng(0)
    X_all = rng.uniform(0.0, 1.0, (300, 1))
    y_all = np.array([bench.fn(X_all[i]) for i in range(300)])
    idx = rng.permutation(300)
    i_tr, i_v, i_te = idx[:200], idx[200:250], idx[250:]
    X_tr, X_val, X_te = X_all[i_tr], X_all[i_v], X_all[i_te]
    y_tr, y_val, y_te = y_all[i_tr], y_all[i_v], y_all[i_te]
    print(f"Benchmark : {bench.name}  formula: {bench.formula}")
    print(f"Train: {X_tr.shape[0]}  Val: {X_val.shape[0]}  Test: {X_te.shape[0]}")

    # ── MGGP training (no LLM in the loop) ───────────────────────────────────
    pset = PrimitiveSet(n_inputs=1, feature_names=["x"])
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=8, alpha=0.1, scale=2.0)

    pop = Population(
        name="forrester",
        pset=pset,
        n_genes=2, n_genes_max=8, pop_size=60,
        elite_ratio=0.05,
        tree_min=2, tree_max=30, tree_init_max=3, height_max=7,
        cxpb=0.90, cxpb_low=0.45,
        mutpb=0.30, mutpb_low=0.20,
        mutation_weights=[0.2, 1.8, 1.0],
        fitness=FitnessEvaluator(metric=mse, penalties=[complexity_penalty(lambda_=1e-4)]),
        selection=TournamentSelection(size=5),
    )

    evolver = SymGeneEvolver(
        populations=[pop],
        n_gen=200,
        seed=0,
        callbacks=[GenerationLogger(every=50)],
        verbose=0,
    )

    print("\nTraining …")
    result = evolver.fit(
        X_tr, {"forrester": y_tr},
        X_val=X_val, y_val={"forrester": y_val},
    )
    print("Training complete.")

    res = result["forrester"]
    yp_te = res.predict(X_te)
    print(f"\nBest expression : {res.best_expression_}")
    print(f"LaTeX           : {res.to_latex()}")
    print(f"Test  RMSE={rmse(y_te, yp_te):.4f}  R²={r2_score(y_te, yp_te):.4f}")

    # ── Phase 2a: Build concept library from result ───────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2a — Building LLMContext from evolution result …")
    print("=" * 60)

    ctx = LLMContext.from_result(
        result=res,
        client=client,
        description=DESCRIPTION,
        target_name=TARGET_NAME,
        n_good=5,
        n_bad=3,
        n_concepts=4,
    )

    print(f"Initial concepts ({len(ctx)} extracted):")
    for i, c in enumerate(ctx.concepts, 1):
        print(f"  {i}. {c}")

    # ── Phase 2b: Evolve the concept library ─────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2b — Evolving concept library (2 iterations) …")
    print("=" * 60)

    ctx.evolve(client, n_concepts=3, n_iterations=2)

    print(f"Evolved concepts ({len(ctx)} total):")
    for i, c in enumerate(ctx.concepts, 1):
        print(f"  {i}. {c}")

    # ── Phase 2c: Interpret the best expression ───────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2c — Physical interpretation of best expression …")
    print("=" * 60)

    interpretation = res.interpret(
        client=client,
        description=DESCRIPTION,
        target_name=TARGET_NAME,
    )

    print(interpretation)
    print(f"\nTrue formula: {bench.formula}")


if __name__ == "__main__":
    main()
