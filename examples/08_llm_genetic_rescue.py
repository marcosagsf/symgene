"""
LLM-Assisted Symbolic Regression — Full Pipeline (Phases 1 + 2 + 3)
=====================================================================
Demonstrates all three LLM integration modes in a single workflow applied
to the Dittus-Boelter heat transfer correlation:

  Nu = 0.023 · Re^0.8 · Pr^0.4

Phase 1 — PrimitiveSet.from_description()
    The LLM selects primitives motivated by power-law heat transfer physics.

Phase 3 — Genetic Rescue (inside the evolutionary loop)
    When evolution stagnates, the LLM rehabilitates the worst individuals
    by generating new gene expressions guided by the best individual's
    structure and an accumulated concept library.

Phase 2 — LLMContext + interpret()
    After training, physical meaning is extracted from the evolved expression
    and a concept library is built for potential use in future runs.

Library features showcased:
- PrimitiveSet.from_description()      — Phase 1
- SymGeneEvolver LLM rescue params     — Phase 3
- LLMContext.from_result()             — Phase 2
- PopulationResult.interpret()         — Phase 2

Requirements:
    pip install symgene[llm]
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY

Run:
    python examples/08_llm_genetic_rescue.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.benchmarks import dittus_boelter
from symgene.callbacks import GenerationLogger
from symgene.fitness import FitnessEvaluator
from symgene.metrics import rmse, nrmse
from symgene.metrics.regression import mse
from symgene.metrics.complexity import complexity_penalty
from symgene.selection import TournamentSelection


DESCRIPTION = (
    "Convective heat transfer in fully turbulent pipe flow. "
    "The Nusselt number (Nu) quantifies the ratio of convective to conductive "
    "heat transfer and follows a power-law relationship with the Reynolds number "
    "(Re, dimensionless flow inertia-to-viscosity ratio) and the Prandtl number "
    "(Pr, dimensionless momentum-to-thermal diffusivity ratio). "
    "The Dittus-Boelter correlation predicts Nu via a product of power-law terms "
    "in Re and Pr with fractional exponents close to 0.8 and 0.4 respectively."
)
TARGET_NAME = "Nu"


def _check_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    return "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"


def main():
    provider = _check_api_key()

    from symgene.llm import LLMClient, LLMContext, InsufficientContextError

    client = LLMClient(
        provider=provider,
        model="claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini",
    )

    # ── Data — Dittus-Boelter ────────────────────────────────────────────────
    data = dittus_boelter(n_train=200, n_test=80, seed=0)
    X_tr, y_tr = data.X_train, data.y_train
    X_te, y_te = data.X_test, data.y_test
    print(f"Benchmark : {data.name}")
    print(f"Formula   : {data.formula}")
    print(f"Inputs    : {data.feature_names}")
    print(f"Re range  : [{X_tr[:, 0].min():.0f}, {X_tr[:, 0].max():.0f}]")
    print(f"Pr range  : [{X_tr[:, 1].min():.2f}, {X_tr[:, 1].max():.1f}]")
    print(f"Nu range  : [{y_tr.min():.1f}, {y_tr.max():.1f}]")
    print(f"Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

    # ── Phase 1: LLM selects primitives ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1 — LLM selects primitives from domain description")
    print("=" * 60)

    pset = PrimitiveSet.from_description(
        description=DESCRIPTION,
        client=client,
        n_inputs=2,
        feature_names=data.feature_names,
        n_min=5, n_max=12,
    )

    selected = [name for _, _, name, *_ in pset.primitives]
    print(f"LLM selected {len(selected)} primitives: {', '.join(selected)}")

    # Ephemerals for power-law exponents (the key to discovering 0.8 and 0.4)
    pset.add_ephemeral("exp_coef", dist="uniform", low=0.1, high=1.5)
    pset.set_squash(lim=12, alpha=0.08, scale=2.0)

    # ── Phase 3: Fit with Genetic Rescue ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 3 — Training with Genetic Rescue (stagnation trigger)")
    print("=" * 60)

    # Build an initial concept library (empty — will populate from result)
    ctx = LLMContext(
        description=DESCRIPTION,
        target_name=TARGET_NAME,
    )

    pop = Population(
        name="Nu",
        pset=pset,
        n_genes=2, n_genes_max=8, pop_size=60,
        elite_ratio=0.05,
        tree_min=2, tree_max=25, tree_init_max=3, height_max=6,
        cxpb=0.85, cxpb_low=0.50,
        mutpb=0.30, mutpb_low=0.20,
        mutation_weights=[0.3, 1.5, 1.2],
        fitness=FitnessEvaluator(
            metric=mse,
            penalties=[complexity_penalty(lambda_=5e-5)],
        ),
        selection=TournamentSelection(size=5),
    )

    evolver = SymGeneEvolver(
        populations=[pop],
        n_gen=250,
        seed=0,
        callbacks=[GenerationLogger(every=50)],
        verbose=1,
        # ── LLM Genetic Rescue ──────────────────────────────────
        llm_rescue=True,
        llm_client=client,
        llm_context={"Nu": ctx},
        llm_rescue_trigger="stagnation",
        llm_rescue_level="gene",
        llm_rescue_fraction=0.15,
        llm_stagnation_patience=30,
        llm_max_retries=3,
    )

    result = evolver.fit(
        X_tr, {"Nu": y_tr},
        X_val=X_te, y_val={"Nu": y_te},
    )
    print("Training complete.")

    res = result["Nu"]
    yp_tr = res.predict(X_tr)
    yp_te = res.predict(X_te)

    print(f"\nBest expression : {res.best_expression_}")
    print(f"Genes           : {res.n_genes_}")
    print(f"LaTeX           : {res.to_latex()}")
    print(f"Train  RMSE={rmse(y_tr, yp_tr):.4f}  NRMSE={nrmse(y_tr, yp_tr):.4f}  R²={r2_score(y_tr, yp_tr):.4f}")
    print(f"Test   RMSE={rmse(y_te, yp_te):.4f}  NRMSE={nrmse(y_te, yp_te):.4f}  R²={r2_score(y_te, yp_te):.4f}")

    # ── Phase 2: Build concept library + interpret ────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 2 — Concept library + physical interpretation")
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

    print(f"Concepts extracted ({len(ctx)}):")
    for i, c in enumerate(ctx.concepts, 1):
        print(f"  {i}. {c}")

    print("\nPhysical interpretation of best expression:")
    interpretation = res.interpret(
        client=client,
        description=DESCRIPTION,
        target_name=TARGET_NAME,
    )
    print(interpretation)

    print(f"\nTrue formula : {data.formula}")
    print(
        "\nNote: the LLMContext built here can be passed to a second run via\n"
        "  llm_context={'Nu': ctx}\n"
        "so Genetic Rescue in future runs benefits from accumulated knowledge."
    )


if __name__ == "__main__":
    main()
