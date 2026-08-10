"""
LLM-Assisted Symbolic Regression — Phase 1: Primitive Set Selection
====================================================================
Demonstrates how PrimitiveSet.from_description() asks an LLM to select
domain-relevant mathematical primitives, compared to using the full
STANDARD preset.

Benchmark: Nguyen-10 — 2·sin(x1)·cos(x2)
Domain description: acoustic standing wave interference

Library features showcased:
- PrimitiveSet.from_description()  — LLM selects primitives from description
- InsufficientContextError         — LLM asks for more context when vague
- LLMClient                        — provider-agnostic LLM wrapper

Requirements:
    pip install symgene[llm]
    export ANTHROPIC_API_KEY=sk-ant-...   # or OPENAI_API_KEY

Run:
    python examples/06_llm_primitive_selection.py
"""
import os
import sys

import numpy as np
from sklearn.metrics import r2_score

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.benchmarks import nguyen10
from symgene.callbacks import GenerationLogger
from symgene.fitness import FitnessEvaluator
from symgene.metrics import rmse, nrmse
from symgene.metrics.regression import mse
from symgene.primitives import STANDARD
from symgene.selection import TournamentSelection


def _check_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        print("Error: set ANTHROPIC_API_KEY or OPENAI_API_KEY before running.")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        sys.exit(1)
    provider = "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "openai"
    return provider


def _make_evolver(pset: PrimitiveSet, label: str, seed: int) -> SymGeneEvolver:
    pop = Population(
        name=label,
        pset=pset,
        n_genes=2, n_genes_max=6, pop_size=60,
        elite_ratio=0.05,
        tree_min=2, tree_max=20, tree_init_max=3, height_max=5,
        cxpb=0.85, cxpb_low=0.55,
        mutpb=0.30, mutpb_low=0.20,
        mutation_weights=[0.3, 1.5, 1.2],
        fitness=FitnessEvaluator(metric=mse),
        selection=TournamentSelection(size=5),
    )
    return SymGeneEvolver(
        populations=[pop],
        n_gen=150,
        seed=seed,
        callbacks=[GenerationLogger(every=50)],
        verbose=0,
    )


def main():
    provider = _check_api_key()

    # ── LLM client ───────────────────────────────────────────────────────────
    from symgene.llm import LLMClient, InsufficientContextError

    client = LLMClient(
        provider=provider,
        model="claude-haiku-4-5-20251001" if provider == "anthropic" else "gpt-4o-mini",
    )

    # ── Data — Nguyen-10: 2·sin(x1)·cos(x2) ─────────────────────────────────
    data = nguyen10(n_train=200, n_test=100, seed=0)
    X_tr, y_tr = data.X_train, data.y_train
    X_te, y_te = data.X_test, data.y_test
    print(f"Benchmark : {data.name}  formula: {data.formula}")
    print(f"Train: {X_tr.shape[0]}  Test: {X_te.shape[0]}")

    # ── Phase 1a: InsufficientContextError demo ──────────────────────────────
    print("\n" + "=" * 60)
    print("Phase 1a — Vague description → InsufficientContextError")
    print("=" * 60)
    try:
        PrimitiveSet.from_description(
            description="predict something with two inputs",
            client=client,
            n_inputs=2,
            feature_names=["x1", "x2"],
        )
    except InsufficientContextError as e:
        print(f"LLM asked for more context:\n  {e.llm_message}")

    # ── Phase 1b: Rich description → domain-relevant pset ───────────────────
    print("\n" + "=" * 60)
    print("Phase 1b — Rich description → LLM selects primitives")
    print("=" * 60)

    DESCRIPTION = (
        "Acoustic standing wave interference pattern produced by two "
        "perpendicular wave sources. The target is the instantaneous "
        "amplitude at a point (x1, x2) in a 2-D medium. The relationship "
        "is periodic in both coordinates and involves wave superposition."
    )

    pset_llm = PrimitiveSet.from_description(
        description=DESCRIPTION,
        client=client,
        n_inputs=2,
        feature_names=["x1", "x2"],
        n_min=4, n_max=10,
    )

    selected = [name for _, _, name, *_ in pset_llm.primitives]
    print(f"LLM selected {len(selected)} primitives: {', '.join(selected)}")

    # ── Standard pset (baseline) ─────────────────────────────────────────────
    pset_std = PrimitiveSet(n_inputs=2, feature_names=["x1", "x2"])
    pset_std.add_from_catalog(STANDARD)
    std_count = len(pset_std.primitives)
    print(f"STANDARD preset has {std_count} primitives")

    # ── Training — both psets, identical hyperparams ─────────────────────────
    print("\n" + "=" * 60)
    print("Training with LLM-selected pset …")
    print("=" * 60)
    ev_llm = _make_evolver(pset_llm, "llm_pset", seed=0)
    res_llm = ev_llm.fit(X_tr, {"llm_pset": y_tr}, X_val=X_te, y_val={"llm_pset": y_te})

    print("\n" + "=" * 60)
    print("Training with STANDARD pset …")
    print("=" * 60)
    ev_std = _make_evolver(pset_std, "std_pset", seed=0)
    res_std = ev_std.fit(X_tr, {"std_pset": y_tr}, X_val=X_te, y_val={"std_pset": y_te})

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for label, res, pset in [
        ("LLM pset", res_llm["llm_pset"], pset_llm),
        ("STANDARD ", res_std["std_pset"], pset_std),
    ]:
        yp_tr = res.predict(X_tr)
        yp_te = res.predict(X_te)
        n_prim = len(pset.primitives)
        print(f"\n[{label}]  primitives={n_prim}  genes={res.n_genes_}")
        print(f"  Expression : {res.best_expression_}")
        print(f"  Train  RMSE={rmse(y_tr, yp_tr):.4f}  R²={r2_score(y_tr, yp_tr):.4f}")
        print(f"  Test   RMSE={rmse(y_te, yp_te):.4f}  R²={r2_score(y_te, yp_te):.4f}")

    print(f"\nTrue formula: {data.formula}")


if __name__ == "__main__":
    main()
