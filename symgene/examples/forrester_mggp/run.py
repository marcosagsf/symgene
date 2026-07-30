"""
Forrester 1D & Koza-1 — Multi-population MGGP Demo

Demonstra o SymGeneEvolver com duas populações simultâneas sobre funções
analíticas 1D. Como as funções são conhecidas analiticamente, geramos
tantos pontos quanto necessário.

Parâmetros demonstrados:
  n_genes_max      — número máximo de genes por indivíduo (mutação add respeita esse limite)
  elite_ratio      — fração da população preservada intacta a cada geração (elitismo)
  tree_min         — mínimo de nós por árvore GP
  tree_max         — máximo de nós por árvore GP (controla bloat)
  height_max       — profundidade máxima de aninhamento
  cxpb             — probabilidade de crossover ocorrer por geração
  cxpb_low         — fração low-order (subtree) vs high-order (troca de gene inteiro)
  cross_population — ativa intercâmbio de genes entre populações distintas
  cxpb_inter       — probabilidade de crossover inter-populacional por geração
  FitnessEvaluator — fitness com penalidade de complexidade embutida
  complexity_penalty — penaliza modelos com muitos nós
  EarlyStopping    — interrompe a evolução quando não há melhora
  GenerationLogger — imprime métricas a cada N gerações
  add_custom       — primitiva matemática definida pelo usuário

Run: python -m symgene.examples.forrester_mggp.run
"""
import numpy as np

from symgene import PrimitiveSet, Population, SymGeneEvolver
from symgene.primitives import STANDARD
from symgene.fitness import FitnessEvaluator
from symgene.metrics.regression import mse
from symgene.metrics.complexity import complexity_penalty
from symgene.metrics import rmse, nrmse
from symgene.selection import TournamentSelection
from symgene.callbacks import EarlyStopping, GenerationLogger
from symgene.benchmarks import forrester_1d


def main():
    bench = forrester_1d()

    # Ambas as funções avaliadas no mesmo domínio [0, 1]
    rng = np.random.default_rng(0)
    X_train = np.sort(rng.uniform(0, 1, (200, 1)), axis=0)
    X_test  = np.linspace(0, 1, 500).reshape(-1, 1)

    y_f_train = np.array([bench.fn(X_train[i]) for i in range(200)])
    y_f_test  = np.array([bench.fn(X_test[i])  for i in range(500)])

    # Koza-1: x^4 + x^3 + x^2 + x — avaliado no mesmo X
    koza_fn   = lambda x: x**4 + x**3 + x**2 + x
    y_k_train = koza_fn(X_train[:, 0])
    y_k_test  = koza_fn(X_test[:, 0])

    # PrimitiveSet compartilhado entre as duas populações
    pset = PrimitiveSet(n_inputs=1, feature_names=["x"])
    pset.add_from_catalog(STANDARD)
    pset.set_squash(lim=8, alpha=0.1, scale=2.0)
    # primitiva customizada: bump gaussiano centrado em 0.5
    # demonstra como adicionar funções específicas do domínio
    pset.add_custom(
        fn=lambda x: float(np.exp(-((x - 0.5) ** 2) / 0.1)),
        arity=1,
        name="bump",
    )

    pop_forrester = Population(
        name="forrester",
        pset=pset,
        n_genes=4,
        n_genes_max=10,       # permite crescer até 10 genes via mutação add
        pop_size=50,
        elite_ratio=0.05,     # 5% da população preservada intacta (elitismo)
        tree_min=10,          # árvores com pelo menos 10 nós
        tree_max=40,          # máximo de 40 nós por árvore (evita bloat)
        height_max=5,         # máximo de 5 níveis de aninhamento
        cxpb=0.90,            # 90% de chance de crossover por geração
        cxpb_low=0.40,        # 40% subtree (low), 60% troca de gene inteiro (high)
        fitness=FitnessEvaluator(
            metric=mse,
            penalties=[complexity_penalty(lambda_=1e-4)],  # penaliza modelos complexos
        ),
        selection=TournamentSelection(size=5),
    )

    pop_koza = Population(
        name="koza1",
        pset=pset,
        n_genes=4,
        n_genes_max=8,
        pop_size=50,
        elite_ratio=0.05,
        tree_min=5,
        tree_max=30,
        height_max=4,
        cxpb=0.85,
        cxpb_low=0.50,
        fitness=FitnessEvaluator(
            metric=mse,
            penalties=[complexity_penalty(lambda_=5e-4)],
        ),
        selection=TournamentSelection(size=5),
    )

    evolver = SymGeneEvolver(
        populations=[pop_forrester, pop_koza],
        n_gen=80,
        cross_population=True,   # genes migram entre as duas populações
        cxpb_inter=0.025,        # 2.5% de chance de crossover inter-populacional
        seed=0,
        callbacks=[
            GenerationLogger(every=20),
            EarlyStopping(monitor="forrester_train_mse", patience=15),
        ],
        verbose=0,
    )

    results = evolver.fit(
        X_train,
        {"forrester": y_f_train, "koza1": y_k_train},
        X_val=X_test,
        y_val={"forrester": y_f_test, "koza1": y_k_test},
    )

    for pop_name, y_te in [("forrester", y_f_test), ("koza1", y_k_test)]:
        res = results[pop_name]
        y_pred = res.predict(X_test)
        print(f"\n=== {pop_name.upper()} ===")
        print(f"  Genes     : {res.n_genes_}")
        print(f"  Expression: {res.best_expression_}")
        print(f"  RMSE      : {rmse(y_te, y_pred):.4f}")
        print(f"  NRMSE     : {nrmse(y_te, y_pred):.4f}")


if __name__ == "__main__":
    main()
