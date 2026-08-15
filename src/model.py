"""Modelo de detecção com métricas de evento raro.

Com 0,296% de fraude, acurácia é inútil e AUC-ROC é enganosa: a taxa de
falso positivo tem no denominador 2,7 milhões de transações legítimas,
então ela quase não se move e a curva parece excelente mesmo quando a
fila de revisão está cheia de lixo.

O que se acompanha aqui:

    PR-AUC          área sob precisão-recall, sensível a evento raro
    precisão@k      o que a mesa de análise realmente experimenta
    Brier           a probabilidade precisa ser probabilidade, porque a
                    triagem ordena por p x valor — e esse produto não
                    sobrevive a uma distorção monotônica do escore
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from . import config

LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=config.RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
)

# Sem `scale_pos_weight` nem `is_unbalance`: os dois reescalam o escore e
# arruinam a calibração. A triagem ordena por p x valor, e esse produto
# só faz sentido se p for probabilidade de verdade. Preferimos tratar o
# desbalanceamento na métrica e na fila, não distorcendo a saída.


@dataclass
class FraudModel:
    booster: LGBMClassifier
    features: list[str]
    nome: str

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.booster.predict_proba(X[self.features])[:, 1]


def precisao_recall_em_k(
    y: np.ndarray, p: np.ndarray, k: int
) -> tuple[float, float]:
    """Precisão e recall considerando só as k transações mais suspeitas."""
    if k <= 0:
        return 0.0, 0.0
    k = min(k, len(p))
    topo = np.argsort(-p)[:k]
    tp = float(y[topo].sum())
    return tp / k, tp / max(y.sum(), 1)


def evaluate(
    y: np.ndarray,
    p: np.ndarray,
    valor: np.ndarray | None = None,
    ks: tuple[int, ...] = (100, 500, 1000, 5000),
) -> dict[str, float]:
    y = np.asarray(y)
    m = {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "prevalencia": float(y.mean()),
    }
    for k in ks:
        prec, rec = precisao_recall_em_k(y, p, k)
        m[f"precisao@{k}"] = prec
        m[f"recall@{k}"] = rec

    if valor is not None:
        valor = np.asarray(valor)
        total = valor[y == 1].sum()
        for k in ks:
            topo = np.argsort(-p)[: min(k, len(p))]
            capturado = valor[topo][y[topo] == 1].sum()
            m[f"vdr@{k}"] = float(capturado / total) if total else 0.0
    return m


def train(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    nome: str,
) -> FraudModel:
    """Treina com parada antecipada pela PR-AUC da validação.

    `eval_metric="average_precision"` importa: parar por AUC-ROC em dado
    tão desbalanceado escolhe o número errado de árvores, porque a métrica
    satura antes de a precisão no topo da fila parar de melhorar.
    """
    booster = LGBMClassifier(**LGBM_PARAMS)
    booster.fit(
        X_tr,
        y_tr,
        eval_X=X_va,
        eval_y=y_va,
        eval_metric="average_precision",
        callbacks=[early_stopping(150, verbose=False), log_evaluation(0)],
    )
    return FraudModel(booster=booster, features=list(X_tr.columns), nome=nome)


def tabela_calibracao(y: np.ndarray, p: np.ndarray, n_bins: int = 8) -> pd.DataFrame:
    """Probabilidade prevista contra frequência observada, por faixa.

    Com evento raro os quantis colapsam (a maioria das transações tem
    p ~ 0), então as faixas são fixas em escala logarítmica em vez de
    quantílicas.
    """
    bordas = np.array([0, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.2, 0.5, 1.0])
    faixa = pd.cut(p, bins=bordas, include_lowest=True)
    return (
        pd.DataFrame({"faixa": faixa, "y": y, "p": p})
        .groupby("faixa", observed=True)
        .agg(
            transacoes=("y", "size"),
            p_media=("p", "mean"),
            taxa_observada=("y", "mean"),
            fraudes=("y", "sum"),
        )
        .reset_index()
    )


def importancia(model: FraudModel, X: pd.DataFrame, amostra: int = 50_000) -> pd.DataFrame:
    """Importância por |SHAP| médio, via TreeSHAP interno do LightGBM."""
    sub = X.sample(min(amostra, len(X)), random_state=0)
    contrib = np.asarray(model.booster.predict(sub[model.features], pred_contrib=True))
    imp = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=model.features)
    imp = imp.sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "impacto_medio": imp.to_numpy()})
