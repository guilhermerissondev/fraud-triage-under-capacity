"""Pipeline completo: auditoria do benchmark -> dois modelos -> triagem.

Uso:
    python -m scripts.train
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, features, leakage, model, triage  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
})
AZUL, VERMELHO, VERDE, CINZA = "#1f4e79", "#c0392b", "#27ae60", "#7f8c8d"


def log(m: str) -> None:
    print(f"  {m}", flush=True)


def main() -> None:
    print("\n=== 1. Dados ===")
    df = data.load()
    log(f"{len(df):,} transacoes | fraude {df[config.TARGET].mean():.4%}".replace(",", "."))
    resumo = data.resumo(df)
    resumo.to_csv(config.REPORTS / "resumo_por_tipo.csv")
    log("fraude so em TRANSFER e CASH_OUT; os outros tres tipos tem zero")

    print("\n=== 2. Auditoria do benchmark ===")
    escopo = features.filtrar_escopo(df)
    log(f"escopo TRANSFER + CASH_OUT: {len(escopo):,} transacoes | "
        f"fraude {escopo[config.TARGET].mean():.4%}".replace(",", "."))

    # Artefato temporal: fraude constante, tráfego legítimo colapsando.
    vol = features.volume_por_dia(escopo)
    vol.to_csv(config.REPORTS / "volume_por_dia.csv", index=False)
    integros = features.dias_integros(escopo)
    descartados = [int(d) for d in vol["dia"] if d not in integros]
    print()
    log("Artefato temporal — fraude constante, trafego legitimo colapsando:")
    log(f"  fraudes/dia: min {vol['fraudes'].min()}, max {vol['fraudes'].max()}, "
        f"mediana {vol['fraudes'].median():.0f} — praticamente constante nos 31 dias")
    log(f"  transacoes/dia: de {vol['transacoes'].max():,} a {vol['transacoes'].min():,}"
        .replace(",", "."))
    pior = vol.loc[vol["prevalencia"].idxmax()]
    log(f"  pior dia: dia {int(pior['dia'])} com {int(pior['transacoes'])} transacoes "
        f"e prevalencia de {pior['prevalencia']:.1%}")
    log(f"  dias descartados por volume < {config.VOLUME_MINIMO_DIARIO:,}: {descartados}"
        .replace(",", "."))
    log(f"  restam {len(integros)} dias integros: {integros}")
    print()
    print(leakage.relatorio(escopo))
    leakage.auditar(escopo).to_csv(config.REPORTS / "artefatos.csv", index=False)
    regra = leakage.desempenho_da_regra(escopo)
    (config.REPORTS / "regra_do_gerador.json").write_text(
        json.dumps(regra, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== 3. Features ===")
    X_h, y, aux = features.build(df)
    X_c, _, _ = features.build_contaminado(df)
    log(f"honestas: {X_h.shape[1]} features | contaminadas: {X_c.shape[1]}")
    log("fora do modelo honesto: saldos de origem e destino")
    log("agregados por conta destino sao causais (so passado)")

    print("\n=== 4. Split temporal ===")
    (Xh_tr, y_tr, a_tr), (Xh_va, y_va, a_va), (Xh_te, y_te, a_te) = features.split_temporal(X_h, y, aux)
    (Xc_tr, _, _), (Xc_va, _, _), (Xc_te, _, _) = features.split_temporal(X_c, y, aux)
    for nome, yy, aa in (("treino", y_tr, a_tr), ("validacao", y_va, a_va), ("teste", y_te, a_te)):
        log(f"{nome:<10} {len(yy):>9,} transacoes | dias {aa['dia'].min():>2}-{aa['dia'].max():<2} | "
            f"fraude {yy.mean():.4%} ({int(yy.sum()):,})".replace(",", "."))
    prevs = [y_tr.mean(), y_va.mean(), y_te.mean()]
    log(f"prevalencia estavel entre as particoes: {max(prevs)/min(prevs):.2f}x de variacao")

    print("\n=== 5. Modelos ===")
    m_cont = model.train(Xc_tr, y_tr, Xc_va, y_va, "com atalho")
    m_hon = model.train(Xh_tr, y_tr, Xh_va, y_va, "honesto")

    valor_te = a_te["valor"].to_numpy()
    p_cont = m_cont.predict_proba(Xc_te)
    p_hon = m_hon.predict_proba(Xh_te)

    av_cont = model.evaluate(y_te.to_numpy(), p_cont, valor_te)
    av_hon = model.evaluate(y_te.to_numpy(), p_hon, valor_te)

    comp = pd.DataFrame({"com atalho": av_cont, "honesto": av_hon})
    comp.to_csv(config.REPORTS / "comparacao_modelos.csv")
    print()
    linhas = ["roc_auc", "pr_auc", "brier", "precisao@100", "precisao@1000",
              "recall@1000", "vdr@1000"]
    print(comp.loc[linhas].to_string(float_format=lambda v: f"{v:.4f}"))

    print()
    log(f"AUC-ROC quase identica ({av_cont['roc_auc']:.4f} vs {av_hon['roc_auc']:.4f}) "
        "enquanto a PR-AUC despenca —")
    log("é exatamente por isso que AUC-ROC nao serve para evento raro.")

    print("\n=== 6. Calibracao do modelo honesto ===")
    tab = model.tabela_calibracao(y_te.to_numpy(), p_hon)
    tab.to_csv(config.REPORTS / "calibracao.csv", index=False)
    print(tab.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== 7. Triagem sob capacidade ===")
    dia_te = a_te["dia"].to_numpy()
    n_dias = len(np.unique(dia_te))
    log(f"{n_dias} dias no teste | {len(y_te)/n_dias:,.0f} transacoes/dia | "
        f"{y_te.sum()/n_dias:.1f} fraudes/dia".replace(",", "."))

    cmp_crit = triage.comparar_criterios(
        p_hon, valor_te, y_te.to_numpy(), dia_te, capacidade_diaria=config.CAPACIDADE_DIARIA
    )
    cmp_crit.to_csv(config.REPORTS / "criterios_de_fila.csv")
    print()
    vis = cmp_crit[["fraudes_pegas", "recall_contagem", "vdr", "valor_evitado",
                    "beneficio_liquido", "ganho_pct"]].copy()
    for c in ("valor_evitado", "beneficio_liquido"):
        vis[c] = vis[c].map(triage.formatar)
    for c in ("recall_contagem", "vdr"):
        vis[c] = vis[c].map(lambda v: f"{v:.2%}")
    vis["ganho_pct"] = vis["ganho_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1%}")
    print(vis.to_string())

    curva = triage.curva_capacidade(p_hon, valor_te, y_te.to_numpy(), dia_te)
    curva.to_csv(config.REPORTS / "curva_capacidade.csv", index=False)
    cap_ot = triage.capacidade_otima(curva)
    log(f"capacidade que maximiza o beneficio liquido: {cap_ot} alertas/dia")

    print("\n=== 8. Importancia ===")
    imp = model.importancia(m_hon, Xh_te)
    imp.to_csv(config.REPORTS / "importancia.csv", index=False)
    log("top 6: " + ", ".join(imp["feature"].head(6)))

    print("\n=== 9. Figuras e artefatos ===")
    _figuras(y_te.to_numpy(), p_cont, p_hon, valor_te, curva, imp, cap_ot)
    joblib.dump(m_hon, config.MODELS / "modelo_honesto.joblib")
    pd.DataFrame({
        "p": p_hon, "valor": valor_te, "fraude": y_te.to_numpy(),
        "dia": dia_te, "hora": a_te["hora"].to_numpy(), "tipo": a_te["tipo"].to_numpy(),
    }).to_parquet(config.DATA_PROCESSED / "teste_scorado.parquet")
    log(f"figuras em {config.FIGURES}")

    melhor = cmp_crit.loc["valor_em_risco"]
    pior = cmp_crit.loc["probabilidade"]
    metricas = {
        "dataset": {
            "transacoes_totais": int(len(df)),
            "escopo": int(len(escopo)),
            "prevalencia_escopo": float(escopo[config.TARGET].mean()),
        },
        "regra_do_gerador": regra,
        "modelos": {"com_atalho": av_cont, "honesto": av_hon},
        "triagem": {
            "capacidade_diaria": config.CAPACIDADE_DIARIA,
            "capacidade_otima": cap_ot,
            "por_probabilidade": {k: float(v) for k, v in pior.items() if k != "criterio"},
            "por_valor_em_risco": {k: float(v) for k, v in melhor.items() if k != "criterio"},
        },
    }
    (config.REPORTS / "metricas.json").write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print(f"  Regra de duas linhas, sem modelo:  recall {regra['recall']:.2%} | "
          f"precisao {regra['precisao']:.2%}")
    print(f"  Modelo com o atalho:               PR-AUC {av_cont['pr_auc']:.4f}")
    print(f"  Modelo honesto:                    PR-AUC {av_hon['pr_auc']:.4f}")
    print()
    print(f"  Fila de {config.CAPACIDADE_DIARIA} alertas/dia, ordenada por probabilidade: "
          f"{triage.formatar(pior['valor_evitado'])} evitados")
    print(f"  Mesma fila, ordenada por valor em risco:      "
          f"{triage.formatar(melhor['valor_evitado'])} evitados")
    ganho = melhor["beneficio_liquido"] - pior["beneficio_liquido"]
    print(f"  Ganho: {triage.formatar(ganho)} — mesmo modelo, so a ordem da fila mudou.")
    print("=" * 76 + "\n")


def _figuras(y, p_cont, p_hon, valor, curva, imp, cap_ot) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    # -- ROC vs PR: a ilusao e a realidade --------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for p, cor, rot in ((p_cont, CINZA, "com atalho"), (p_hon, AZUL, "honesto")):
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, color=cor, lw=1.8, label=rot)
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, color=cor, lw=1.8, label=rot)
    axes[0].plot([0, 1], [0, 1], "--", color=VERMELHO, lw=1)
    axes[0].set_xlabel("Falso positivo"); axes[0].set_ylabel("Verdadeiro positivo")
    axes[0].set_title("AUC-ROC: as duas parecem otimas")
    axes[1].axhline(y.mean(), color=VERMELHO, ls="--", lw=1, label="acaso")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precisao")
    axes[1].set_title("Precisao-Recall: a diferenca aparece")
    for a in axes:
        a.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "roc_vs_pr.png"); plt.close(fig)

    # -- Curva de capacidade ---------------------------------------------
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(curva["capacidade_diaria"], curva["beneficio_liquido"], color=AZUL, lw=2)
    ax.axvline(cap_ot, color=VERDE, ls=":", lw=1.5, label=f"otimo ({cap_ot}/dia)")
    ax.axvline(config.CAPACIDADE_DIARIA, color=VERMELHO, ls="--", lw=1.3,
               label=f"mesa atual ({config.CAPACIDADE_DIARIA}/dia)")
    ax.axhline(0, color=CINZA, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Capacidade da mesa (alertas revisados por dia)")
    ax.set_ylabel(f"Beneficio liquido ({config.MOEDA})")
    ax.set_title("Contratar mais um analista se paga?")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "curva_capacidade.png"); plt.close(fig)

    # -- VDR vs recall de contagem ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(curva["capacidade_diaria"], curva["recall_contagem"] * 100,
            color=CINZA, lw=1.8, label="recall de contagem")
    ax.plot(curva["capacidade_diaria"], curva["vdr"] * 100,
            color=AZUL, lw=2, label="VDR (fracao do valor)")
    ax.set_xscale("log")
    ax.set_xlabel("Capacidade da mesa (alertas/dia)"); ax.set_ylabel("%")
    ax.set_title("Contar fraudes e barrar prejuizo nao sao a mesma coisa")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "vdr_vs_recall.png"); plt.close(fig)

    # -- Importancia ------------------------------------------------------
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(top["feature"], top["impacto_medio"], color=AZUL)
    ax.set_xlabel("|SHAP| medio (log-odds)")
    ax.set_title("O que o modelo honesto usa")
    ax.grid(axis="y", alpha=0)
    fig.tight_layout(); fig.savefig(config.FIGURES / "importancia.png"); plt.close(fig)


if __name__ == "__main__":
    main()
