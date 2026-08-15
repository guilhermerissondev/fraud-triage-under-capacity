"""Auditoria do benchmark.

O PaySim é o dataset mais usado em portfólio de antifraude, e a maior
parte dos notebooks reporta AUC acima de 0,99. Este módulo mostra por quê,
e por que esse número não significa nada.

O simulador gera fraude executando um roteiro fixo: assume a conta e
transfere **exatamente o saldo inteiro**. Isso deixa uma assinatura
aritmética que nenhuma fraude real tem — em dado de verdade, o valor de
uma transferência fraudulenta não bate ao centavo com o saldo anterior.

Um modelo treinado com as colunas de saldo não aprende fraude. Aprende a
detectar o gerador.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Colunas que carregam a assinatura do simulador. Ficam fora do modelo
# honesto; entram apenas no modelo-espelho que reproduz o resultado
# inflado da literatura de notebooks.
COLUNAS_CONTAMINADAS = [
    "saldo_origem_antes",
    "saldo_origem_depois",
    "saldo_destino_antes",
    "saldo_destino_depois",
]


def regra_do_gerador(df: pd.DataFrame) -> np.ndarray:
    """A regra de duas linhas que dispensa aprendizado de máquina.

    Conta drenada ao centavo e zerada em seguida. É o roteiro do PaySim
    escrito como condição booleana.
    """
    return (
        np.isclose(df["saldo_origem_antes"], df["valor"])
        & (df["saldo_origem_depois"] == 0)
    ).to_numpy()


def auditar(df: pd.DataFrame) -> pd.DataFrame:
    """Mede cada artefato: prevalência em fraude contra prevalência em
    transação legítima."""
    f = df[df[config.TARGET] == 1]
    l = df[df[config.TARGET] == 0]

    def erro_contabil(x):
        return x["saldo_origem_antes"] - x["valor"] - x["saldo_origem_depois"]

    testes = {
        "conta drenada ao centavo (saldo_antes == valor)": (
            np.isclose(f["saldo_origem_antes"], f["valor"]),
            np.isclose(l["saldo_origem_antes"], l["valor"]),
        ),
        "origem zerada apos a transacao": (
            f["saldo_origem_depois"] == 0,
            l["saldo_origem_depois"] == 0,
        ),
        "destino zerado antes e depois": (
            (f["saldo_destino_antes"] == 0) & (f["saldo_destino_depois"] == 0),
            (l["saldo_destino_antes"] == 0) & (l["saldo_destino_depois"] == 0),
        ),
        "contabilidade da origem fecha exatamente": (
            np.isclose(erro_contabil(f), 0),
            np.isclose(erro_contabil(l), 0),
        ),
    }

    linhas = []
    for nome, (em_fraude, em_legitima) in testes.items():
        pf, pl = float(np.mean(em_fraude)), float(np.mean(em_legitima))
        linhas.append(
            {
                "artefato": nome,
                "em_fraude": pf,
                "em_legitima": pl,
                # Quantas vezes mais provável em fraude. Acima de ~100x em
                # dado tabular real é sinal de vazamento, não de talento.
                "razao": pf / pl if pl > 0 else np.inf,
            }
        )
    return pd.DataFrame(linhas)


def desempenho_da_regra(df: pd.DataFrame) -> dict[str, float]:
    """O que a regra de duas linhas entrega, sem treinar nada.

    Este é o número que qualquer modelo treinado sobre as colunas de saldo
    está apenas reproduzindo com mais passos.
    """
    alerta = regra_do_gerador(df)
    y = df[config.TARGET].to_numpy().astype(bool)

    tp = int((alerta & y).sum())
    fp = int((alerta & ~y).sum())
    fn = int((~alerta & y).sum())

    return {
        "alertas": int(alerta.sum()),
        "verdadeiros_positivos": tp,
        "falsos_positivos": fp,
        "falsos_negativos": fn,
        "recall": tp / max(tp + fn, 1),
        "precisao": tp / max(tp + fp, 1),
        "valor_capturado": float(df.loc[alerta & y, "valor"].sum()),
        "valor_total_fraudado": float(df.loc[y, "valor"].sum()),
    }


def relatorio(df: pd.DataFrame) -> str:
    """Texto pronto para o README e para o log do pipeline."""
    aud = auditar(df)
    reg = desempenho_da_regra(df)

    linhas = ["Artefatos do simulador (prevalencia em fraude vs legitima):"]
    for r in aud.itertuples():
        razao = "inf" if np.isinf(r.razao) else f"{r.razao:>8,.0f}x"
        linhas.append(f"  {r.artefato:<48} {r.em_fraude:>7.2%} vs {r.em_legitima:>7.2%}  ({razao})")

    linhas += [
        "",
        "Regra de duas linhas, sem modelo:",
        f"  {reg['alertas']:,} alertas -> {reg['verdadeiros_positivos']:,} fraudes",
        f"  recall {reg['recall']:.2%} | precisao {reg['precisao']:.2%}",
        f"  captura {reg['valor_capturado'] / reg['valor_total_fraudado']:.2%} do valor fraudado",
    ]
    return "\n".join(linhas).replace(",", ".")
