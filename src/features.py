"""Features honestas, calculadas só com o passado.

Duas restrições moldaram esse conjunto, e as duas vieram de olhar o dado
antes de modelar:

1. **Os saldos de origem estão contaminados.** Carregam a assinatura
   aritmética do simulador (ver `leakage.py`), então ficam de fora.

2. **A conta de origem aparece uma única vez em 99,9% dos casos.** O
   PaySim não reutiliza pagadores, então não existe histórico do lado de
   quem paga — nenhuma feature de velocidade por origem é possível.

O que sobra é o lado do destino, e por sorte é justamente onde mora o
sinal real: a conta laranja. Ela recebe de várias vítimas em sequência,
e esse padrão é o mesmo em PaySim e em Pix.

Todo agregado por destino é **estritamente causal**: usa apenas
transações anteriores à linha atual. Um `groupby().mean()` simples
aqui vazaria o futuro e inflaria o resultado — que é exatamente o erro
que este projeto se propôs a não repetir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TIPOS_COM_FRAUDE = ["TRANSFER", "CASH_OUT"]


def filtrar_escopo(df: pd.DataFrame) -> pd.DataFrame:
    """Mantém apenas TRANSFER e CASH_OUT.

    Fraude é estruturalmente impossível nos outros três tipos por
    construção do simulador. Mantê-los deixaria o modelo aprender um
    filtro que uma cláusula WHERE já faz, e inflaria toda métrica: a
    prevalência cai de 0,296% para 0,129% só por diluição.
    """
    return df[df["tipo"].isin(TIPOS_COM_FRAUDE)].copy()


def volume_por_dia(df: pd.DataFrame) -> pd.DataFrame:
    """Transações, fraudes e prevalência por dia simulado."""
    d = df.assign(dia=df["hora"] // 24)
    g = d.groupby("dia").agg(
        transacoes=(config.TARGET, "size"),
        fraudes=(config.TARGET, "sum"),
    )
    g["prevalencia"] = g["fraudes"] / g["transacoes"]
    return g.reset_index()


def dias_integros(df: pd.DataFrame) -> list[int]:
    """Dias em que o volume legítimo não colapsou.

    O simulador emite fraude a taxa constante mas para de emitir tráfego
    legítimo a partir do dia 17. Os dias afetados não representam
    operação nenhuma — no dia 30 sobram 282 transações e todas são
    fraude — e mantê-los deslocaria a prevalência do teste em 17x.
    """
    vol = volume_por_dia(df)
    return vol.loc[vol["transacoes"] >= config.VOLUME_MINIMO_DIARIO, "dia"].tolist()


def filtrar_periodo_integro(df: pd.DataFrame) -> pd.DataFrame:
    """Recorta o dataset aos dias de volume íntegro."""
    ok = set(dias_integros(df))
    return df[(df["hora"] // 24).isin(ok)].copy()


def build(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Monta a matriz de features.

    Retorna
    -------
    X   : features do modelo honesto
    y   : alvo
    aux : valor, hora e dia — para a camada de triagem e o split temporal
    """
    d = filtrar_periodo_integro(filtrar_escopo(df))

    # A ordem temporal é pré-requisito de tudo que vem abaixo. `hora` tem
    # empates (várias transações na mesma hora simulada); o índice
    # original desempata de forma estável.
    d = d.sort_values(["hora"], kind="stable").reset_index(drop=True)

    out = pd.DataFrame(index=d.index)

    # ---- Valor ----------------------------------------------------------
    valor = d["valor"]
    out["valor"] = valor
    out["log_valor"] = np.log1p(valor)
    # Fraudador costuma sacar cifra redonda; cliente comum paga valor quebrado.
    out["valor_redondo_100"] = (valor % 100 == 0).astype("int8")
    out["valor_redondo_1000"] = (valor % 1000 == 0).astype("int8")
    out["centavos"] = ((valor * 100) % 100).astype("int16")

    # ---- Tempo ----------------------------------------------------------
    # `hora` é o passo do simulador: 1 a 743, uma unidade por hora.
    out["hora_do_dia"] = (d["hora"] % 24).astype("int8")
    out["dia_da_semana"] = ((d["hora"] // 24) % 7).astype("int8")
    out["madrugada"] = ((d["hora"] % 24) < 6).astype("int8")

    # ---- Tipo -----------------------------------------------------------
    out["e_transferencia"] = (d["tipo"] == "TRANSFER").astype("int8")

    # ---- Histórico da conta destino (causal) ----------------------------
    g = d.groupby("conta_destino", sort=False)

    # Quantas vezes esta conta já recebeu ANTES desta transação.
    out["destino_recebimentos_antes"] = g.cumcount().astype("int32")

    # Volume acumulado antes: cumsum inclui a linha atual, então subtrai.
    acumulado = g["valor"].cumsum() - valor
    out["destino_valor_antes"] = acumulado

    # Ticket médio histórico do destino e o quanto esta transação destoa.
    n_antes = out["destino_recebimentos_antes"].replace(0, np.nan)
    media_antes = acumulado / n_antes
    out["destino_ticket_medio_antes"] = media_antes.fillna(0.0)
    out["razao_valor_ticket_destino"] = (valor / media_antes).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(-1.0)

    # Intervalo desde o último recebimento. Conta laranja recebe em rajada.
    ultimo = g["hora"].shift()
    out["destino_horas_desde_ultimo"] = (d["hora"] - ultimo).fillna(-1).astype("float32")

    # Há quanto tempo a conta destino existe na base. A primeira aparição
    # é sempre passada ou igual à linha atual, então o transform não vaza.
    primeira = g["hora"].transform("min")
    out["destino_idade_horas"] = (d["hora"] - primeira).astype("int32")

    # Conta destino estreando: recebe pela primeira vez agora.
    out["destino_primeira_vez"] = (out["destino_recebimentos_antes"] == 0).astype("int8")

    y = d[config.TARGET].astype("int8")

    aux = pd.DataFrame(
        {
            "valor": valor,
            "hora": d["hora"],
            "dia": (d["hora"] // 24).astype("int32"),
            "tipo": d["tipo"].astype(str),
            "conta_destino": d["conta_destino"],
        },
        index=d.index,
    )

    return out, y, aux


def build_contaminado(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Mesmas features, mais as colunas de saldo.

    Serve só para uma coisa: reproduzir o resultado que a literatura de
    notebooks reporta, para poder compará-lo com o honesto lado a lado.
    Não é candidato a nada.
    """
    from . import leakage

    X, y, aux = build(df)
    d = (
        filtrar_periodo_integro(filtrar_escopo(df))
        .sort_values(["hora"], kind="stable")
        .reset_index(drop=True)
    )

    for col in leakage.COLUNAS_CONTAMINADAS:
        X[col] = d[col].to_numpy()

    # A própria assinatura, explicitada. Um modelo com árvore suficiente
    # encontra isso sozinho a partir das colunas acima.
    X["erro_contabil_origem"] = (
        d["saldo_origem_antes"] - d["valor"] - d["saldo_origem_depois"]
    ).to_numpy()

    return X, y, aux


def split_temporal(
    X: pd.DataFrame, y: pd.Series, aux: pd.DataFrame
) -> tuple[tuple, tuple, tuple]:
    """Treino, validação e teste em ordem cronológica.

    Split aleatório em dado transacional vaza de duas formas: coloca a
    mesma conta destino nos dois lados, e deixa o modelo ver o futuro do
    histórico que ele mesmo usa como feature. A separação por `hora`
    reproduz a única situação que importa — pontuar transação de amanhã
    com modelo treinado até hoje.
    """
    d = aux["dia"].to_numpy()
    m_tr = d <= config.DIA_TREINO_FIM
    m_va = (d > config.DIA_TREINO_FIM) & (d <= config.DIA_VALIDACAO_FIM)
    m_te = d > config.DIA_VALIDACAO_FIM

    def corte(m):
        return X[m], y[m], aux[m]

    return corte(m_tr), corte(m_va), corte(m_te)
