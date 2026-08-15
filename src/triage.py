"""Triagem sob capacidade finita.

A ideia central do projeto está aqui.

Em crédito, a decisão é um corte de probabilidade e todo mundo abaixo dele
é aprovado — não há limite de quantas contas se pode aprovar. Em
antifraude não é assim: a mesa de análise revisa um número fixo de
alertas por dia. Se o modelo gerar 5.000 alertas e a mesa aguentar 200,
os outros 4.800 não são "recusados" — eles simplesmente passam.

Isso muda a pergunta. Não é "qual a probabilidade de corte", é **quais
200 transações entram na fila hoje**. E a resposta não é "as 200 mais
prováveis".

Ranquear por probabilidade maximiza o número de fraudes pegas. Mas o
prejuízo não se mede em contagem, se mede em reais. Uma fraude de
R$ 50.000 com 40% de probabilidade vale mais atenção que uma de R$ 200
com 95%:

    perda esperada evitada = p x valor x (1 - recuperação do MED)

Ordenar por essa quantidade, e não por `p`, é o que separa um modelo de
fraude de uma operação antifraude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------
# Critérios de ordenação da fila
# --------------------------------------------------------------------------
def score_probabilidade(p: np.ndarray, valor: np.ndarray) -> np.ndarray:
    """Ordenação ingênua: só a probabilidade. É o padrão da indústria e o
    termo de comparação deste projeto."""
    return p


def score_valor_em_risco(
    p: np.ndarray,
    valor: np.ndarray,
    recuperacao: float | None = None,
) -> np.ndarray:
    """Perda esperada evitada ao revisar a transação."""
    rec = config.RECUPERACAO_MED if recuperacao is None else recuperacao
    return p * valor * (1.0 - rec)


CRITERIOS = {
    "probabilidade": score_probabilidade,
    "valor_em_risco": score_valor_em_risco,
}


# --------------------------------------------------------------------------
# Simulação da fila
# --------------------------------------------------------------------------
def aplicar_triagem(
    p: np.ndarray,
    valor: np.ndarray,
    fraude: np.ndarray,
    dia: np.ndarray,
    capacidade_diaria: int | None = None,
    criterio: str = "valor_em_risco",
    custo_revisao: float | None = None,
    custo_atrito: float | None = None,
    recuperacao: float | None = None,
) -> dict[str, float]:
    """Roda a mesa de análise dia a dia e apura o resultado.

    A capacidade é diária, não global: sobra de segunda não acumula para
    terça. Tratar o mês inteiro como uma fila única daria ao modelo uma
    folga que a operação real não tem.

    Toda transação não revisada segue seu curso — se era fraude, o valor
    se perde (menos o que o MED recupera).
    """
    cap = config.CAPACIDADE_DIARIA if capacidade_diaria is None else capacidade_diaria
    c_rev = config.CUSTO_REVISAO if custo_revisao is None else custo_revisao
    c_atr = config.CUSTO_ATRITO if custo_atrito is None else custo_atrito
    rec = config.RECUPERACAO_MED if recuperacao is None else recuperacao

    ordenador = CRITERIOS[criterio]
    prioridade = ordenador(p, valor) if criterio == "probabilidade" else ordenador(p, valor, rec)

    revisado = np.zeros(len(p), dtype=bool)
    for d in np.unique(dia):
        no_dia = np.flatnonzero(dia == d)
        if len(no_dia) == 0:
            continue
        topo = no_dia[np.argsort(-prioridade[no_dia])][:cap]
        revisado[topo] = True

    fraude = fraude.astype(bool)

    # Fraude revisada: bloqueada a tempo, perda evitada por inteiro.
    pegas = revisado & fraude
    # Fraude não revisada: perde o valor, menos o resgate do MED.
    escapadas = ~revisado & fraude
    # Transação legítima retida na fila: o analista libera depois, mas o
    # cliente já esperou. É atrito, não bloqueio definitivo — por isso o
    # custo unitário é menor que o de uma fraude perdida, e não zero.
    falsos_positivos = revisado & ~fraude

    valor_evitado = float(valor[pegas].sum())
    valor_perdido = float(valor[escapadas].sum() * (1.0 - rec))
    custo_operacao = float(revisado.sum() * c_rev)
    custo_friccao = float(falsos_positivos.sum() * c_atr)

    total_fraude = float(valor[fraude].sum())
    n_dias = len(np.unique(dia))

    return {
        "criterio": criterio,
        "capacidade_diaria": cap,
        "alertas_revisados": int(revisado.sum()),
        "alertas_por_dia": revisado.sum() / max(n_dias, 1),
        "fraudes_pegas": int(pegas.sum()),
        "fraudes_escapadas": int(escapadas.sum()),
        "recall_contagem": float(pegas.sum() / max(fraude.sum(), 1)),
        # Value Detection Rate: fração do VALOR fraudado que foi barrada.
        # É a métrica que a operação acompanha, não o recall de contagem.
        "vdr": valor_evitado / total_fraude if total_fraude else 0.0,
        "precisao_da_fila": float(pegas.sum() / max(revisado.sum(), 1)),
        "valor_evitado": valor_evitado,
        "valor_perdido": valor_perdido,
        "custo_operacao": custo_operacao,
        "custo_friccao": custo_friccao,
        "beneficio_liquido": valor_evitado - custo_operacao - custo_friccao,
    }


def comparar_criterios(
    p: np.ndarray,
    valor: np.ndarray,
    fraude: np.ndarray,
    dia: np.ndarray,
    capacidade_diaria: int | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Mesma fila, mesmo modelo, duas ordens de prioridade."""
    linhas = [
        aplicar_triagem(p, valor, fraude, dia, capacidade_diaria, criterio=c, **kwargs)
        for c in CRITERIOS
    ]
    df = pd.DataFrame(linhas).set_index("criterio")
    base = df.loc["probabilidade", "beneficio_liquido"]
    df["ganho_vs_probabilidade"] = df["beneficio_liquido"] - base
    df["ganho_pct"] = (df["beneficio_liquido"] - base) / base if base > 0 else np.nan
    return df


def curva_capacidade(
    p: np.ndarray,
    valor: np.ndarray,
    fraude: np.ndarray,
    dia: np.ndarray,
    capacidades: np.ndarray | None = None,
    criterio: str = "valor_em_risco",
    **kwargs,
) -> pd.DataFrame:
    """Resultado em função do tamanho da mesa.

    Responde a pergunta que o gestor de fato faz: contratar mais um
    analista se paga? O retorno é decrescente — cada alerta adicional é
    menos promissor que o anterior — e existe um ponto em que o custo de
    revisão supera a perda evitada.
    """
    if capacidades is None:
        # O teto vai até revisar tudo. Um grid curto acharia o "ótimo" na
        # borda da própria busca — foi o que aconteceu na primeira versão,
        # que parava em 5.000 e reportava 5.000 como ótimo.
        teto = int(np.ceil(len(p) / max(len(np.unique(dia)), 1)))
        capacidades = np.unique(np.round(np.geomspace(10, teto, 60)).astype(int))
    linhas = [
        aplicar_triagem(p, valor, fraude, dia, int(c), criterio=criterio, **kwargs)
        for c in capacidades
    ]
    return pd.DataFrame(linhas)


def capacidade_otima(curva: pd.DataFrame) -> int:
    """Capacidade que maximiza o benefício líquido."""
    return int(curva.loc[curva["beneficio_liquido"].idxmax(), "capacidade_diaria"])


def precisao_em_k(p: np.ndarray, fraude: np.ndarray, k: int) -> float:
    """Precisão nas k transações mais suspeitas.

    Com 0,13% de fraude, acurácia e AUC-ROC não dizem nada útil — um
    modelo que chuta "tudo legítimo" acerta 99,87%. Precisão@k é o que
    corresponde ao trabalho real da mesa.
    """
    if k <= 0:
        return 0.0
    topo = np.argsort(-p)[:k]
    return float(fraude[topo].mean())


def formatar(v: float) -> str:
    """Valor em notação brasileira, com o símbolo intacto.

    A troca de separadores é aplicada só ao número. Fazê-la na string
    inteira transformava "u.m." em "u,m,".
    """
    numero, sufixo = v, ""
    for limiar, s in ((1e9, " bi"), (1e6, " mi"), (1e3, " mil")):
        if abs(v) >= limiar:
            numero, sufixo = v / limiar, s
            break
    texto = f"{numero:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{config.MOEDA} {texto}{sufixo}"
