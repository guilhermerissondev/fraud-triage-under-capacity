"""Simulador de mesa de análise antifraude.

A pergunta que o app responde não é "esta transação é fraude". É "com N
analistas, quais transações a mesa consegue olhar hoje, e quanto isso
salva".

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, triage  # noqa: E402

st.set_page_config(page_title="Mesa Antifraude", page_icon="▲", layout="wide")

AZUL, VERMELHO, VERDE, CINZA = "#1f4e79", "#c0392b", "#27ae60", "#7f8c8d"


@st.cache_data
def carregar():
    caminho = config.DATA_PROCESSED / "teste_scorado.parquet"
    return pd.read_parquet(caminho) if caminho.exists() else None


d = carregar()
if d is None:
    st.error(
        "Artefatos não encontrados. Rode o pipeline antes:\n\n"
        "```\npython -m scripts.train\n```"
    )
    st.stop()

p = d["p"].to_numpy()
valor = d["valor"].to_numpy()
fraude = d["fraude"].to_numpy()
dia = d["dia"].to_numpy()
n_dias = len(np.unique(dia))


# --------------------------------------------------------------------------
# Parâmetros
# --------------------------------------------------------------------------
st.sidebar.header("Operação")
capacidade = st.sidebar.slider(
    "Capacidade da mesa (alertas/dia)", 10, 5000, int(config.CAPACIDADE_DIARIA), 10,
    help="Quantas transações a equipe consegue revisar por dia.",
)
criterio = st.sidebar.radio(
    "Ordenação da fila",
    ["valor_em_risco", "probabilidade"],
    format_func=lambda s: {
        "valor_em_risco": "Valor em risco (p × valor)",
        "probabilidade": "Probabilidade (padrão)",
    }[s],
)

st.sidebar.markdown("---")
st.sidebar.header("Custos")
custo_revisao = st.sidebar.number_input(
    "Custo por revisão", 0.0, 100_000.0, float(config.CUSTO_REVISAO), 100.0
)
custo_atrito = st.sidebar.number_input(
    "Custo de reter transação legítima", 0.0, 100_000.0, float(config.CUSTO_ATRITO), 100.0
)
recuperacao = st.sidebar.slider(
    "Recuperação via MED", 0.0, 1.0, float(config.RECUPERACAO_MED), 0.05,
    help="Fração do valor devolvida quando a fraude escapa e é contestada depois.",
)
st.sidebar.caption(
    f"Valores em {config.MOEDA} — unidade do simulador, não reais. "
    "A transação mediana do PaySim é de 171 mil unidades."
)

custos = dict(
    custo_revisao=custo_revisao, custo_atrito=custo_atrito, recuperacao=recuperacao
)

st.title("Mesa antifraude sob capacidade")
st.caption(
    f"{len(d):,} transações · {n_dias} dias · {int(fraude.sum()):,} fraudes "
    f"({fraude.mean():.3%}) · modelo treinado sem os atalhos do simulador".replace(",", ".")
)

r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade, criterio, **custos)
r_prob = triage.aplicar_triagem(p, valor, fraude, dia, capacidade, "probabilidade", **custos)
r_val = triage.aplicar_triagem(p, valor, fraude, dia, capacidade, "valor_em_risco", **custos)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Fraudes barradas", f"{r['fraudes_pegas']:,}".replace(",", "."),
          delta=f"de {int(fraude.sum()):,}".replace(",", "."))
c2.metric("Recall de contagem", f"{r['recall_contagem']:.1%}")
c3.metric("VDR (valor barrado)", f"{r['vdr']:.1%}",
          delta=f"{r['vdr'] - r['recall_contagem']:+.1%} vs contagem")
c4.metric("Precisão da fila", f"{r['precisao_da_fila']:.1%}",
          help="Fração dos alertas revisados que era fraude de verdade.")

st.markdown("---")

aba1, aba2, aba3 = st.tabs(["Capacidade", "Ordenação da fila", "A fila de hoje"])

# ==========================================================================
with aba1:
    curva = triage.curva_capacidade(p, valor, fraude, dia, criterio=criterio, **custos)
    otima = triage.capacidade_otima(curva)

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=curva["capacidade_diaria"], y=curva["beneficio_liquido"],
            mode="lines", line=dict(color=AZUL, width=2.5),
            hovertemplate="%{x} alertas/dia<br>%{y:,.0f}<extra></extra>",
        ))
        fig.add_vline(x=capacidade, line=dict(color=VERMELHO, width=2, dash="dash"),
                      annotation_text="mesa atual")
        fig.add_vline(x=otima, line=dict(color=VERDE, width=1.5, dash="dot"),
                      annotation_text=f"ótimo ({otima})")
        fig.update_layout(
            title="Benefício líquido por tamanho da mesa",
            xaxis_title="Alertas revisados por dia", xaxis_type="log",
            yaxis_title=f"Benefício ({config.MOEDA})", height=400, showlegend=False,
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=curva["capacidade_diaria"], y=curva["vdr"] * 100,
            mode="lines", name="VDR (valor)", line=dict(color=AZUL, width=2.5),
        ))
        fig2.add_trace(go.Scatter(
            x=curva["capacidade_diaria"], y=curva["recall_contagem"] * 100,
            mode="lines", name="recall (contagem)", line=dict(color=CINZA, width=2),
        ))
        fig2.add_vline(x=capacidade, line=dict(color=VERMELHO, width=2, dash="dash"))
        fig2.update_layout(
            title="Cobertura por tamanho da mesa",
            xaxis_title="Alertas revisados por dia", xaxis_type="log",
            yaxis_title="%", height=400,
            legend=dict(orientation="h", y=-0.25),
        )
        st.plotly_chart(fig2, width="stretch")

    st.info(
        f"Com {capacidade} alertas/dia a mesa barra {r['vdr']:.1%} do valor fraudado "
        f"revisando {r['alertas_por_dia']:.0f} transações por dia — de "
        f"{len(d)/n_dias:,.0f} que passam.".replace(",", ".")
        + f" O ótimo desta curva está em {otima} alertas/dia, mas repare na inclinação: "
        "o retorno por analista adicional cai rápido depois das primeiras centenas."
    )

# ==========================================================================
with aba2:
    st.subheader("A mesma fila, duas ordens de prioridade")
    st.markdown(
        "Ordenar por probabilidade maximiza **quantas** fraudes a mesa pega. "
        "Ordenar por `p × valor` maximiza **quanto dinheiro** ela salva. "
        "Não são a mesma coisa, e a diferença aparece abaixo."
    )

    tabela = pd.DataFrame({
        "Ordenado por probabilidade": r_prob,
        "Ordenado por valor em risco": r_val,
    }).T
    vis = pd.DataFrame({
        "Fraudes barradas": tabela["fraudes_pegas"].map(lambda v: f"{int(v):,}".replace(",", ".")),
        "Recall (contagem)": tabela["recall_contagem"].map(lambda v: f"{v:.1%}"),
        "VDR (valor)": tabela["vdr"].map(lambda v: f"{v:.1%}"),
        "Precisão da fila": tabela["precisao_da_fila"].map(lambda v: f"{v:.1%}"),
        "Valor evitado": tabela["valor_evitado"].map(triage.formatar),
        "Benefício líquido": tabela["beneficio_liquido"].map(triage.formatar),
    })
    st.dataframe(vis, width="stretch")

    delta = r_val["beneficio_liquido"] - r_prob["beneficio_liquido"]
    if r_val["recall_contagem"] < r_prob["recall_contagem"]:
        st.warning(
            f"Repare no trade-off: ordenar por valor em risco pega **menos** fraudes "
            f"({r_val['fraudes_pegas']} contra {r_prob['fraudes_pegas']}) e ainda assim "
            f"salva {triage.formatar(delta)} a mais. Um painel que mede recall de "
            f"contagem registraria isso como piora."
        )

# ==========================================================================
with aba3:
    st.subheader("Fila de um dia")
    dia_sel = st.selectbox("Dia", sorted(np.unique(dia)))
    no_dia = np.flatnonzero(dia == dia_sel)

    prioridade = (
        p[no_dia] if criterio == "probabilidade"
        else p[no_dia] * valor[no_dia] * (1 - recuperacao)
    )
    ordem = no_dia[np.argsort(-prioridade)][:capacidade]

    fila = pd.DataFrame({
        "posição": np.arange(1, len(ordem) + 1),
        "probabilidade": p[ordem],
        "valor": valor[ordem],
        "valor em risco": p[ordem] * valor[ordem] * (1 - recuperacao),
        "tipo": d["tipo"].to_numpy()[ordem],
        "hora": d["hora"].to_numpy()[ordem] % 24,
        "era fraude": np.where(fraude[ordem] == 1, "sim", "não"),
    })
    st.caption(
        f"{len(no_dia):,} transações no dia {dia_sel}; a mesa revisa as {len(ordem)} "
        f"primeiras desta ordem.".replace(",", ".")
    )
    st.dataframe(
        fila.head(200).style.format({
            "probabilidade": "{:.3f}", "valor": "{:,.0f}", "valor em risco": "{:,.0f}",
        }),
        width="stretch", height=460,
    )
