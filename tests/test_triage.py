"""Testes da triagem sob capacidade.

A fila é a peça que decide quanto dinheiro a operação salva. Se a
contabilidade dela estiver errada, o número do README está errado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import triage  # noqa: E402

CUSTOS = dict(custo_revisao=8.0, custo_atrito=25.0, recuperacao=0.15)


@pytest.fixture
def lote():
    """Um dia, seis transações, duas fraudes — uma cara e uma barata."""
    p = np.array([0.95, 0.40, 0.30, 0.10, 0.05, 0.02])
    valor = np.array([200.0, 50_000.0, 100.0, 80.0, 60.0, 40.0])
    fraude = np.array([1, 1, 0, 0, 0, 0])
    dia = np.array([1, 1, 1, 1, 1, 1])
    return p, valor, fraude, dia


# --------------------------------------------------------------------------
# Capacidade
# --------------------------------------------------------------------------
def test_capacidade_limita_a_fila(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade_diaria=2, **CUSTOS)
    assert r["alertas_revisados"] == 2


def test_capacidade_e_diaria_e_nao_acumula():
    """Sobra de um dia não vira folga do dia seguinte."""
    n = 20
    p = np.linspace(0.9, 0.1, n)
    valor = np.full(n, 1_000.0)
    fraude = np.zeros(n, dtype=int)
    dia = np.repeat([1, 2], n // 2)

    r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade_diaria=3, **CUSTOS)
    assert r["alertas_revisados"] == 6  # 3 por dia, em dois dias


def test_capacidade_maior_que_o_volume_revisa_tudo(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade_diaria=999, **CUSTOS)
    assert r["alertas_revisados"] == len(p)
    assert r["fraudes_escapadas"] == 0
    assert r["vdr"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# O ponto do projeto: ordenar por valor em risco
# --------------------------------------------------------------------------
def test_probabilidade_pega_a_fraude_errada(lote):
    """Com uma vaga só, ranquear por probabilidade escolhe a fraude de
    R$ 200 e deixa passar a de R$ 50.000."""
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(
        p, valor, fraude, dia, capacidade_diaria=1, criterio="probabilidade", **CUSTOS
    )
    assert r["fraudes_pegas"] == 1
    assert r["valor_evitado"] == pytest.approx(200.0)


def test_valor_em_risco_pega_a_fraude_cara(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(
        p, valor, fraude, dia, capacidade_diaria=1, criterio="valor_em_risco", **CUSTOS
    )
    assert r["fraudes_pegas"] == 1
    assert r["valor_evitado"] == pytest.approx(50_000.0)


def test_mesmo_recall_resultados_financeiros_opostos(lote):
    """Os dois critérios pegam uma fraude cada — mesmo recall de contagem,
    250x de diferença em dinheiro. É por isso que recall de contagem não
    serve para decidir."""
    p, valor, fraude, dia = lote
    por_p = triage.aplicar_triagem(p, valor, fraude, dia, 1, "probabilidade", **CUSTOS)
    por_v = triage.aplicar_triagem(p, valor, fraude, dia, 1, "valor_em_risco", **CUSTOS)

    assert por_p["recall_contagem"] == por_v["recall_contagem"]
    assert por_v["vdr"] > por_p["vdr"]
    assert por_v["beneficio_liquido"] > por_p["beneficio_liquido"]


def test_comparacao_reporta_ganho_relativo(lote):
    p, valor, fraude, dia = lote
    df = triage.comparar_criterios(p, valor, fraude, dia, capacidade_diaria=1, **CUSTOS)
    assert set(df.index) == {"probabilidade", "valor_em_risco"}
    assert df.loc["probabilidade", "ganho_vs_probabilidade"] == pytest.approx(0.0)
    assert df.loc["valor_em_risco", "ganho_vs_probabilidade"] > 0


# --------------------------------------------------------------------------
# Contabilidade
# --------------------------------------------------------------------------
def test_beneficio_liquido_fecha_com_as_parcelas(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade_diaria=3, **CUSTOS)
    assert r["beneficio_liquido"] == pytest.approx(
        r["valor_evitado"] - r["custo_operacao"] - r["custo_friccao"]
    )


def test_custo_de_revisao_conta_alerta_e_nao_acerto(lote):
    """O analista é pago por revisar, acertando ou não."""
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade_diaria=4, **CUSTOS)
    assert r["custo_operacao"] == pytest.approx(4 * 8.0)


def test_fraude_escapada_perde_valor_liquido_do_med(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(
        p, valor, fraude, dia, capacidade_diaria=1, criterio="probabilidade", **CUSTOS
    )
    # Escapou a de 50.000; o MED devolve 15%.
    assert r["valor_perdido"] == pytest.approx(50_000.0 * 0.85)


def test_sem_capacidade_nada_e_revisado(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(p, valor, fraude, dia, capacidade_diaria=0, **CUSTOS)
    assert r["alertas_revisados"] == 0
    assert r["valor_evitado"] == 0.0
    assert r["beneficio_liquido"] == pytest.approx(0.0)


def test_vdr_e_recall_divergem_quando_o_valor_e_desigual(lote):
    p, valor, fraude, dia = lote
    r = triage.aplicar_triagem(
        p, valor, fraude, dia, capacidade_diaria=1, criterio="probabilidade", **CUSTOS
    )
    assert r["recall_contagem"] == pytest.approx(0.5)
    assert r["vdr"] < 0.01  # 200 de 50.200


# --------------------------------------------------------------------------
# Curva de capacidade
# --------------------------------------------------------------------------
def test_vdr_nao_cai_ao_aumentar_a_mesa():
    rng = np.random.default_rng(0)
    n = 2_000
    p = rng.uniform(0, 1, n)
    valor = rng.lognormal(7, 1.5, n)
    fraude = rng.binomial(1, p * 0.2)
    dia = rng.integers(1, 6, n)

    curva = triage.curva_capacidade(
        p, valor, fraude, dia, capacidades=np.array([5, 20, 50, 200, 1000]), **CUSTOS
    )
    assert curva["vdr"].is_monotonic_increasing


def test_capacidade_otima_esta_no_grid():
    rng = np.random.default_rng(1)
    n = 1_000
    p = rng.uniform(0, 1, n)
    valor = rng.lognormal(7, 1.5, n)
    fraude = rng.binomial(1, p * 0.1)
    dia = rng.integers(1, 4, n)

    caps = np.array([5, 25, 100, 400])
    curva = triage.curva_capacidade(p, valor, fraude, dia, capacidades=caps, **CUSTOS)
    assert triage.capacidade_otima(curva) in caps.tolist()


# --------------------------------------------------------------------------
# Precisão@k
# --------------------------------------------------------------------------
def test_precisao_em_k(lote):
    p, valor, fraude, dia = lote
    assert triage.precisao_em_k(p, fraude, 1) == pytest.approx(1.0)
    assert triage.precisao_em_k(p, fraude, 2) == pytest.approx(1.0)
    assert triage.precisao_em_k(p, fraude, 4) == pytest.approx(0.5)
    assert triage.precisao_em_k(p, fraude, 0) == 0.0


def test_formatacao_usa_o_simbolo_da_config():
    from src import config

    assert triage.formatar(1234.0).startswith(config.MOEDA)
    assert "mi" in triage.formatar(2_500_000.0)
    assert "mil" in triage.formatar(4_050.0)
    assert "bi" in triage.formatar(3_200_000_000.0)


def test_formatacao_nao_mastiga_o_simbolo():
    """Regressão: a troca de separadores era aplicada à string inteira e
    transformava "u.m." em "u,m,"."""
    from src import config

    saida = triage.formatar(1_010_000_000.0)
    assert config.MOEDA in saida
    assert saida == f"{config.MOEDA} 1,01 bi"


def test_formatacao_separadores_brasileiros():
    assert triage.formatar(1_234.5).endswith("1,23 mil")
    assert triage.formatar(12.5).endswith("12,50")
