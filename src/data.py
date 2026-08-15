"""Ingestão do PaySim.

6,3 milhões de transações simuladas ao longo de 30 dias, com cinco tipos
de operação e marcação de fraude. As fraudes ocorrem apenas em TRANSFER
e CASH_OUT — o padrão do simulador é assumir a conta, transferir para uma
conta controlada pelo fraudador e sacar em seguida.

É a mesma sequência de um golpe de Pix: transferência instantânea para
conta laranja e saque imediato antes do bloqueio.
"""

from __future__ import annotations

import os
import zipfile

import pandas as pd

from . import config

_RENAME = {
    "step": "hora",
    "type": "tipo",
    "amount": "valor",
    "nameOrig": "conta_origem",
    "oldbalanceOrg": "saldo_origem_antes",
    "newbalanceOrig": "saldo_origem_depois",
    "nameDest": "conta_destino",
    "oldbalanceDest": "saldo_destino_antes",
    "newbalanceDest": "saldo_destino_depois",
    "isFraud": config.TARGET,
    "isFlaggedFraud": "regra_do_simulador",
}


class KaggleCredentialsMissing(RuntimeError):
    pass


def download(force: bool = False) -> None:
    """Baixa o PaySim via API do Kaggle.

    A credencial nunca passa por este código. Onde ela mora é problema da
    biblioteca do Kaggle, não nosso: dependendo da versão e do fluxo pode
    ser um `kaggle.json`, um `access_token` ou um cache de OAuth. Checar
    um caminho específico aqui só cria um guard que envelhece mal — então
    a autenticação é delegada, e a mensagem útil aparece se ela falhar.
    """
    if config.RAW_CSV.exists() and not force:
        return

    import kaggle  # importa só aqui: o módulo exige credencial no import

    try:
        kaggle.api.authenticate()
    except Exception as e:
        raise KaggleCredentialsMissing(
            "Não foi possível autenticar no Kaggle.\n"
            "  Rode:  kaggle auth login\n"
            "  (fluxo OAuth pelo navegador; não é preciso gerenciar token)\n"
            f"  Erro original: {e}"
        ) from e

    kaggle.api.dataset_download_files(
        config.KAGGLE_DATASET, path=str(config.DATA_RAW), quiet=False
    )

    for z in config.DATA_RAW.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(config.DATA_RAW)
        z.unlink()


def load(nrows: int | None = None) -> pd.DataFrame:
    """Carrega o CSV com nomes legíveis e tipos enxutos.

    6,3M linhas em float64 consomem memória à toa; os downcasts abaixo
    cortam o dataframe pela metade sem perder precisão relevante.
    """
    download()

    dtypes = {
        "step": "int32",
        "type": "category",
        "amount": "float64",
        "oldbalanceOrg": "float64",
        "newbalanceOrig": "float64",
        "oldbalanceDest": "float64",
        "newbalanceDest": "float64",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    df = pd.read_csv(config.RAW_CSV, dtype=dtypes, nrows=nrows)
    return df.rename(columns=_RENAME)


def resumo(df: pd.DataFrame) -> pd.DataFrame:
    """Panorama por tipo de operação: volume, valor e incidência de fraude."""
    return (
        df.groupby("tipo", observed=True)
        .agg(
            transacoes=(config.TARGET, "size"),
            fraudes=(config.TARGET, "sum"),
            valor_total=("valor", "sum"),
            valor_fraudado=("valor", lambda s: s[df.loc[s.index, config.TARGET] == 1].sum()),
        )
        .assign(taxa_fraude=lambda d: d["fraudes"] / d["transacoes"])
        .sort_values("fraudes", ascending=False)
    )
