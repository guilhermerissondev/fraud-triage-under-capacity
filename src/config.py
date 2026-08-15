"""Parâmetros centrais do detector de fraude.

Como no projeto de crédito, toda premissa de negócio vive aqui. A
diferença é o que se decide: em crédito o produto da decisão é um corte
de probabilidade; em antifraude é uma **fila de revisão com capacidade
finita**, e essa restrição muda a matemática toda.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

for _p in (DATA_RAW, DATA_PROCESSED, MODELS, REPORTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset
#
# PaySim (Lopez-Rojas, Elmir & Axelsson, 2016) é um simulador baseado em
# agentes de transferências de dinheiro móvel, calibrado sobre logs reais
# de um serviço africano. Foi escolhido como proxy de Pix porque a
# mecânica é a mesma: transferência P2P instantânea seguida de saque.
#
# Não existe base pública de Pix. Essa limitação é do domínio, não do
# projeto, e está declarada no README em vez de escondida.
# --------------------------------------------------------------------------
KAGGLE_DATASET = "ealaxi/paysim1"
RAW_CSV = DATA_RAW / "PS_20174392719_1491204439457_log.csv"

TARGET = "fraude"
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Recorte temporal
#
# `step` é a hora simulada, de 1 a 743 (31 dias). A validação é temporal:
# treina no passado, testa no futuro. Split aleatório em dado transacional
# vaza — o modelo veria a mesma conta destino dos dois lados, e o
# histórico causal usado como feature enxergaria o futuro.
#
# Mas o eixo temporal do PaySim tem um defeito grave: o simulador emite
# fraude a taxa praticamente constante (~270/dia nos 31 dias) enquanto o
# volume legítimo despenca a partir do dia 17, de ~180 mil transações/dia
# para 3 mil. No dia 30 sobram 282 transações e TODAS são fraude.
#
# Treinar no começo e testar no fim significaria avaliar com prevalência
# 17x maior que a de treino. Por isso o período é recortado nos dias de
# volume íntegro, e o split acontece dentro dele.
# --------------------------------------------------------------------------
VOLUME_MINIMO_DIARIO = 100_000  # dias abaixo disso são descartados
DIA_TREINO_FIM = 10
DIA_VALIDACAO_FIM = 13
# teste = dias 14 a 16

# --------------------------------------------------------------------------
# Colunas proibidas
#
# `isFlaggedFraud` é a saída de uma regra do próprio simulador: usar como
# feature é treinar no gabarito. `nameOrig`/`nameDest` são identificadores
# e não generalizam para contas novas.
# --------------------------------------------------------------------------
COLUNAS_VAZADAS = ["isFlaggedFraud", "nameOrig", "nameDest"]

# --------------------------------------------------------------------------
# Economia da operação antifraude
#
# A decisão não é "bloquear ou liberar" no vácuo. É: dado que a mesa de
# análise revisa no máximo K alertas por dia, quais transações entram na
# fila?
#
# Três custos entram na conta:
#   fraude não detectada -> perde o valor, menos o que o MED recupera
#   alerta revisado      -> custo de analista, independente do veredito
#   bloqueio indevido    -> atrito com cliente legítimo (suporte, churn)
# --------------------------------------------------------------------------
# Os valores do PaySim NÃO estão em reais. A transação mediana é de
# 171 mil unidades e a maior passa de 92 milhões — escala que não
# corresponde a Pix nenhum. Chamar isso de "R$" seria inventar precisão,
# então a unidade fica declarada como o que é: unidade monetária do
# simulador.
MOEDA = "u.m."

# Custo de uma revisão manual (minutos de analista + sistema).
CUSTO_REVISAO = 500.00

# Custo de bloquear indevidamente uma transação legítima: atendimento,
# reversão e risco de perder o cliente.
CUSTO_ATRITO = 2_000.00

# Nota honesta sobre os dois números acima: na escala de valor do PaySim
# eles são desprezíveis: uma fraude média custa 300 mil u.m., seiscentas
# vezes o custo de revisar. A conclusão que sai disso não é um ótimo
# econômico bonito — é que revisar quase sempre compensa, e a restrição
# real da operação é quanta gente existe na mesa, não quanto custa cada
# revisão. Está no README como resultado, não escondido como premissa.

# Fração do valor recuperada quando a fraude é detectada tarde. No Pix, o
# MED (Mecanismo Especial de Devolução) devolve só o que ainda restou na
# conta destino — na prática, pouco.
RECUPERACAO_MED = 0.15

# Capacidade da mesa de análise, em alertas por dia.
CAPACIDADE_DIARIA = 200
