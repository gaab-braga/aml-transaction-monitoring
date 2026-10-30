# AML-Ops-Engine: detecção de lavagem de dinheiro com XGBoost

[![Tests](https://github.com/gaab-braga/AML-Ops-Engine/actions/workflows/test.yml/badge.svg)](https://github.com/gaab-braga/AML-Ops-Engine/actions/workflows/test.yml)
[![Docker Build](https://github.com/gaab-braga/AML-Ops-Engine/actions/workflows/docker.yml/badge.svg)](https://github.com/gaab-braga/AML-Ops-Engine/actions/workflows/docker.yml)

Modelo que dá um score de risco para cada transação bancária e marca as que parecem lavagem de dinheiro. O projeto começou em notebooks e depois virou código em `src/`, com API, linha de comando, testes e imagem Docker.

A base é a IBM AML HI-Small (dados sintéticos): 5.078.336 transações, das quais 0,10% são lavagem (`artifacts/sampling_report.json`).

## Resultado

Fonte de todos os números desta seção: [`artifacts/final_evaluation_results.json`](artifacts/final_evaluation_results.json). O [model card](artifacts/model_card.json) usa a mesma fonte.

O teste é temporal: as transações mais antigas (4.062.668) treinam o modelo e as mais recentes (1.015.668) ficam para o teste.

| Métrica no teste | Valor | O que quer dizer |
|---|---|---|
| **PR-AUC** | **0,219** | Métrica principal. Mede quantos alertas são lavagem de verdade em cada nível de cobertura. Um modelo que chuta ao acaso fica perto da taxa de positivos do teste, 0,18% (1.797 de 1.015.668, contado em `tabular_predictions.csv`). |
| ROC-AUC | 0,956 | Mede se o modelo ordena bem positivos acima de negativos. Com classe tão rara, fica alta mesmo quando a maioria dos alertas é falsa, por isso não é a métrica principal. |
| Precisão nos 100 maiores scores | 0,86 | Das 100 transações com score mais alto, 86 eram lavagem. |
| Precisão nos 500 maiores scores | 0,46 | Nas 500 primeiras, 230 eram lavagem. |
| Recall com 5% de falso positivo | 0,80 | Se a operação aceitar marcar 5% das transações legítimas, o modelo acha 80% da lavagem. No teste, 5% das legítimas são cerca de 50 mil alertas falsos. |

A calibração isotônica deixou o score próximo de uma probabilidade (ECE 0,00013, Brier 0,0011, no mesmo arquivo).

### Sobre a comparação com GNN

Uma versão anterior deste README dizia que o XGBoost tinha F1 "235 vezes" o benchmark de GNN da IBM. Isso estava errado:

- O F1 = 0,0012 em `artifacts/gnn_comparison_results.json` é de uma reprodução minha da arquitetura Multi-GNN (código em `benchmark/Multi-GNN`). Essa reprodução não funcionou. Não é um resultado publicado pela IBM.
- Outra rodada de GNN (`models/multignn/outputs/final_metrics.json`) avaliou 4.572 nós do grafo, não transações. Os números não são comparáveis com os do XGBoost.

Por isso o projeto não traz comparação com GNN.

### Outros artefatos com números diferentes

A pasta `artifacts/` guarda saídas de várias rodadas. Alguns arquivos mostram outros valores para o XGBoost, com outros hiperparâmetros:

| Arquivo | PR-AUC | Situação |
|---|---|---|
| `final_evaluation_results.json` | 0,219 | Avaliação final. Fonte do README e do model card. |
| `competition_results.json` | 0,389 | Competição de modelos (notebook 03). Bate com as previsões salvas em `tabular_predictions.csv`. |
| `calibration_results.json` | 0,369 | Calibração (notebook 04), em 812.535 transações. |

O model card antigo dizia PR-AUC = 0,509. Esse valor não aparece em nenhum outro artefato e foi removido.

## Estrutura

```
entrypoints/   API (FastAPI), linha de comando (Typer) e processamento em lote
src/           código do pipeline: dados, features, modelo, monitoramento
notebooks/     01 EDA · 02 GNN · 03 competição de modelos · 04 calibração · 05 SHAP · 06 robustez · 07 resumo
artifacts/     métricas, gráficos e previsões das rodadas
benchmark/     código da reprodução Multi-GNN
tests/         testes com pytest
```

## Como rodar

```bash
pip install -r requirements.txt
pytest tests/
```

API com Docker:

```bash
docker build -t aml-project .
docker run -p 8000:8000 aml-project
curl http://localhost:8000/health
```

A imagem não traz modelo treinado (`models/*.pkl` fica fora do git). Sem ele, `/health` responde `"model_loaded": false` e `/predict` devolve 503.

Os dados brutos (`data/`) estão no Git LFS. Para clonar sem baixá-los:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/gaab-braga/AML-Ops-Engine.git
```

## Limitações

- Uma única divisão temporal de teste. Não há medição em produção.
- Base sintética. O desempenho em dados reais não foi medido.
- A precisão cai rápido: 0,86 nos 100 primeiros alertas, 0,46 nos 500 primeiros.
- O notebook que gerou `final_evaluation_results.json` não está no repositório, e as previsões desse modelo não foram salvas. Os números da tabela de resultado não podem ser recalculados a partir dos arquivos do repositório.

## Licença

Apache 2.0. Ver [LICENSE](LICENSE).
