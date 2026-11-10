# Propensão a Depósito a Prazo — Trabalho Final de Machine Learning

Pós-graduação em IA Generativa e Aplicações com LLMs (PUC Minas) · Prof. Maurício Rodrigues da Silva · Aluno: Rafael Brasil

Modelo de classificação que estima, **antes da ligação**, a chance de um cliente bancário contratar um
depósito a prazo, para priorizar a fila de telemarketing. Dados: *Bank Marketing* (UCI Machine Learning
Repository, `bank-full.csv`, 45.211 contatos). Modelo final: XGBoost com probabilidades calibradas
(Platt; ROC-AUC ≈ 0,80 no teste) e limiar escolhido pelo F2 ("ligar para quem tem chance acima da média"), servido
por uma API FastAPI com página web.

Relatório completo: [`relatorio/Relatorio_Trabalho_Final.pdf`](relatorio/Relatorio_Trabalho_Final.pdf) ·
Notebook: [`trabalho_final_bank_marketing.ipynb`](trabalho_final_bank_marketing.ipynb)

## Estrutura

```
Trabalho final/
├── trabalho_final_bank_marketing.ipynb   # notebook do experimento (executado do início ao fim)
├── dados/bank-full.csv                   # dataset (UCI, id 222)
├── modelo/
│   ├── preprocessador.joblib             # imputadores + StandardScaler + OneHotEncoder (ajustados no treino)
│   ├── modelo.joblib                     # classificador XGBoost
│   ├── calibrador.joblib                 # calibração de Platt: score do XGBoost -> probabilidade real
│   └── metadados.json                    # features, categorias, faixas válidas, limiar, métricas
├── app/
│   ├── app.py                            # API FastAPI (carrega os .joblib uma vez; não retreina)
│   └── index.html                        # interface web (HTML + CSS + JS)
├── figuras/                              # gráficos gerados pelo notebook
├── relatorio/Relatorio_Trabalho_Final.pdf
└── requirements.txt
```

## Como executar

Na pasta `Trabalho final`:

```bash
pip install -r requirements.txt
```

```bash
uvicorn app.app:app --reload
```

Depois abra <http://127.0.0.1:8000> no navegador. A documentação automática da API fica em
<http://127.0.0.1:8000/docs>.

Para reproduzir o experimento, abra o notebook e use **Kernel > Restart & Run All** (leva cerca de 5 minutos,
por causa do Grid Search). Ele regrava os arquivos de `modelo/` e `figuras/`.

## Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/` | página web |
| GET | `/api/metadados` | categorias, faixas válidas, limiar e métricas do modelo |
| POST | `/api/prever` | recebe os 15 campos do cliente e devolve a probabilidade calibrada, a prioridade, a taxa real observada na faixa e a recomendação |
| GET | `/api/saude` | verificação simples de funcionamento |

Entradas inválidas (campo vazio, texto no lugar de número, valor fora da faixa, histórico incoerente) são
recusadas com mensagens em português, tanto na página quanto na API (HTTP 422).

## Fonte dos dados

Moro, S., Rita, P. & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C5K306>
