"""
API do Trabalho Final de Machine Learning (PUC Minas) - Rafael Brasil

Prevê a propensão de um cliente bancário a contratar um depósito a prazo
(dataset Bank Marketing, UCI). Carrega o pré-processador, o modelo e o calibrador
exportados com joblib UMA vez, na inicialização, e nunca retreina.

Como executar (a partir da pasta do projeto):
    pip install -r requirements.txt
    uvicorn app.app:app --reload
    # depois abra http://127.0.0.1:8000
"""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PASTA_APP = Path(__file__).resolve().parent
PASTA_MODELO = PASTA_APP.parent / "modelo"

# ------------------------------------------------------------------ artefatos
preprocessador = joblib.load(PASTA_MODELO / "preprocessador.joblib")
modelo = joblib.load(PASTA_MODELO / "modelo.joblib")
calibrador = joblib.load(PASTA_MODELO / "calibrador.joblib")
META = json.loads((PASTA_MODELO / "metadados.json").read_text(encoding="utf-8"))

LIMIAR = META["limiar"]
TAXA_BASE = META["taxa_base_adesao"]
FAIXAS = META["faixas_numericas"]
CATEGORIAS = META["categorias"]

# Faixas de probabilidade do teste (para mostrar a taxa real observada na faixa do cliente)
FAIXAS_CALIBRACAO = []
for linha in META["calibracao"]["tabela_teste"]:
    lo, hi = (float(v) for v in re.findall(r"-?\d+\.?\d*", linha["faixa"])[:2])
    FAIXAS_CALIBRACAO.append((lo, hi, linha["taxa_real"], linha["clientes"]))


def logit(p):
    """Mesma transformação usada no notebook antes do calibrador de Platt."""
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def probabilidade(entrada: pd.DataFrame) -> float:
    """Score do XGBoost -> probabilidade calibrada."""
    score = modelo.predict_proba(preprocessador.transform(entrada))[:, 1]
    return float(calibrador.predict_proba(logit(score))[0, 1])


def faixa_do_teste(prob: float):
    for lo, hi, taxa, n in FAIXAS_CALIBRACAO:
        if lo < prob <= hi or (prob <= hi and lo < 0):
            return {"de": max(lo, 0.0), "ate": hi, "taxa_real": taxa, "clientes": n}
    return None

ROTULOS = {
    "age": "Idade", "job": "Profissão", "marital": "Estado civil", "education": "Escolaridade",
    "default": "Crédito em atraso", "balance": "Saldo médio anual", "housing": "Financiamento imobiliário",
    "loan": "Empréstimo pessoal", "contact": "Meio de contato", "day": "Dia do contato",
    "month": "Mês do contato", "campaign": "Contatos nesta campanha",
    "pdays": "Dias desde o último contato", "previous": "Contatos anteriores",
    "poutcome": "Resultado da campanha anterior",
}


# ------------------------------------------------------------------ esquema de entrada
class Cliente(BaseModel):
    """Dados do cliente. As faixas espelham metadados.json (faixas plausíveis)."""
    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=FAIXAS["age"][0], le=FAIXAS["age"][1])
    job: str
    marital: str
    education: str
    default: str
    balance: float = Field(..., ge=FAIXAS["balance"][0], le=FAIXAS["balance"][1])
    housing: str
    loan: str
    contact: str
    day: int = Field(..., ge=FAIXAS["day"][0], le=FAIXAS["day"][1])
    month: str
    campaign: int = Field(..., ge=FAIXAS["campaign"][0], le=FAIXAS["campaign"][1])
    pdays: int = Field(..., ge=FAIXAS["pdays"][0], le=FAIXAS["pdays"][1])
    previous: int = Field(..., ge=FAIXAS["previous"][0], le=FAIXAS["previous"][1])
    poutcome: str

    @field_validator(*CATEGORIAS.keys())
    @classmethod
    def categoria_valida(cls, valor, info):
        validas = CATEGORIAS[info.field_name]
        if valor not in validas:
            raise ValueError(f"opção inválida '{valor}'. Use uma de: {', '.join(validas)}")
        return valor

    @field_validator("pdays")
    @classmethod
    def pdays_valido(cls, valor):
        if valor == 0:
            raise ValueError("use -1 para 'nunca contatado' ou um número de dias a partir de 1")
        return valor

    @model_validator(mode="after")
    def historico_coerente(self):
        # Na base de treino: pdays = -1  <=>  previous = 0  <=>  poutcome = 'unknown'
        if self.pdays == -1 and self.previous > 0:
            raise ValueError("Histórico incoerente: se o cliente nunca foi contatado (dias = -1), "
                             "'Contatos anteriores' deve ser 0.")
        if self.pdays == -1 and self.poutcome != "unknown":
            raise ValueError("Histórico incoerente: cliente nunca contatado (dias = -1) não tem "
                             "resultado de campanha anterior. Use 'Desconhecido / sem campanha'.")
        if self.pdays > 0 and self.previous == 0:
            raise ValueError("Histórico incoerente: se houve contato há alguns dias, "
                             "'Contatos anteriores' deve ser pelo menos 1.")
        return self


# ------------------------------------------------------------------ aplicação
app = FastAPI(title="Propensão a Depósito a Prazo",
              description="Trabalho Final de Machine Learning - PUC Minas - Rafael Brasil",
              version="1.0.0")


@app.exception_handler(RequestValidationError)
async def erro_validacao(request: Request, exc: RequestValidationError):
    """Traduz os erros do Pydantic para mensagens amigáveis em português."""
    erros = []
    for e in exc.errors():
        campo = e["loc"][-1] if e["loc"] and e["loc"][-1] in ROTULOS else None
        rotulo = ROTULOS.get(campo, "Requisição")
        tipo = e["type"]
        if tipo == "missing":
            msg = "campo obrigatório não preenchido."
        elif tipo in ("int_parsing", "float_parsing", "int_type", "float_type"):
            msg = "informe um número válido (sem letras)."
        elif tipo == "int_from_float":
            msg = "informe um número inteiro."
        elif tipo in ("greater_than_equal", "less_than_equal") and campo in FAIXAS:
            lo, hi = FAIXAS[campo]
            msg = f"valor fora da faixa plausível ({lo} a {hi})."
        elif tipo == "extra_forbidden":
            rotulo, msg = "Requisição", f"campo desconhecido '{e['loc'][-1]}'."
        elif tipo == "json_invalid":
            msg = "o corpo da requisição não é um JSON válido."
        else:
            msg = str(e.get("msg", "valor inválido")).replace("Value error, ", "")
        erros.append({"campo": campo, "mensagem": f"{rotulo}: {msg}" if campo else msg})
    return JSONResponse(status_code=422, content={"erro": "Entrada inválida", "detalhes": erros})


@app.get("/", include_in_schema=False)
def pagina():
    return FileResponse(PASTA_APP / "index.html")


@app.get("/api/saude")
def saude():
    return {"status": "ok", "modelo": META["modelo"]}


@app.get("/api/metadados")
def metadados():
    """Informações que a página usa para montar o formulário e o painel do modelo."""
    return {"categorias": CATEGORIAS, "faixas": FAIXAS, "limiar": LIMIAR, "taxa_base": TAXA_BASE,
            "metricas_teste": META["metricas_teste"], "modelo": META["modelo"]}


@app.post("/api/prever")
def prever(cliente: Cliente):
    entrada = pd.DataFrame([cliente.model_dump()])[META["ordem_features"]]
    prob = probabilidade(entrada)
    prioriza = prob >= LIMIAR
    return {
        "probabilidade": round(prob, 4),
        "classe": int(prioriza),
        "rotulo": "Prioridade alta" if prioriza else "Prioridade baixa",
        "recomendacao": ("Priorizar a ligação: a chance deste cliente está acima da média de adesão."
                         if prioriza else
                         "Baixa prioridade: chance abaixo da média; ligar apenas se houver capacidade ociosa."),
        "limiar": LIMIAR,
        "criterio_limiar": META["criterio_limiar"],
        "taxa_base": TAXA_BASE,
        "vezes_taxa_base": round(prob / TAXA_BASE, 2),
        "faixa_teste": faixa_do_teste(prob),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
