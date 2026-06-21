"""
Backend do Agente Preditivo Especialista — Etapa B
FastAPI que carrega o modelo treinado (.pkl), faz a predicao e usa o
Gemini para explicar o resultado em linguagem natural.

Rodar:
    uvicorn backend:app --reload
    (ou)  python backend.py
"""
import os
from pathlib import Path

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Configuracao
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "modelo_agente_preditivo.pkl"

if not MODELO_PATH.exists():
    raise FileNotFoundError(
        f"Modelo nao encontrado em {MODELO_PATH}. "
        "Rode o notebook 'trabalho_ia_classificacao.ipynb' para gera-lo."
    )

bundle = joblib.load(MODELO_PATH)
modelo = bundle["modelo"]
modelo_nome = bundle["modelo_nome"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]
colunas = bundle["colunas"]
classes_label = bundle["classes"]            # {0: 'Custo Baixo/Normal', 1: 'Custo Alto'}
ponto_corte = bundle["ponto_corte_charges"]

# indice da classe positiva (1 = custo alto) em predict_proba
idx_classe_alto = list(modelo.classes_).index(1)

# ---------------------------------------------------------------------------
# Agente Gemini
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """Voce e um assistente especialista que explica, em portugues claro e \
acessivel, o resultado de um modelo de machine learning. Esse modelo estima se o CUSTO \
de plano de saude de uma pessoa tende a ser ALTO ou BAIXO/NORMAL.

Regras obrigatorias:
- Baseie-se EXCLUSIVAMENTE nos dados e no resultado numerico fornecidos na mensagem. \
NAO invente numeros, fatos ou informacoes que nao estejam ali.
- NAO forneca diagnosticos, conselhos medicos ou recomendacoes de saude. O modelo e \
apenas estatistico/atuarial, nao clinico.
- Explique de forma fundamentada quais caracteristicas informadas provavelmente mais \
influenciaram o resultado (ex.: ser fumante, IMC elevado, idade avancada).
- Deixe claro que e uma estimativa probabilistica, e nao uma certeza.
- Seja conciso: no maximo 2 paragrafos curtos, linguagem simples.
- Se algum dado parecer ausente ou inconsistente, aponte isso em vez de supor."""


def explicar_com_gemini(resultado: dict, dados: dict) -> tuple[str, str]:
    """Retorna (texto_explicacao, fonte). Cai para explicacao local se nao houver key."""
    if not GEMINI_API_KEY:
        return _explicacao_local(resultado, dados), "fallback_local"

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        prompt = f"""Resultado do modelo:
- Predicao: {resultado['classe_label']} (classe {resultado['classe']})
- Probabilidade estimada de custo ALTO: {resultado['probabilidade_alto']:.1%}
- Modelo utilizado: {resultado['modelo']}

Dados informados pelo usuario:
- Idade: {dados['age']} anos
- Sexo: {dados['sex']}
- IMC: {dados['bmi']}
- Numero de filhos: {dados['children']}
- Fumante: {dados['smoker']}
- Regiao: {dados['region']}

Explique esse resultado para o usuario final seguindo as regras."""

        resp = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=600,
            ),
        )
        texto = (resp.text or "").strip()
        if not texto:
            return _explicacao_local(resultado, dados), "fallback_local"
        return texto, "gemini"
    except Exception as e:  # rede/cota/key invalida -> nao derruba a API
        return (
            _explicacao_local(resultado, dados)
            + f"\n\n(Obs.: o agente Gemini nao pode ser consultado: {e})"
        ), "fallback_local"


def _explicacao_local(resultado: dict, dados: dict) -> str:
    """Explicacao determinística construida a partir dos numeros (sem LLM)."""
    fatores = []
    if str(dados["smoker"]).lower() in ("yes", "sim"):
        fatores.append("ser fumante (fator de maior peso no modelo)")
    if float(dados["bmi"]) >= 30:
        fatores.append(f"IMC elevado ({dados['bmi']}, faixa de obesidade)")
    if int(dados["age"]) >= 50:
        fatores.append(f"idade mais avancada ({dados['age']} anos)")
    fatores_txt = ", ".join(fatores) if fatores else "o conjunto geral das variaveis informadas"

    return (
        f"O modelo classificou este perfil como '{resultado['classe_label']}', "
        f"com probabilidade estimada de {resultado['probabilidade_alto']:.1%} de o custo ser ALTO. "
        f"Os elementos que mais tendem a puxar essa estimativa neste caso sao: {fatores_txt}. "
        "Trata-se de uma estimativa estatistica baseada em padroes historicos, "
        "e nao de uma certeza nem de uma avaliacao clinica."
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
app = FastAPI(title="Agente Preditivo Especialista", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


class DadosEntrada(BaseModel):
    age: int = Field(..., ge=0, le=120, description="Idade em anos")
    sex: str = Field(..., description="male ou female")
    bmi: float = Field(..., gt=0, lt=100, description="Indice de massa corporal")
    children: int = Field(..., ge=0, le=20, description="Numero de filhos/dependentes")
    smoker: str = Field(..., description="yes ou no")
    region: str = Field(..., description="northeast, northwest, southeast ou southwest")


@app.get("/")
def raiz():
    return {
        "servico": "Agente Preditivo Especialista",
        "modelo": modelo_nome,
        "gemini_ativo": bool(GEMINI_API_KEY),
        "endpoints": ["/opcoes", "/prever (POST)", "/docs"],
    }


@app.get("/opcoes")
def opcoes():
    """Valores validos de cada variavel categorica (do proprio modelo)."""
    return {
        "features": colunas,
        "categorias": {col: list(enc.classes_) for col, enc in encoders.items()},
    }


@app.post("/prever")
def prever(dados: DadosEntrada):
    entrada = dados.model_dump()

    # validar categorias contra o que o modelo conhece
    for col, enc in encoders.items():
        if str(entrada[col]) not in list(enc.classes_):
            raise HTTPException(
                status_code=422,
                detail=f"Valor invalido para '{col}': {entrada[col]}. "
                f"Validos: {list(enc.classes_)}",
            )

    # pre-processamento identico ao do treino
    df = pd.DataFrame([entrada])
    for col, enc in encoders.items():
        df[col] = enc.transform(df[col].astype(str))
    df = df[colunas]
    X = scaler.transform(df)

    classe = int(modelo.predict(X)[0])
    prob_alto = float(modelo.predict_proba(X)[0][idx_classe_alto])

    resultado = {
        "classe": classe,
        "classe_label": classes_label[classe],
        "probabilidade_alto": prob_alto,
        "modelo": modelo_nome,
        "ponto_corte_charges": ponto_corte,
    }

    explicacao, fonte = explicar_com_gemini(resultado, entrada)
    resultado["explicacao"] = explicacao
    resultado["fonte_explicacao"] = fonte
    return resultado


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
