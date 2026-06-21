"""
Interface web do Agente Preditivo Especialista — Etapa C
Streamlit que coleta os dados, chama o backend e mostra o resultado
bruto do modelo + a explicacao do agente inteligente.

Rodar (com o backend ja no ar):
    streamlit run frontend.py
"""
import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Agente Preditivo de Custo de Saude", page_icon="🩺")

st.title("🩺 Agente Preditivo de Custo de Plano de Saude")
st.caption(
    "Informe os dados e o modelo estima se o custo tende a ser **ALTO** ou "
    "**BAIXO/NORMAL**. Em seguida, um agente inteligente explica o resultado."
)


# --- buscar opcoes validas direto do backend -------------------------------
@st.cache_data(show_spinner=False)
def carregar_opcoes():
    r = requests.get(f"{BACKEND_URL}/opcoes", timeout=10)
    r.raise_for_status()
    return r.json()["categorias"]


try:
    categorias = carregar_opcoes()
    backend_ok = True
except Exception as e:
    backend_ok = False
    st.error(
        f"Nao consegui falar com o backend em {BACKEND_URL}.\n\n"
        f"Inicie-o com `uvicorn backend:app --reload` e recarregue a pagina.\n\nDetalhe: {e}"
    )
    categorias = {
        "sex": ["female", "male"],
        "smoker": ["no", "yes"],
        "region": ["northeast", "northwest", "southeast", "southwest"],
    }

# --- formulario ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Idade", min_value=0, max_value=120, value=35, step=1)
    bmi = st.number_input("IMC (indice de massa corporal)", min_value=10.0,
                          max_value=70.0, value=27.5, step=0.1, format="%.1f")
    children = st.number_input("Numero de filhos", min_value=0, max_value=20, value=0, step=1)
with col2:
    sex = st.selectbox("Sexo", categorias["sex"])
    smoker = st.selectbox("Fumante", categorias["smoker"])
    region = st.selectbox("Regiao", categorias["region"])

st.divider()

if st.button("🔮 Prever custo", type="primary", use_container_width=True, disabled=not backend_ok):
    payload = {"age": int(age), "sex": sex, "bmi": float(bmi),
               "children": int(children), "smoker": smoker, "region": region}
    try:
        with st.spinner("Consultando o modelo e o agente..."):
            r = requests.post(f"{BACKEND_URL}/prever", json=payload, timeout=60)
        if r.status_code != 200:
            st.error(f"Erro {r.status_code}: {r.text}")
        else:
            res = r.json()
            prob = res["probabilidade_alto"]

            # resultado bruto do modelo
            if res["classe"] == 1:
                st.error(f"### Resultado: {res['classe_label']}")
            else:
                st.success(f"### Resultado: {res['classe_label']}")

            mcol1, mcol2 = st.columns(2)
            mcol1.metric("Probabilidade de custo ALTO", f"{prob:.1%}")
            mcol2.metric("Modelo utilizado", res["modelo"])
            st.progress(prob)

            # explicacao do agente
            fonte = "Gemini" if res["fonte_explicacao"] == "gemini" else "explicacao local (sem API key)"
            st.subheader("🤖 Explicacao do agente")
            st.caption(f"Fonte: {fonte}")
            st.write(res["explicacao"])

            with st.expander("Ver resposta bruta (JSON)"):
                st.json(res)
    except Exception as e:
        st.error(f"Falha na requisicao: {e}")

st.divider()
st.caption(
    "Projeto academico — Agente Preditivo Especialista (UNOESC). "
    "Estimativa estatistica, sem finalidade clinica."
)
