[LEIAME.md](https://github.com/user-attachments/files/29184154/LEIAME.md)
# Agente Preditivo Especialista — Custo de Plano de Saúde

Projeto que unifica **exploração de dados + machine learning + IA generativa**.
A partir dos dados de um beneficiário (idade, sexo, IMC, nº de filhos, fumante, região),
o sistema prevê se o **custo do plano de saúde tende a ser ALTO ou BAIXO/NORMAL** e usa o
**Gemini** para explicar o resultado em linguagem natural.

Base de dados: `insurance.csv` (Kaggle). O problema, originalmente de regressão (prever o
valor `charges`), foi reformulado para **classificação binária** (corte na mediana do custo)
para permitir o uso do **Naive Bayes** e das métricas de classificação exigidas.

## Estrutura

```
agente-preditivo-saude/
├── trabalho_ia_classificacao.ipynb   # Etapa A: análise + treino dos 4 modelos
├── modelo_agente_preditivo.pkl       # melhor modelo exportado pelo notebook
├── backend.py                        # Etapa B: API FastAPI + agente Gemini
├── frontend.py                       # Etapa C: interface Streamlit
├── requirements.txt
├── .env.example                      # modelo de configuração da chave
└── .gitignore
```

## Como rodar (local, no VSCode)

### 1. Criar o ambiente e instalar dependências

```bash
# dentro da pasta do projeto
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Opcional, recomendado) Re-treinar o modelo no seu ambiente

Abra `trabalho_ia_classificacao.ipynb` no VSCode e rode todas as células ("Run All").
Isso regenera o `modelo_agente_preditivo.pkl` com a sua versão do scikit-learn, evitando
avisos de incompatibilidade de versão ao carregar o `.pkl`.

### 3. Configurar a chave do Gemini

```bash
# copie o exemplo e edite
cp .env.example .env        # (Windows: copy .env.example .env)
```

Abra `.env` e cole sua chave (pegue em https://aistudio.google.com/apikey):

```
GEMINI_API_KEY=sua_chave_aqui
GEMINI_MODEL=gemini-2.5-flash
```

> Sem chave o sistema **continua funcionando**: o backend retorna uma explicação local
> gerada a partir dos próprios números (útil para testar antes de ter a API).

### 4. Subir o backend (terminal 1)

```bash
uvicorn backend:app --reload
```

Acesse http://localhost:8000/docs para testar a API direto pelo Swagger.

### 5. Subir o frontend (terminal 2, com o venv ativado)

```bash
streamlit run frontend.py
```

Abre em http://localhost:8501. Preencha os dados e clique em **Prever custo**.

## Endpoints da API

| Método | Rota       | Descrição                                            |
|--------|------------|------------------------------------------------------|
| GET    | `/`        | status do serviço e modelo carregado                 |
| GET    | `/opcoes`  | valores válidos de cada variável categórica          |
| POST   | `/prever`  | recebe os dados, devolve predição + explicação do agente |
| GET    | `/docs`    | documentação interativa (Swagger)                    |

Exemplo de corpo do `POST /prever`:

```json
{
  "age": 45, "sex": "male", "bmi": 31.5,
  "children": 2, "smoker": "yes", "region": "southeast"
}
```

## Algoritmos e métricas

Foram comparados **Regressão Logística, KNN, MLP e Naive Bayes**, avaliados por
**acurácia, precisão, sensibilidade e especificidade**. O melhor modelo (maior acurácia)
é exportado automaticamente pelo notebook. Detalhes e análise crítica no relatório técnico.

---

## Diário de Bordo de Contribuições

> Cada integrante descreve o que fez ao longo dos 15 dias. Preencham com as datas/tarefas reais.

### [Integrante 1 — Nome]
- dd/mm — ...
- dd/mm — ...

### [Integrante 2 — Nome]
- dd/mm — ...
- dd/mm — ...

### [Integrante 3 — Nome]
- dd/mm — ...
- dd/mm — ...
