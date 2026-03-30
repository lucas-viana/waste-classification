# Classificacao de Residuos - Trabalho IA

Aplicacao web completa para deploy de um modelo de classificacao de residuos treinado com TensorFlow/Keras.

## Estrutura

- `backend/`: API em FastAPI para inferencia
  - `models/modelo_residuos.h5`: modelo treinado no Colab
- `frontend/`: aplicacao web em Vue.js

## Requisitos atendidos do trabalho

1. Treinamento do modelo feito com TensorFlow e Keras (arquivo `backend/models/modelo_residuos.h5`).
2. Deploy em aplicacao web com:
   - Backend FastAPI para predicao
   - Frontend Vue.js para upload e visualizacao do resultado

## 1) Rodar o backend (FastAPI)

No terminal:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API disponivel em `http://localhost:8000`.

Endpoints principais:

- `GET /api/health`
- `GET /api/model-info`
- `POST /api/predict` (campo `file` com imagem)

### Configuracao opcional do backend

Copie `backend/.env.example` para `.env` (opcional):

- `MODEL_PATH`: caminho para o arquivo `.h5`
- `WASTE_CLASS_NAMES`: nomes das classes separados por virgula. Exemplo: `organic,recyclable`
- `RESCALE_INPUT`: `true` para dividir pixels por 255
- `CORS_ORIGINS`: origens do frontend

## 2) Rodar o frontend (Vue + Vite)

Em outro terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend disponivel em `http://localhost:5173`.

### Configuracao opcional do frontend

- Arquivo `frontend/.env.example`
- Se quiser apontar para outra API, configure:
  - `VITE_API_BASE_URL=http://localhost:8000`

Se vazio, o projeto usa proxy do Vite para `http://localhost:8000` no caminho `/api`.

## Como usar

1. Abra o frontend no navegador.
2. Clique em **Choose Image**.
3. Clique em **Run Prediction**.
4. Veja a classe prevista e as probabilidades.

## Observacoes

- Se os nomes das classes estiverem diferentes do seu treinamento, ajuste `WASTE_CLASS_NAMES` no backend.
- Se o seu preprocessing no treinamento foi diferente, ajuste `RESCALE_INPUT` e/ou preprocessamento em `backend/app/main.py`.

## 3) Executar com Docker

Importante: rode os comandos a partir da raiz do projeto (`classificacao-residuos`), pois os Dockerfiles usam contexto da raiz para copiar backend e frontend.

### Build da imagem do backend

```powershell
docker build -f backend/Dockerfile -t waste-backend:1.0 .
```

### Rodar backend em container

```powershell
docker run --rm -p 8000:8000 --name waste-backend waste-backend:1.0
```

### Build da imagem do frontend

Se backend e frontend forem publicados separados, informe a URL publica do backend no build:

```powershell
docker build -f frontend/Dockerfile -t waste-frontend:1.0 --build-arg VITE_API_BASE_URL=https://SEU_BACKEND.azurecontainerapps.io .
```

### Rodar frontend em container

```powershell
docker run --rm -p 5173:80 --name waste-frontend waste-frontend:1.0
```

Acesse o frontend em `http://localhost:5173`.

## 4) Publicar imagens na Azure (ACR)

Exemplo com Azure Container Registry (ACR):

```powershell
az login
az group create --name rg-residuos --location brazilsouth
az acr create --resource-group rg-residuos --name meuacrresiduos --sku Basic
az acr login --name meuacrresiduos

docker tag waste-backend:1.0 meuacrresiduos.azurecr.io/waste-backend:1.0
docker push meuacrresiduos.azurecr.io/waste-backend:1.0

docker tag waste-frontend:1.0 meuacrresiduos.azurecr.io/waste-frontend:1.0
docker push meuacrresiduos.azurecr.io/waste-frontend:1.0
```

Depois disso, voce pode usar essas imagens no Azure Container Apps, Azure App Service for Containers ou AKS.
