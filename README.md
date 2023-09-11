# 📝 Atividade-3-Modulo-7-Inteli -> Previsão do "Spending Score"

## 📺 Demo

https://github.com/Lemos1347/Atividade-3-Modulo-7-Inteli/assets/99190347/d925633e-7499-4e7c-9a59-6e4879b6518b

## 📊 Preparação dos Dados

Para a criação desse modelo, foi utilizado o conjunto de dados do `kaggle` de [Customer Segmentation](https://www.kaggle.com/datasets/joebeachcapital/customer-segmentation?resource=download). O conjunto de dados contém 200 registros de clientes com 5 atributos: `customerID`, `gender`, `age`, `annual_income` e `spending_score`. O objetivo é prever o `spending_score` de um cliente com base nos outros atributos. Depois de carregado os dados apropriadamente utilizando a biblioteca `pandas`, foi feita a seguinte transformação dos dados:

- O atributo `gender` foi transformado em um atributo numérico, onde `1` representa `Male` e `0` representa `Female`.
- O atributo `customerID` foi removido, pois não é relevante para o modelo.

**_obs.: os demais dados não foram normalizados pois devido a quantidade de dados e a sua natureza não julguei necessário._**

Por fim, os dados foram divididos em dados de treinamento e dados de teste, onde 20% dos dados foram separados para teste. O código completo pode ser encontrado [aqui](/model.ipynb).

## 🛠 Treinamento

Para esse caso foi escolhido o modelo `Linear Regression` do `skitlearn` pois o objetivo era prever um valor contínuo - o "spending score", e a regressão linear é uma das abordagens mais diretas para modelar uma relação linear entre um conjunto de variáveis independentes e uma variável dependente. O modelo foi treinado com os dados de treinamento e avaliado com os dados de teste. O código completo pode ser encontrado [aqui](/model.ipynb).

## 🚀 API FastAPI

Para a criação da API foi utilizado o framework `FastAPI` do `python`. O modelo foi exportado utilizando a biblioteca `joblib`, então a utilizamos para carregá-lo na API.
A API possui apenas um endpoint `predict` que recebe os seguintes dados no "body" da request:

- "gender": homem/male ou mulher/female,
- "age": idade em anos,
- "annual_income": em milhares de dólares por ano (170 representaria $170.000 por ano)

Caso a requisição proceda corretamente, a API retorna o "spending score" previsto para o cliente de acordo com o que o modelo informa. O código completo pode ser encontrado [aqui](/main.py).

## 🐳 Docker

A API foi containerizada utilizando o `Docker`. O arquivo `Dockerfile` utiliza como base a imagem oficial do python, baixa todas as bibliotecas corretamente informadas no arquivo `requirements.txt` e por fim roda a aplicação. O arquivo completo pode ser encontrado [aqui](/Dockerfile) e o link para a imagem no Docker hub é:
[https://hub.docker.com/repository/docker/lemos12/predictive-model-docker/general](https://hub.docker.com/repository/docker/lemos12/predictive-model-docker/general).

## 📦 Deploy

A API foi deployada na `AWS` utilizando a conta `AWS Acadamy` providenciada para mim. Caso queira testar a API, basta acessar o link [http://18.211.126.21:8000/predict](http://18.211.126.21:8000/predict) e fazer a requisição com o "body" correto.
**_Obs.: caso não consiga acessar o serviço, isso significa que a instância passou das 4h que é permitida ficar ativa._**

## 💻 Como rodar

Caso você queira rodar o projeto localmente, é possível de duas formas:

### 1º Docker hub

Basta rodar os seguintes comandos em seu terminal:

```bash
docker pull lemos12/predictive-model-docker
```

```bash
docker run -p 8000:8000 lemos12/predictive-model-docker
```

Pronto! Agora você pode acessar a API através do link [http://localhost:8000/predict](http://localhost:8000/predict).

### 2º Clonando o repositório

Baixe o repositório e entre na pasta do projeto. Em seguida, rode os seguintes comandos em seu terminal:

```bash
pip install -r requirements.txt
```

```bash
python3 main.py
```

Pronto! Agora você pode acessar a API através do link [http://localhost:8000/predict](http://localhost:8000/predict).
