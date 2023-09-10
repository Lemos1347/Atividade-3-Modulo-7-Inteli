from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd
import uvicorn

app = FastAPI()

# Carregando o modelo e o scaler
model = load('modelo_regressao.joblib')

class PredictionData(BaseModel):
    gender: str
    age: int
    annual_income: int

@app.post("/predict")
async def predict_spending_score(data: PredictionData):
    # Convertendo o gênero de string para 0 ou 1
   if data.gender.lower() in ["female", "mulher", "feminino"]:
      data.gender = 0
   elif data.gender.lower() in ["male", "homem", "masculino"]:
      data.gender = 1
   else:
      raise HTTPException(status_code=400, detail="Gênero inválido. Por favor, use 'male'/'homem' ou 'female'/'mulher'.")
   try:     
        # Convertendo dados para DataFrame
        input_data = pd.DataFrame([dict(data)])
        
        # Realizando previsão
        prediction = model.predict(input_data)
        
        return {"predicted_spending_score": prediction[0]}
   except:
        raise HTTPException(status_code=400, detail="Error in prediction.")
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)