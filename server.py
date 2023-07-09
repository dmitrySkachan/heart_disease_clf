from fastapi import FastAPI
import uvicorn
from ml import predict

app = FastAPI()

@app.get('/predict')
def get_prediction(Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
       Sex_M, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
       RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y,
       ST_Slope_Flat, ST_Slope_Up):
    sample = [Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak,
       Sex_M, ChestPainType_ATA, ChestPainType_NAP, ChestPainType_TA,
       RestingECG_Normal, RestingECG_ST, ExerciseAngina_Y,
       ST_Slope_Flat, ST_Slope_Up]
    res = predict([sample])
    return res

if __name__ == '__main__':
    uvicorn.run(app, host= 'localhost', port=8000)
