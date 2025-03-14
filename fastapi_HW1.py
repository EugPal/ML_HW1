from fastapi import FastAPI, UploadFile, File, Response
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import pickle
import io
import re

app = FastAPI()


# Загрузка обученной модели линейной регрессии и стандартизатора.
with open("linear_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def clean_numeric(value: str) -> float:
    """
    Убирает из строки всё, кроме цифр и точки, и возвращает число типа float.
    Если преобразование невозможно, возвращает np.nan.
    """
    cleaned = re.sub(r"[^0-9.]", "", value)
    try:
        return float(cleaned)
    except ValueError:
        return np.nan

def prepare_features(item: Item) -> pd.DataFrame:
    """
    Извлекает необходимые признаки для модели из объекта Item.
    Предполагается, что для предсказания используются признаки:
      - year, km_driven (уже числовые)
      - mileage, engine, max_power (перед преобразованием они могут содержать единицы измерения)
      - seats
    """
    features = {
        "year": item.year,
        "km_driven": item.km_driven,
        "mileage": clean_numeric(item.mileage) if isinstance(item.mileage, str) else item.mileage,
        "engine": clean_numeric(item.engine) if isinstance(item.engine, str) else item.engine,
        "max_power": clean_numeric(item.max_power) if isinstance(item.max_power, str) else item.max_power,
        "seats": item.seats
    }
    return pd.DataFrame([features])


def predict_price_for_item(item: Item) -> float:
    """
    Обрабатывает один объект, стандартизирует признаки и возвращает предсказанную стоимость.
    """
    df_features = prepare_features(item)
    features_scaled = scaler.transform(df_features)
    prediction = model.predict(features_scaled)
    return float(prediction[0])

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """
    Принимает JSON с признаками одного объекта и возвращает предсказанную стоимость автомобиля.
    """
    return predict_price_for_item(item)


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Принимает CSV-файл с признаками тестовых объектов, делает предсказания и возвращает CSV-файл с добавленным столбцом 'prediction'.
    Предполагается, что CSV содержит столбцы: year, km_driven, mileage, engine, max_power, seats.
    """
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    # Приводим столбцы, содержащие единицы измерения, к числовому виду
    df["mileage"] = df["mileage"].apply(lambda x: clean_numeric(str(x)))
    df["engine"] = df["engine"].apply(lambda x: clean_numeric(str(x)))
    df["max_power"] = df["max_power"].apply(lambda x: clean_numeric(str(x)))
    
    # Извлекаем нужные признаки для модели
    features = df[["year", "km_driven", "mileage", "engine", "max_power", "seats"]]
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    df["prediction"] = predictions
    
    output_csv = df.to_csv(index=False)
    return Response(content=output_csv, media_type="text/csv")
