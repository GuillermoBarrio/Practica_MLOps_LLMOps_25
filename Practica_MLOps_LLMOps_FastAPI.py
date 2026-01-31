# Empezamos cargardo las librerías, que tuve que cambiar en el caso de Google.

from fastapi import FastAPI
import os
# import google.generativeai as genai
from google import genai
from google.genai import types
from dotenv import load_dotenv
from transformers import pipeline

import pandas as pd
import numpy as np

# Cargo las funciones que definimos en el fichero Scripts.py, que hemos confeccionado anteriormente.

from Scripts import Preprocesado_train_test, Normalizacion, random_forest_regressor

# Cargamos los datasets de airbnb, porque les utilizaremos en los módulos que siguen.

airbnb_train = pd.read_csv('airbnb_train.csv', sep=';', decimal='.')
airbnb_test = pd.read_csv('airbnb_test.csv', sep=';', decimal='.')

app = FastAPI()

# El primer módulo es simplemente el que calcula la suma de enteros desde 1 hasta el dado.

@app.get('/numeros')
def return_name(numero: int): 
    suma = 0.5 * numero * (numero + 1)
    return {'Message': f'Hola, la suma de los numeros de 1 a {numero} es {suma}'}


# En segundo módulo devuelve la media del precio por noche en cada distrito de Madrid.
# La media la calculamos tomando el fichero de train que hemos cargado antes.

@app.get('/media_por_noche')
def media(distrito: str): 

    distritos = ['Vicálvaro', 'Arganzuela', 'Latina', 'Retiro', 'Centro',
       'Villaverde', 'Salamanca', 'Carabanchel', 'Ciudad Lineal',
       'Puente de Vallecas', 'Fuencarral - El Pardo', 'Hortaleza',
       'Chamberí', 'Tetuán', 'Usera', 'Moncloa - Aravaca',
       'Villa de Vallecas', 'Chamartín', 'Barajas',
       'San Blas - Canillejas', 'Moratalaz']
    
    if distrito not in distritos:
        return {'Message': f'Nombre de distrito equivocado, elige entre {distritos}'}
    
    media = round(airbnb_train.loc[airbnb_train['District'] == distrito, ]['Price'].mean(), 1)

    return {'Message': f'El precio medio por noche en el distrito de {distrito}, es de {media} euros'}


# Para el siguiente módulo tenemos que preprocesar los datasets, y normalizarlos.

train, test = Preprocesado_train_test(airbnb_train, airbnb_test)
XtrainScaled, y_train, XtestScaled, y_test = Normalizacion(train, test)


# El tercer módulo modela mediante Random Forest, a partir de la profundidad y estimadores dados.
# El módulo devuelve las métricas de R2 y RMSE de tanto train como test.

@app.get('/random_forest_regressor')
def regresor(depth: int, estimators: int): 
    r2_train, r2_test, rmse_train, rmse_test = random_forest_regressor(depth, estimators, XtrainScaled, y_train, XtestScaled, y_test)

    return {'Message': f'Los R2 para train y test son {r2_train} y {r2_test}, mientras que los RMSE son {rmse_train} y {rmse_test}, respectivamente.'}


# El cuarto módulo define una pipeline de HuggingFace, concretamente zero-shot-classification.

@app.get('/zero-shot-classification')
def zero_shot_classification(text: str):
    pipe = pipeline("zero-shot-classification")
    labels = ['geography', 'sports', 'politics', 'technology']
    response =  pipe(text, candidate_labels = labels)

    return {'Value': response}


# Intenté definir otras dos pipelines, una de NER, y otro un traductor.
# En ambos casos recibí un error relacionado con ANGI que, por desgracia, no pude solucionar.

'''
@app.get('/translation_en_to_fr')
def translation(text: str):
    pipe = pipeline("translation_en_to_fr")
    response =  pipe('Hugging Face is creating a tool that democratizes AI.')

    return {'Value': response}

@app.get('/NER_extraction')
def ner_extraction(text: str):
    pipe = pipeline("ner")
    response =  pipe(text)

    return {'Value': response}
'''

# El quinto módulo utiliza el modelo de Gemini 2.5 Flash dos veces para el mismo texto (prompt) variando la temperatura.
# Tuve que cambiar la estructura del código, ya que ahora hace falta definir un cliente.

@app.get("/gemini_comparison") 
def gemini_flash(query: str): 
  load_dotenv()
  client = genai.Client()
  # genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

  temperature = 1
  temperature_2 = 0.35

  response = client.models.generate_content(
    model='gemini-2.5-flash-lite',
    contents= query,
    config= types.GenerateContentConfig(
      max_output_tokens= 500,
      top_k= 2,
      top_p= 0.5,
      temperature= temperature,
      #response_mime_type= 'application/json',
      #stop_sequences= ['\n'],
      # seed=42,
  ) )

  response_2 = client.models.generate_content(
    model='gemini-2.5-flash-lite',
    contents= query,
    config= types.GenerateContentConfig(
      max_output_tokens= 500,
      top_k= 2,
      top_p= 0.5,
      temperature= temperature_2,
      #response_mime_type= 'application/json',
      #stop_sequences= ['\n'],
      # seed=42,
  ) )


  return {'Respuestas' : f'El modelo con temperatura {temperature} contestó: {response.text}. ********** El modelo con temperatura {temperature_2} contestó: {response_2.text}'}

  









