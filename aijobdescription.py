#funcion para llamar a openai desde la api y que devuelva solamente la respuesta. Para correr "python aicontent.py"
import os
import openai
import config
import pinecone
import pandas as pd
from tqdm.auto import tqdm
import json
from openai.embeddings_utils import get_embedding
import matplotlib

import json

openai.api_key = "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"

def jobDescription(query):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt="Genera un texto que describa el siguiente puesto de trabajo de manera formal de cara a ser publicado en linkedin para contratar gente:{} ".format(query),
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
    if "choices" in response:
        if len(response['choices']) > 0:
                answer = response['choices'][0]['text']
        else: 
                "Lo siento, no tengo respuestas para ti, intenta reformular tu pregunta"

    else:
        "Lo siento, no tengo respuestas para ti, intenta reformular tu pregunta"

    return answer