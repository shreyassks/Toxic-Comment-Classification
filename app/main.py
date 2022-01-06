from data_cleaning import clean_text

import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from fastapi import FastAPI
from pydantic import BaseModel
import requests
from langdetect import detect
from typing import Optional
from datetime import datetime
import json
import nltk
nltk.download('omw-1.4')

app = FastAPI()
# params = yaml.safe_load(open("params.yaml"))["data-preprocess"]
# sequence_len = params["sequence_len"]
time = datetime.utcnow()

model = load_model("toxicity_classifier.h5")
with open("tokenizer.pkl", 'rb') as handle:
    tokenizer = pickle.load(handle)


class Toxicity(BaseModel):
    message: str


@app.get("/")
async def root():
    return "Toxic Comment Classification"


@app.get("/api/toxicity-detection-fake")
async def health(company="Samsung", comment="Khoros is awesome", start=time, lang="en"):
    """
    Fake API implementation with hard coded values to test the API
    :param company: Customer Organization
    :param comment: Review Comment
    :param start: Time in ISO format when message is sent to API
    :param lang: Language of the sent comment
    :return: JSON Response with hardcoded predictions
    """
    return \
        {
           "company": company,
           "message": comment,
           "is_toxic": "True",
           "time": start,
           "language": lang,
           "toxicity_levels": {
               "toxic": 1,
               "severe_toxic": 1,
               "obscene": 1,
               "threat": 1,
               "insult": 1,
               "identity_hate": 1
           }
        }


@app.get("/api/toxicity-detection/{company}/{comment}", status_code=200)
def comment_request(company: str, comment: str, start: datetime = time, lang: Optional[str] = None):
    """
    An Extended Implementation of the fake API, It will fetch the client data and route to /predict API to get the model response
    :param company: Customer Organization
    :param comment: Review Comment
    :param start: Time in ISO format when message is sent to API
    :param lang: Language of the sent comment
    :return: JSON Response with model predictions
    """
    if lang is None:
        lang = detect(comment)

    payload = {"message": comment}
    result = requests.post(url="http://127.0.0.1:80/predict",
                           data=json.dumps(payload),
                           headers={'Content-Type': 'application/json'})

    response = result.json()
    if response["Neutral"] > 0.5:
        toxic = False
    else:
        toxic = True

    return {
        "company": company,
        "message": comment,
        "is_toxic": toxic,
        "time": start,
        "language": lang,
        "toxicity_levels": {
           "toxic": response["Toxic"],
           "severe_toxic": response["Very Toxic"],
           "obscene": response["Obscene"],
           "threat": response["Threat"],
           "insult": response["Insult"],
           "identity_hate": response["Hate"]
        }
    }


@app.post("/predict")
def predict(comment: Toxicity):
    """
    Makes a prediction call to the NLP Model to get the level of Toxicity
    :param comment: Review Comment
    :return: JSON Response with Toxicity levels predicted by NLP model
    """

    input_comment = clean_text(comment.message)
    input_comment = input_comment.split(" ")

    sequences = tokenizer.texts_to_sequences(input_comment)
    sequences = [[item for sublist in sequences for item in sublist]]

    padded_data = pad_sequences(sequences, maxlen=150)
    result = model.predict(padded_data, len(padded_data), verbose=1)

    return {
            "Toxic": str(result[0][0]),
            "Very Toxic": str(result[0][1]),
            "Obscene": str(result[0][2]),
            "Threat": str(result[0][3]),
            "Insult": str(result[0][4]),
            "Hate": str(result[0][5]),
            "Neutral": str(result[0][6])
        }


# if __name__ == '__main__':
#     uvicorn.run(app, host="127.0.0.1", port=8000)
#     # res = make_prediction(input_comment="COCKSUCKER BEFORE YOU PISS AROUND ON MY WORK")
