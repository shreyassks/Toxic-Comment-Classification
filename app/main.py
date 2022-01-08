from data_cleaning import clean_text

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import logging
from logging.config import dictConfig
from logger import LogConfig, Toxicity

from language_detect import LanguageIdentification
from fastapi import FastAPI

import requests, json, pickle, os, sys
from typing import Optional
from datetime import datetime

dictConfig(LogConfig().dict())
logger = logging.getLogger("Khoros_demo")

app = FastAPI()
language = LanguageIdentification()
time = datetime.utcnow()

logger.info("Loading Tokenizer and LSTM Model")
model = load_model("toxicity_classifier.h5")
with open("tokenizer.pkl", 'rb') as handle:
    tokenizer = pickle.load(handle)


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
    logger.info("Entered into Fake API for testing, returning hardcoded values")
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


@app.get("/api/toxicity-detection/{company}/{comment}")
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
        logger.info("Language was not mentioned in request, Predicting the language using Fast API pretrained model")
        prediction = language.predict_lang(comment)
        # extract predicted labels + accuracy
        labels, accuracy_list = prediction
        # convert tuple into list
        label_list = list(labels)
        # strip __label__ in predictions
        lang = [label.replace("__label__", "") for label in label_list]
        # get accuracy scores
        accuracy_scores = [float(acc) for acc in accuracy_list]

        logger.info("detected language: {}".format(lang))
        logger.info("accuracy score:    {}".format(accuracy_scores))

    try:
        payload = {"message": comment}
        result = requests.post(url="http://127.0.0.1:80/predict",
                               data=json.dumps(payload),
                               headers={'Content-Type': 'application/json'})

        logger.info("Received response from /predict API")
        response = result.json()

        logger.info("Classifying the comment as toxic or non-toxic based on set threshold")
        if float(response["Neutral"]) > 0.5:
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

    except Exception as e:
        logger.error(f"Error has Occurred - {str(e)}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_num = exc_tb.tb_lineno
        logger.error(f"{exc_type} has occurred in filename - {filename} at Line number - {line_num}")


@app.post("/predict")
def predict(comment: Toxicity):
    """
    Makes a prediction call to the NLP Model to get the level of Toxicity
    :param comment: Review Comment
    :return: JSON Response with Toxicity levels predicted by NLP model
    """
    try:
        logger.info("Entered /predict API")
        input_comment = clean_text(comment.message)
        input_comment = input_comment.split(" ")
        logger.info("Performed data cleaning")
        sequences = tokenizer.texts_to_sequences(input_comment)
        sequences = [[item for sublist in sequences for item in sublist]]

        padded_data = pad_sequences(sequences, maxlen=200)
        logger.info("Completed tokenization and padding")
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
    except Exception as e:
        logger.error(f"Error has Occurred - {str(e)}")
        exc_type, exc_obj, exc_tb = sys.exc_info()
        filename = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        line_num = exc_tb.tb_lineno
        logger.error(f"{exc_type} has occurred in filename - {filename} at Line number - {line_num}")


# if __name__ == '__main__':
#     uvicorn.run(app, port=8080, host='localhost')
