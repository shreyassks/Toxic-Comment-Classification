FROM python:3.7
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY ./app /code/app
COPY ./src/data_cleaning.py /code/data_cleaning.py
COPY ./models/tokenizer.pkl /code/tokenizer.pkl
COPY ./models/toxicity_classifier.h5 /code/toxicity_classifier.h5
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
