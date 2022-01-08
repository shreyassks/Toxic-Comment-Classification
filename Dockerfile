FROM python:3.7
WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt --no-cache-dir
RUN python3 -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
COPY ./app /code/app
COPY ./src/data_cleaning.py /code/data_cleaning.py
COPY ./app/language_detect.py /code/language_detect.py
COPY ./app/logger.py /code/logger.py
COPY ./models/tokenizer.pkl /code/tokenizer.pkl
COPY ./models/toxicity_classifier.h5 /code/toxicity_classifier.h5
COPY lid.176.bin /code/lid.176.bin
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
