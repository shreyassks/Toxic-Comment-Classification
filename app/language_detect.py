import fasttext
from pathlib import Path


class LanguageIdentification:

    def __init__(self):
        pretrained_lang_model = str(Path().absolute()) + "/lid.176.bin"
        self.model = fasttext.load_model(pretrained_lang_model)

    def predict_lang(self, text):
        # returns top 1 matching languages
        predictions = self.model.predict(text, k=1)
        return predictions
