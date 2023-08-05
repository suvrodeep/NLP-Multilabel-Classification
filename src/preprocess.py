import re
from text_preprocessing import to_lower, remove_phone_number, remove_url, remove_email, remove_punctuation, \
    remove_special_character, remove_itemized_bullet_and_numbering
from text_preprocessing import preprocess_text
import pandas as pd

import spacy

spacy.prefer_gpu()


def remove_non_latin(text):
    return re.sub(pattern='[^\u0000-\u05C0\u2100-\u214F]+', repl='', string=text)


class PreProcess:

    def __init__(self, text=None):
        self.text = text

    def common_cleaning_functions(self, data=None) -> pd.Series:
        if data is None:
            data = self.text

        if type(data) == pd.Series:
            data = pd.Series(data)

        preprocess_functions = [to_lower, remove_phone_number, remove_url, remove_email, remove_punctuation,
                                remove_special_character, remove_itemized_bullet_and_numbering]
        processed_text = data.apply(preprocess_text, processing_function_list=preprocess_functions)
        processed_text = processed_text.apply(remove_non_latin)

        return pd.Series(processed_text)

    def spacy_remove_stop_words(self, data=None) -> pd.Series:
        if data is None:
            data = self.text

        if type(data) in [list, pd.Series]:
            if type(data) == pd.Series:
                data = data.tolist()
            else:
                pass
            blank_pipe = spacy.blank('en')
            processed_text = []
            for doc in blank_pipe.pipe(texts=data):
                tokens = []
                for token in doc:
                    if token.is_stop or token.is_space:
                        pass
                    else:
                        tokens.append(token.text)
                processed_text.append(' '.join(tokens))
        else:
            processed_text = None

        return pd.Series(processed_text)

    def spacy_lemmatizer(self, data=None, model=None):
        if data is None:
            data = self.text
        if model is None:
            model = 'en_core_web_lg'

        if model == 'en_core_web_trf':
            disable_components = ['parser', 'ner']
        else:
            disable_components = ['parser', 'senter', 'ner']

        if type(data) in [list, pd.Series]:
            if type(data) == pd.Series:
                data = data.tolist()
            else:
                pass
            pipeline = spacy.load(model, disable=disable_components)
            processed_text = []
            for doc in pipeline.pipe(texts=data):
                lemmas = []
                for token in doc:
                    lemmas.append(token.lemma_)
                processed_text.append(" ".join(lemmas))
        else:
            processed_text = None

        return pd.Series(processed_text)

