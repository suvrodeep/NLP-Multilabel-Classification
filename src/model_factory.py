from preprocess import PreProcess

import os
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from spacy.training import Example
from spacy.tokens import DocBin
from spacy.cli.evaluate import evaluate
from spacy.cli.train import train
from spacy.util import load_model
from tqdm import tqdm
from scipy.sparse import hstack

import spacy
spacy.prefer_gpu()


def make_dirs():
    path_list = ["../data", "../outputs", "../trained_pipelines"]

    for path in path_list:
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            continue


class BaselineModel:

    def __init__(self, data=None, model_type=None):
        make_dirs()
        self.df = data
        self.model_name = model_type

        self.train_x, self.val_x, self.train_y, self.val_y, self.test_df = self.data_split()

    def data_split(self):
        data = self.df

        preprocess = PreProcess()
        data['clean_description'] = preprocess.spacy_remove_stop_words(
            data=preprocess.common_cleaning_functions(data=data['description']))
        data['clean_title'] = preprocess.spacy_remove_stop_words(
            data=preprocess.common_cleaning_functions(data=data['title']))

        train_df = data.loc[~data['level'].isna()]
        test_df = data.loc[data['level'].isna()]

        train_x, val_x, train_y, val_y = train_test_split(self.get_word_vectors(data=train_df),
                                                          train_df['level'],
                                                          test_size=0.2, random_state=3137)

        return train_x, val_x, train_y, val_y, test_df

    def get_word_vectors(self, data):
        if data is None:
            data = self.test_df

        description_list = data['clean_description'].tolist()
        title_list = data['clean_title'].tolist()

        count_vect = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 3), max_df=0.9, min_df=0.1)
        tfidf_transformer = TfidfTransformer()

        word_counts_desc = count_vect.fit_transform(description_list)
        tfidf_matrix_desc = tfidf_transformer.fit_transform(word_counts_desc)
        word_counts_title = count_vect.fit_transform(title_list)
        tfidf_matrix_title = tfidf_transformer.fit_transform(word_counts_title)

        tfidf_matrix = hstack([tfidf_matrix_desc, tfidf_matrix_title])

        return tfidf_matrix

    def model(self):
        if self.model_name == 'NaiveBayesClassifier':
            model = MultinomialNB().fit(self.train_x, self.train_y)
        elif self.model_name == 'SupportVectorClassifier':
            model = SVC().fit(self.train_x, self.train_y)
        else:
            print("Unrecognized baseline model name. Name should be in [NaiveBayesClassifier, SupportVectorClassifier]")
            model = None

        return model

    def predict(self, data=None):
        if data is None:
            data = self.test_df

        model_obj = self.model()
        y_predicted = []
        if model_obj is None:
            print("No model object found")
        else:
            y_predicted = model_obj.predict(data).tolist()

        return y_predicted

    def performance_metrics(self):
        y_predicted = self.predict(data=self.val_x)

        labels = list(set(self.val_y.tolist()))
        report = classification_report(y_true=self.val_y, y_pred=y_predicted, labels=labels)

        print(f"Classification report for {self.model_name}:\n {report}")


class SpacyModel:
    def __init__(self, data=None, model_name=None, train_file=None, val_file=None, test_file=None, config_file=None):
        make_dirs()
        self.df = data
        self.model_name = model_name
        self.model_output_path = "../trained_pipelines/"
        self.pred_output_file = "../outputs/test_pred.spacy"

        if config_file is None:
            self.config_file = "../config/full_config.cfg"
        else:
            self.config_file = config_file

        if train_file is None:
            self.train_file = "../data/train.spacy"
        else:
            self.train_file = train_file

        if val_file is None:
            self.val_file = "../data/val.spacy"
        else:
            self.val_file = val_file

        if test_file is None:
            self.test_file = "../data/test.spacy"
        else:
            self.test_file = test_file

        self.train_x, self.val_x, self.train_y, self.val_y, self.test_df = self.data_split()
        self.train_labels = list(set(self.train_y))

    def data_split(self):
        data = self.df

        preprocess = PreProcess()
        data['desc_title'] = data['title'].astype(str) + " " + data['description'].astype(str)
        data['desc_title'] = preprocess.spacy_remove_stop_words(
            data=preprocess.common_cleaning_functions(data=data['desc_title']))

        train_df = data.loc[~data['level'].isna()]
        test_df = data.loc[data['level'].isna()]

        train_x, val_x, train_y, val_y = train_test_split(train_df['desc_title'].tolist(),
                                                          train_df['level'].tolist(),
                                                          test_size=0.2, random_state=3137)

        return train_x, val_x, train_y, val_y, test_df

    def initialize_docbins(self, generate_test_file=False):
        train_docs = self.generate_docs(x_values=self.train_x, y_values=self.train_y, model_name='en_core_web_trf')
        val_docs = self.generate_docs(x_values=self.val_x, y_values=self.val_y, model_name='en_core_web_trf')
        train_doc_bin = DocBin(docs=train_docs)
        val_doc_bin = DocBin(docs=val_docs)

        train_doc_bin.to_disk(self.train_file)
        val_doc_bin.to_disk(self.val_file)

        if generate_test_file:
            self.generate_test_file()

    def generate_docs(self, x_values=None, y_values=None, disable_components=None, model_name=None):
        docs = []

        if model_name is None:
            model_name = self.model_name
        if disable_components is None:
            if model_name == 'en_core_web_trf':
                disable_components = ['tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
            else:
                disable_components = ['tagger', 'parser', 'senter', 'attribute_ruler', 'lemmatizer', 'ner']

        pipeline = spacy.load(model_name, disable=disable_components)

        if type(x_values) == pd.Series:
            x_values = x_values.tolist()

        if y_values is not None and type(y_values) == pd.Series:
            y_values = y_values.tolist()

        if y_values is not None:
            labels = self.train_labels
            data = list(zip(x_values, y_values))
            for doc, label in pipeline.pipe(texts=data, as_tuples=True):
                for doc_label in labels:
                    if label == doc_label:
                        doc.cats[doc_label] = 1
                    else:
                        doc.cats[doc_label] = 0
                docs.append(doc)
        else:
            for doc in pipeline.pipe(texts=x_values):
                docs.append(doc)

        del pipeline
        return docs

    def generate_test_file(self, data=None):
        if data is None:
            test_x = self.test_df['desc_title']
        else:
            test_x = data

        test_docs = self.generate_docs(x_values=test_x, model_name='en_core_web_trf')
        test_doc_bin = DocBin(docs=test_docs)
        test_doc_bin.to_disk(self.test_file)

    def train_cli(self):
        train(config_path=self.config_file, output_path=self.model_output_path,
              overrides={"paths.train": self.train_file,
                         "paths.dev": self.val_file}, use_gpu=0)

    def eval_cli(self, on_test=False):
        model_path = self.model_output_path + "model-best/"

        if on_test is False:
            data_path = self.val_file
        else:
            data_path = self.test_file
        evaluate(model=model_path, data_path=data_path, use_gpu=0, silent=False)

    def get_test_preds(self):
        model_path = self.model_output_path + "model-best/"
        pipeline = load_model(name=model_path)

        test_pred_doc_bin = DocBin().from_disk(path=self.pred_output_file)
        test_preds = []
        for doc in test_pred_doc_bin.get_docs(pipeline.vocab):
            test_preds.append(max(doc.cats, key=doc.cats.get))

        del pipeline, test_pred_doc_bin
        return test_preds

    def generate_examples(self, docs=None):
        if docs is None:
            docs = self.generate_docs(x_values=self.train_x, y_values=self.train_y)

        examples = []
        for doc in docs:
            cats_dict = {'cats': doc.cats}
            examples.append(Example.from_dict(doc, cats_dict))

        return examples

    def create_classifier(self, iters=None):
        pipeline = spacy.load(name=self.model_name)
        train_examples = self.generate_examples()

        if iters is None:
            iters = 1000

        text_cat = pipeline.add_pipe('textcat_multilabel')
        text_cat.initialize(self.generate_examples, labels=self.train_labels)
        optimizer = text_cat.create_optimizer()

        for iter in tqdm(range(iters), total=iters):
            text_cat.update(train_examples, sgd=optimizer, drop=0.5)
        text_cat.use_params(optimizer.averages)

        return text_cat

    def predict(self, classifier=None, data=None):
        if classifier is None:
            classifier = self.create_classifier()

        if data is None:
            test_docs = self.generate_docs(x_values=self.val_x, y_values=self.val_y)
        else:
            test_docs = self.generate_docs(x_values=data)

        scores = classifier.predict(test_docs)
        classifier.set_annotations(test_docs, scores)

        y_pred = []
        for doc in test_docs:
            y_pred.append(max(doc.cats, key=doc.cats.get))

        del classifier
        return y_pred

    def performance_metrics(self, from_saved_model=False):
        if from_saved_model:
            y_predicted = self.get_test_preds()
            model_name = "transformer roberta-base"
        else:
            y_predicted = self.predict()
            model_name = self.model_name

        test_labels = list(set(self.val_y))
        report = classification_report(y_true=self.val_y, y_pred=y_predicted, labels=test_labels)

        print(f"Classification report for {model_name}:\n {report}")
