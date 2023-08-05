import pandas as pd
import spacy
from preprocess import PreProcess


class EvaluateSimilarity:
    def __init__(self, predicted=None, reference=None):
        if predicted is None:
            self.predicted = pd.read_json("../outputs/test_pred.json")
        else:
            self.predicted = predicted

        if reference is None:
            self.reference = pd.read_json("../data.json")
        else:
            self.reference = reference

    def preprocess_data(self):
        pred = self.predicted
        ref = self.reference

        preprocess = PreProcess()
        for data in [pred, ref]:
            data['desc_title'] = data['title'].astype(str) + " " + data['description'].astype(str)
            data['desc_title'] = preprocess.spacy_remove_stop_words(
                data=preprocess.common_cleaning_functions(data=data['desc_title']))

        return pred, ref

    def calculate_similarity_score(self):
        pred, ref = self.preprocess_data()
        pipeline = spacy.load('en_core_web_lg')
        labels = list(set(pred['level'].tolist()))

        pred_docs = []
        ref_docs = []

        for text in pred['desc_title'].tolist():
            pred_docs.append(pipeline(text))
        for text in ref['desc_title'].tolist():
            ref_docs.append(pipeline(text))

        pred['docs'] = pred_docs
        ref['docs'] = ref_docs

        sim_dict = {}
        for label in labels:
            pred_subset = pred.loc[pred['level'] == label, 'docs'].tolist()
            ref_subset = ref.loc[ref['level'] == label, 'docs'].tolist()

            similarity = 0
            for pred_doc in pred_subset:
                sim_per_pred_doc = 0
                for ref_doc in ref_subset:
                    sim_per_pred_doc += pred_doc.similarity(ref_doc)

                similarity += sim_per_pred_doc / len(ref_subset)

            sim_dict[label] = round(similarity / len(pred_subset) * 100, 2)

        return pd.DataFrame.from_dict(data=sim_dict, orient='index')


