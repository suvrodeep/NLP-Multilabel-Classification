import sys
import pandas as pd
from model_factory import BaselineModel, SpacyModel

sys.path.extend(["../", "../config", "../trained_pipelines", "../data", "../outputs"])

df = pd.read_json("../data.json")

baseline_models = ['SupportVectorClassifier', 'NaiveBayesClassifier']

for model in baseline_models:
    baseline_model = BaselineModel(data=df, model_type=model)
    baseline_model.performance_metrics()

model = SpacyModel(data=df, model_name='en_core_web_trf', preprocess_steps=["remove_stop_words"])

model.performance_metrics()

model.initialize_docbins(generate_test_file=True)
model.train_cli()   # Train transformer model
model.eval_cli()

# Apply model on test/validation data
model.performance_metrics(from_saved_model=True)