# scripts/retrain.py
from database.generate_dataset import generate
from scripts.load_data import load_dataset
from scripts.train import train_models
from scripts.evaluate import evaluate_model, log_model_history

meta = generate(dataset_version=None, seed=123)  # generate creates file and returns meta
load_dataset(append=True, dataset_version=meta["dataset_version"])
train_models()
mae, rmse, r2, smape = evaluate_model()
log_model_history(mae, rmse, r2, smape)
