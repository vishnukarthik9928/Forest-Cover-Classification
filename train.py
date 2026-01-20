from data_preparation import (
    load_data,
    clean_data,
    feature_engineering,
    prepare_features
)
from modeling import train_and_evaluate

print("Loading data...")
df = load_data("cover_type.csv")

print("Cleaning data...")
df = clean_data(df)

print("Feature engineering...")
df = feature_engineering(df)

print("Preparing features & handling imbalance...")
X, y = prepare_features(df)

print("Training model...")
model = train_and_evaluate(X, y)

print("\n✅ TRAINING COMPLETE")
print("Saved files:")
print("- best_model.pkl")
print("- preprocessor.pkl")
