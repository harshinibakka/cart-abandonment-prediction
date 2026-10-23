"""Train the discount-label classifier and write model + evaluation metrics."""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "customer_segmentation_discount.csv"
TARGET = "should_receive_discount"

df = pd.read_csv(DATA_PATH)
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")
df = df.dropna(subset=[TARGET]).copy()

X = df.drop(columns=[TARGET, "user_id"], errors="ignore")
y = df[TARGET].astype(int)

categorical_cols = [
    "gender", "device_mobile", "traffic_source", "product_category",
    "basket_icon_click", "scrolled_to_reviews", "checked_delivery_pdp",
    "saw_checkout", "sign_in", "is_returning_user", "previous_review_sentiment"
]
categorical_cols = [c for c in categorical_cols if c in X.columns]
for col in categorical_cols:
    X[col] = X[col].astype(str)
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_cols)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight="balanced",
        min_samples_leaf=2, n_jobs=-1
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

joblib.dump(model, BASE_DIR / "discount_model.pkl")
metrics = {
    "rows": int(df.shape[0]),
    "features": int(X.shape[1]),
    "test_size": int(len(y_test)),
    "accuracy": float(accuracy_score(y_test, pred)),
    "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
    "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    "target_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
    "numeric_features": numeric_cols,
    "categorical_features": categorical_cols,
    "note": "Predicts the supplied should_receive_discount label. The original label-generation rule and semantic meaning of 0/1 were not verified; this is not a validated cart-abandonment model."
}
with open(BASE_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print(f"Rows: {len(df)} | Features: {X.shape[1]}")
print(f"Test accuracy: {metrics['accuracy']:.3f}")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, pred))
