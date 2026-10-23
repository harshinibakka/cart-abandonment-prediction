from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1bYPFDS7A1YLlJ_rFsla-C_dhUZ-mTg3LLJjJAFRFuR4/export?format=csv"
)
TARGET = "should_receive_discount"
ID_COLUMNS = ["user_id"]
NOMINAL_COLUMNS = {
    "gender",
    "traffic_source",
    "product_category",
    "previous_review_sentiment",
}


@st.cache_data(ttl=3600)
def load_shared_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in data.columns:
        raise ValueError(f"The CSV must contain the target column '{TARGET}'.")

    labeled = data.dropna(subset=[TARGET]).copy()
    target_values = set(labeled[TARGET].unique())
    if not target_values.issubset({0, 1, False, True}):
        raise ValueError(f"'{TARGET}' must contain only 0/1 labels.")

    features = labeled.drop(columns=[TARGET, *ID_COLUMNS], errors="ignore")
    features = features.dropna(axis=1, how="all")
    if features.empty:
        raise ValueError("No usable feature columns were found in the CSV.")

    target = labeled[TARGET].astype(int)
    counts = target.value_counts()
    if len(counts) != 2 or counts.min() < 2:
        raise ValueError("The target needs at least two rows for each 0/1 label.")

    return features, target


def build_model(features: pd.DataFrame) -> Pipeline:
    categorical_columns = [
        column
        for column in features.columns
        if column in NOMINAL_COLUMNS or not is_numeric_dtype(features[column])
    ]
    numeric_columns = [
        column for column in features.columns if column not in categorical_columns
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


st.set_page_config(page_title="Cart Session Classifier", page_icon="🛒", layout="wide")
st.title("Cart Session Classifier")
st.caption("Random Forest demonstration using the shared shopping-session data")

st.warning(
    "The spreadsheet target is **`should_receive_discount`**, not a recorded "
    "purchase or cart-abandonment outcome. This app therefore predicts the "
    "existing discount label; it does not validate abandonment prediction or "
    "the causal impact of discounts."
)

source = st.radio(
    "Choose data source",
    ["Shared Google Sheet", "Upload CSV"],
    horizontal=True,
)

try:
    if source == "Shared Google Sheet":
        with st.spinner("Loading the shared spreadsheet..."):
            data = load_shared_data()
    else:
        uploaded_data = st.file_uploader("Upload labeled session data (CSV)", type="csv")
        if uploaded_data is None:
            st.info("Upload a CSV with a `should_receive_discount` column to continue.")
            st.stop()
        data = pd.read_csv(BytesIO(uploaded_data.getvalue()))

    features, target = prepare_data(data)
except (ValueError, pd.errors.ParserError, OSError) as error:
    st.error(f"Could not use this dataset: {error}")
    st.stop()
except Exception as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

st.subheader("Dataset overview")
overview, positive_rate, feature_count = st.columns(3)
overview.metric("Labeled sessions", f"{len(target):,}")
positive_rate.metric("Discount label = 1", f"{target.mean():.1%}")
feature_count.metric("Model features", f"{features.shape[1]}")
st.dataframe(data.head(10), use_container_width=True)

train_features, test_features, train_target, test_target = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42,
    stratify=target,
)
model = build_model(features)

with st.spinner("Training and evaluating the Random Forest..."):
    model.fit(train_features, train_target)
    predictions = model.predict(test_features)
    probabilities = model.predict_proba(test_features)[:, 1]

st.subheader("Held-out evaluation")
st.caption("Metrics use a stratified 80/20 split. Class 1 is the existing discount label.")
metric_columns = st.columns(6)
metric_values = [
    ("Accuracy", accuracy_score(test_target, predictions)),
    ("Balanced accuracy", balanced_accuracy_score(test_target, predictions)),
    ("Precision (label 1)", precision_score(test_target, predictions, zero_division=0)),
    ("Recall (label 1)", recall_score(test_target, predictions, zero_division=0)),
    ("F1 (label 1)", f1_score(test_target, predictions, zero_division=0)),
    ("ROC AUC", roc_auc_score(test_target, probabilities)),
]
for column, (label, value) in zip(metric_columns, metric_values):
    column.metric(label, f"{value:.3f}")

confusion = confusion_matrix(test_target, predictions, labels=[0, 1])
left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    st.dataframe(
        pd.DataFrame(
            confusion,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        ),
        use_container_width=True,
    )

with right:
    st.markdown("**Most influential features**")
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    importance_table = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(12)
        .set_index("Feature")
    )
    st.bar_chart(importance_table)

st.subheader("Score new sessions")
st.caption(
    "Upload one or more unlabeled session rows. Include the same feature columns "
    "used for training; `user_id` and the target column are not required."
)
scoring_file = st.file_uploader(
    "Upload sessions to score (CSV)",
    type="csv",
    key="scoring_csv",
)
if scoring_file is not None:
    try:
        scoring_data = pd.read_csv(BytesIO(scoring_file.getvalue()))
        missing_columns = sorted(set(features.columns) - set(scoring_data.columns))
        if missing_columns:
            st.error(f"Scoring CSV is missing required columns: {', '.join(missing_columns)}")
        else:
            scores = model.predict_proba(scoring_data[features.columns])[:, 1]
            results = scoring_data.copy()
            results["predicted_discount_label"] = model.predict(
                scoring_data[features.columns]
            )
            results["probability_of_discount_label_1"] = scores
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download scored sessions",
                results.to_csv(index=False).encode("utf-8"),
                file_name="scored_sessions.csv",
                mime="text/csv",
            )
    except (ValueError, pd.errors.ParserError, OSError) as error:
        st.error(f"Could not score this CSV: {error}")
    except Exception as error:
        st.error(f"Scoring failed: {error}")
