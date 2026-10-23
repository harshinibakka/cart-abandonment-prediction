import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Cart Session Discount Predictor",
    page_icon="🛒",
    layout="wide",
)

st.title("🛒 Cart Session Discount Predictor")
st.caption(
    "Machine Learning Case Study | Logistic Regression vs Random Forest"
)

st.markdown(
    """
    **Business goal:** Identify shopping sessions for which a discount is
    recommended, while helping the business investigate unnecessary
    discount costs.

    This application predicts the dataset's existing
    `should_receive_discount` label. It does not directly predict verified
    purchases or cart abandonment.
    """
)

DATA_PATH = "cart_sessions.csv"
TARGET = "should_receive_discount"

# Encoded category columns. Their original category labels were not
# supplied, so the app preserves and displays their numeric codes.
CATEGORICAL_COLUMNS = [
    "gender",
    "device_mobile",
    "traffic_source",
    "product_category",
    "basket_icon_click",
    "scrolled_to_reviews",
    "checked_delivery_pdp",
    "saw_checkout",
    "sign_in",
    "is_returning_user",
    "previous_review_sentiment",
]

# Identifiers are excluded from model training.
ID_COLUMNS = ["user_id"]

# --------------------------------------------------
# LOAD AND VALIDATE DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"The dataset must contain a '{TARGET}' column."
        )

    df = df.dropna(subset=[TARGET]).copy()

    # The target must contain binary labels 0 and 1.
    labels = set(pd.to_numeric(df[TARGET], errors="coerce").dropna().unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(
            f"'{TARGET}' must contain both binary labels 0 and 1."
        )

    df[TARGET] = df[TARGET].astype(int)
    return df


try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load or validate the dataset: {exc}")
    st.info(
        "Check that cart_sessions.csv is committed to the root "
        "of your GitHub repository alongside app.py."
    )
    st.stop()

# --------------------------------------------------
# DATASET OVERVIEW
# --------------------------------------------------
st.header("1. Dataset Overview")

total_rows, total_columns = df.shape
missing_count = int(df.isna().sum().sum())
duplicate_count = int(df.duplicated().sum())
discount_rate = float(df[TARGET].mean() * 100)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Shopping sessions", f"{total_rows:,}")
c2.metric("Dataset columns", f"{total_columns}")
c3.metric("Missing values", f"{missing_count:,}")
c4.metric("Label 1 rate", f"{discount_rate:.1f}%")

with st.expander("View dataset preview"):
    st.dataframe(df.head(20), use_container_width=True)

with st.expander("View data-quality report"):
    quality = pd.DataFrame({
        "Data type": df.dtypes.astype(str),
        "Missing values": df.isna().sum(),
        "Unique values": df.nunique(),
    })
    st.dataframe(quality, use_container_width=True)
    st.write(f"Exact duplicate rows: **{duplicate_count}**")

st.subheader("Target label distribution")
target_counts = (
    df[TARGET]
    .value_counts()
    .reindex([0, 1], fill_value=0)
)
target_distribution = pd.DataFrame({
    "Label": [
        "0 — Discount not recommended",
        "1 — Discount recommended",
    ],
    "Sessions": [
        int(target_counts.loc[0]),
        int(target_counts.loc[1]),
    ],
})
st.bar_chart(
    target_distribution.set_index("Label"),
    y="Sessions",
)
st.caption(
    "These are the dataset's recorded labels, not verified purchase "
    "completion or cart-abandonment outcomes."
)

# --------------------------------------------------
# PREPARE FEATURES
# --------------------------------------------------
st.header("2. Data Preparation")

excluded_columns = ID_COLUMNS + [TARGET]
feature_columns = [
    column for column in df.columns
    if column not in excluded_columns
]

X = df[feature_columns].copy()
y = df[TARGET].copy()

# Convert numeric features safely.
for column in X.columns:
    X[column] = pd.to_numeric(X[column], errors="coerce")

categorical_columns = [
    column for column in CATEGORICAL_COLUMNS
    if column in X.columns
]
numeric_columns = [
    column for column in X.columns
    if column not in categorical_columns
]

st.write(
    f"**Predictive features:** {len(feature_columns)}  |  "
    f"**Categorical features:** {len(categorical_columns)}  |  "
    f"**Numeric features:** {len(numeric_columns)}"
)
st.write("Excluded from training: `user_id` and the target column.")

# Stratified split keeps class proportions approximately similar.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns),
    ])

# --------------------------------------------------
# TRAIN AND EVALUATE MODELS
# --------------------------------------------------
@st.cache_resource
def train_models(dataframe):
    # Build the split inside this function so that cached training
    # is deterministic for this dataset.
    features = [
        col for col in dataframe.columns
        if col not in ID_COLUMNS + [TARGET]
    ]
    x_all = dataframe[features].copy()
    y_all = dataframe[TARGET].astype(int).copy()

    for column in x_all.columns:
        x_all[column] = pd.to_numeric(
            x_all[column], errors="coerce"
        )

    cats = [
        col for col in CATEGORICAL_COLUMNS
        if col in x_all.columns
    ]
    nums = [col for col in x_all.columns if col not in cats]

    x_tr, x_te, y_tr, y_te = train_test_split(
        x_all,
        y_all,
        test_size=0.20,
        random_state=42,
        stratify=y_all,
    )

    def preprocessor():
        return ColumnTransformer([
            (
                "numeric",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                nums,
            ),
            (
                "categorical",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                cats,
            ),
        ])

    model_definitions = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}
    metrics = []

    for model_name, estimator in model_definitions.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor()),
            ("model", estimator),
        ])

        # Evaluate on the held-out test set.
        pipeline.fit(x_tr, y_tr)
        predictions = pipeline.predict(x_te)
        probabilities = pipeline.predict_proba(x_te)[:, 1]

        scores = {
            "Accuracy": accuracy_score(y_te, predictions),
            "Precision": precision_score(
                y_te, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_te, predictions, zero_division=0
            ),
            "F1-score": f1_score(
                y_te, predictions, zero_division=0
            ),
            "ROC-AUC": roc_auc_score(y_te, probabilities),
        }

        results[model_name] = {
            "test_pipeline": pipeline,
            "predictions": predictions,
            "probabilities": probabilities,
            "scores": scores,
            "confusion_matrix": confusion_matrix(
                y_te, predictions, labels=[0, 1]
            ),
            "report": classification_report(
                y_te,
                predictions,
                labels=[0, 1],
                target_names=[
                    "Discount not recommended (0)",
                    "Discount recommended (1)",
                ],
                output_dict=True,
                zero_division=0,
            ),
        }

        metrics.append({"Model": model_name, **scores})

    # After evaluation, fit fresh copies on all available labelled
    # rows for the interactive prediction form.
    deployment_models = {}
    for model_name, estimator in model_definitions.items():
        final_pipeline = Pipeline([
            ("preprocessor", preprocessor()),
            ("model", estimator),
        ])
        final_pipeline.fit(x_all, y_all)
        deployment_models[model_name] = final_pipeline

    return (
        pd.DataFrame(metrics).set_index("Model"),
        results,
        deployment_models,
        x_all,
        y_all,
        x_te,
        y_te,
    )

metrics, results, deployment_models, X_all, y_all, X_test_final, y_test_final = (
    train_models(df)
)

# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------
st.header("3. Model Training and Comparison")
st.write(
    "Both models use the same stratified 80/20 train-test split "
    "and random seed. Reported scores are from the held-out test set."
)

st.dataframe(
    metrics.style.format("{:.3f}"),
    use_container_width=True,
)

st.subheader("Compare evaluation metrics")
st.bar_chart(metrics.T)

# Use F1-score as the default comparison criterion; business
# costs may require a different decision threshold or metric.
best_model_name = metrics["F1-score"].idxmax()
st.success(
    f"Best model by test F1-score: **{best_model_name}** "
    f"({metrics.loc[best_model_name, 'F1-score']:.3f})"
)
st.caption(
    "The highest F1-score is a useful initial comparison, not proof "
    "of the lowest business cost. Discount costs and false-negative "
    "costs should guide the final business decision."
)

# --------------------------------------------------
# CONFUSION MATRIX AND CLASSIFICATION REPORT
# --------------------------------------------------
st.header("4. Detailed Evaluation")

selected_evaluation_model = st.selectbox(
    "Select a model to inspect",
    list(results.keys()),
)

evaluation = results[selected_evaluation_model]
cm = evaluation["confusion_matrix"]

fig, ax = plt.subplots(figsize=(5, 4))
image = ax.imshow(cm)
ax.set_title(f"Confusion Matrix — {selected_evaluation_model}")
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_xticks([0, 1], labels=["0", "1"])
ax.set_yticks([0, 1], labels=["0", "1"])

for row in range(2):
    for col in range(2):
        ax.text(
            col,
            row,
            str(cm[row, col]),
            ha="center",
            va="center",
        )

fig.colorbar(image, ax=ax)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
    - **True negative:** label 0 correctly predicted as 0.
    - **False positive:** label 0 incorrectly predicted as 1.
    - **False negative:** label 1 incorrectly predicted as 0.
    - **True positive:** label 1 correctly predicted as 1.
    """
)

report_df = pd.DataFrame(evaluation["report"]).T
st.subheader("Classification report")
st.dataframe(
    report_df.style.format("{:.3f}"),
    use_container_width=True,
)

# --------------------------------------------------
# RANDOM FOREST FEATURE IMPORTANCE
# --------------------------------------------------
st.header("5. Feature Importance")

rf_pipeline = results["Random Forest"]["test_pipeline"]
rf_model = rf_pipeline.named_steps["model"]
rf_preprocessor = rf_pipeline.named_steps["preprocessor"]

try:
    transformed_names = rf_preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame({
        "Feature": transformed_names,
        "Importance": rf_model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    st.bar_chart(
        importance_df.head(15).set_index("Feature"),
        y="Importance",
    )
    with st.expander("View all feature importances"):
        st.dataframe(importance_df, use_container_width=True)
except Exception as exc:
    st.warning(f"Feature importance could not be displayed: {exc}")

st.caption(
    "Feature importance shows how the fitted Random Forest used features "
    "to make predictions. It does not establish causation."
)

# --------------------------------------------------
# MODEL EQUATIONS AND R-SQUARED
# --------------------------------------------------
st.header("6. Model Equations")

with st.expander("Logistic Regression equation"):
    st.latex(
        r"z=b_0+\sum_{j=1}^{p}b_jx_j"
    )
    st.latex(
        r"P(y=1\mid x)=\frac{1}{1+e^{-z}}"
    )
    st.write(
        "Logistic Regression estimates the probability of class 1. "
        "The predicted class depends on the probability threshold."
    )

with st.expander("Random Forest model"):
    st.latex(
        r"\hat{y}=\operatorname{mode}"
        r"\{T_1(x),T_2(x),\ldots,T_n(x)\}"
    )
    st.write(
        "Random Forest combines predictions from multiple decision trees. "
        "It has no single linear regression equation."
    )

with st.expander("R-squared equation"):
    st.latex(
        r"R^2=1-\frac{\sum_i(y_i-\hat{y}_i)^2}"
        r"{\sum_i(y_i-\bar{y})^2}"
    )
    st.write(
        "R² is primarily a regression metric. For this binary "
        "classification task, use accuracy, precision, recall, "
        "F1-score and ROC-AUC instead."
    )

# --------------------------------------------------
# INTERACTIVE PREDICTION
# --------------------------------------------------
st.header("7. Try a Shopping Session")

st.write(
    "Enter shopping-session details below. The app estimates the "
    "dataset's discount-recommendation label; it does not guarantee "
    "what a real customer will do."
)

with st.form("prediction_form"):
    input_values = {}

    col_left, col_right = st.columns(2)

    for index, feature in enumerate(feature_columns):
        column = col_left if index % 2 == 0 else col_right

        with column:
            series = X_all[feature].dropna()

            if feature in categorical_columns:
                choices = sorted(series.unique().tolist())
                if not choices:
                    choices = [0]
                input_values[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    choices,
                    help=(
                        "This feature is numerically encoded in the "
                        "dataset; original category meanings were not provided."
                    ),
                )
            else:
                if series.empty:
                    minimum, maximum, default = 0.0, 100.0, 0.0
                else:
                    minimum = float(series.min())
                    maximum = float(series.max())
                    default = float(series.median())

                if minimum == maximum:
                    input_values[feature] = minimum
                else:
                    input_values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=minimum,
                        max_value=maximum,
                        value=default,
                    )

    chosen_model = st.selectbox(
        "Prediction model",
        list(deployment_models.keys()),
    )
    submitted = st.form_submit_button(
        "Predict discount recommendation",
        type="primary",
    )

if submitted:
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    pipeline = deployment_models[chosen_model]

    predicted_label = int(pipeline.predict(input_df)[0])
    predicted_probability = float(
        pipeline.predict_proba(input_df)[0][1]
    )

    st.subheader("Prediction result")

    if predicted_label == 1:
        st.success("Prediction: Discount recommended (class 1)")
    else:
        st.info("Prediction: Discount not recommended (class 0)")

    st.metric(
        "Model-estimated probability of class 1",
        f"{predicted_probability:.1%}",
    )
    st.progress(predicted_probability)

    st.caption(
        "This probability reflects the model's estimate for the dataset "
        "label. It is not a proven probability of purchase or abandonment, "
        "and it is not evidence that a discount will cause conversion."
    )

# --------------------------------------------------
# BUSINESS RECOMMENDATIONS
# --------------------------------------------------
st.header("8. Business Insights and Limitations")

st.markdown(
    """
    **Suggested business actions**

    1. Compare both models on the held-out test set before choosing one.
    2. Review false positives because they may represent unnecessary
       discount recommendations.
    3. Review false negatives because potential discount opportunities
       may be missed.
    4. Validate the model with an experiment before claiming that
       discounts increase conversion or reduce costs.

    **Dataset limitations**

    - The target is `should_receive_discount`, not an observed
      `purchase_completed` or `cart_abandoned` outcome.
    - Numeric category codes are retained as categories, but their
      original meanings were not supplied.
    - A model's prediction is not proof that a discount will change
      customer behaviour.
    - Test metrics may vary with different data and split choices.
    """
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------
st.header("9. Download Results")

metrics_csv = metrics.reset_index().to_csv(index=False).encode("utf-8")
st.download_button(
    "Download model comparison (CSV)",
    data=metrics_csv,
    file_name="model_comparison_results.csv",
    mime="text/csv",
)

st.success("Dashboard loaded. Explore the sections above to review the results.")

