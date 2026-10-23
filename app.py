import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# =========================================================
# CONFIGURATION
# =========================================================
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "discount_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(
    page_title="Cart Session Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PROFESSIONAL NAVY + BLUE THEME
# =========================================================
st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    [data-testid="stSidebar"] {
        background-color: #111a2e;
        border-right: 1px solid #263854;
    }

    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.9);
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #172554, #14213d, #102b3f);
        border: 1px solid #304768;
        margin-bottom: 22px;
    }

    .hero h1 {
        color: #f8fafc;
        font-size: 30px;
        margin: 0 0 8px 0;
    }

    .hero p {
        color: #b8c7dc;
        font-size: 15px;
        margin: 0;
        line-height: 1.6;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 20px;
        font-weight: 650;
        margin-top: 8px;
        margin-bottom: 4px;
    }

    .muted {
        color: #9caec7;
        font-size: 13px;
    }

    .result-card {
        background: #111c31;
        border: 1px solid #30415f;
        border-radius: 14px;
        padding: 20px;
        min-height: 120px;
    }

    .result-label {
        color: #9fb0c8;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .result-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 700;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: #2563eb;
        color: #ffffff;
        border: 1px solid #3b82f6;
        border-radius: 9px;
        min-height: 44px;
        font-weight: 600;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #1d4ed8;
        color: #ffffff;
        border-color: #60a5fa;
    }

    button[data-baseweb="tab"] {
        color: #b8c7dc;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #93c5fd;
    }

    [data-testid="stMetric"] {
        background: #111c31;
        border: 1px solid #263854;
        border-radius: 12px;
        padding: 14px;
    }

    [data-testid="stProgressBar"] > div > div {
        background-color: #3b82f6;
    }

    hr {
        border-color: #263854;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL AND METRICS
# =========================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}


if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please add discount_model.pkl "
        "to the repository."
    )
    st.stop()

try:
    model = load_model()
    metrics = load_metrics()
except Exception as exc:
    st.error("Unable to load the saved model or metrics.")
    st.caption(f"Technical details: {exc}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🛍️ Cart Intelligence")
    st.caption("MACHINE LEARNING CASE STUDY")
    st.divider()

    st.markdown("### Project overview")
    st.write(
        "A binary classification application that analyses "
        "shopping-session features and predicts the label "
        "provided by the original dataset."
    )

    st.markdown("### Model information")
    st.write("**Algorithm:** Random Forest")
    st.write("**Task:** Binary classification")
    st.code("should_receive_discount", language=None)

    if metrics.get("accuracy") is not None:
        st.divider()
        st.metric(
            "Test accuracy",
            f"{float(metrics['accuracy']):.1%}"
        )
        st.caption("Reported held-out test performance.")

    st.divider()
    st.caption(
        "Educational prototype · Validate before business use"
    )


# =========================================================
# PAGE HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>Cart Session Intelligence</h1>
    <p>
        Understand shopping-session patterns through machine learning.
        Enter customer and session details to generate a classification
        prediction using your existing trained model.
    </p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Interpretation note: this application predicts the existing "
    "'should_receive_discount' dataset label. The meanings of labels "
    "0 and 1 have not been independently verified. The model does not "
    "directly predict verified cart abandonment or prove that a "
    "discount will increase purchases."
)


# =========================================================
# INPUT FORM
# One shared form and one shared prediction button
# =========================================================
st.markdown(
    '<div class="section-title">Customer session</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted">Complete the three sections below, then run '
    'the prediction once.</p>',
    unsafe_allow_html=True,
)

with st.form("customer_session_form"):

    tab1, tab2, tab3 = st.tabs([
        "🛒 Shopping activity",
        "👤 Customer profile",
        "📊 Engagement history",
    ])

    # -----------------------------------------------------
    # TAB 1: SHOPPING ACTIVITY
    # -----------------------------------------------------
    with tab1:
        st.markdown("#### Current shopping session")

        c1, c2, c3 = st.columns(3)

        with c1:
            current_session_duration_sec = st.number_input(
                "Session duration (seconds)",
                min_value=0.0,
                value=180.0,
                step=10.0,
            )

            pages_viewed = st.number_input(
                "Pages viewed",
                min_value=0,
                max_value=1000,
                value=4,
            )

            traffic_source = st.selectbox(
                "Traffic source code",
                [1, 2, 3, 4],
            )

        with c2:
            cart_items_count_this_session = st.number_input(
                "Cart items",
                min_value=0,
                max_value=100,
                value=1,
            )

            cart_total_value = st.number_input(
                "Cart total value",
                min_value=0.0,
                value=50.0,
                step=5.0,
            )

            product_category = st.selectbox(
                "Product category code",
                [0, 1, 2, 3],
            )

        with c3:
            basket_icon_click = st.selectbox(
                "Clicked basket?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            saw_checkout = st.selectbox(
                "Viewed checkout?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            device_mobile = st.selectbox(
                "Using mobile device?",
                [1, 0],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        s1, s2, s3 = st.columns(3)

        with s1:
            scrolled_to_reviews = st.selectbox(
                "Viewed product reviews?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with s2:
            checked_delivery_pdp = st.selectbox(
                "Checked delivery information?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

        with s3:
            time_spent_deviation = st.number_input(
                "Time-spent deviation",
                value=0.0,
                step=1.0,
            )

    # -----------------------------------------------------
    # TAB 2: CUSTOMER PROFILE
    # -----------------------------------------------------
    with tab2:
        st.markdown("#### Customer information")

        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
            )

            gender = st.selectbox(
                "Gender code",
                [0, 1],
            )

            annual_income = st.number_input(
                "Annual income",
                min_value=0.0,
                value=50000.0,
                step=1000.0,
            )

        with c2:
            sign_in = st.selectbox(
                "Signed in?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            is_returning_user = st.selectbox(
                "Returning customer?",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
            )

            lifetime_purchases_count = st.number_input(
                "Lifetime purchases",
                min_value=0,
                max_value=10000,
                value=2,
            )

        with c3:
            customer_conversion_rate = st.number_input(
                "Historical conversion rate",
                min_value=0.0,
                max_value=1.0,
                value=0.10,
                step=0.01,
                format="%.3f",
            )

            days_since_last_discount = st.number_input(
                "Days since last discount",
                min_value=0,
                max_value=5000,
                value=999,
            )

    # -----------------------------------------------------
    # TAB 3: ENGAGEMENT HISTORY
    # -----------------------------------------------------
    with tab3:
        st.markdown("#### Previous customer engagement")

        c1, c2 = st.columns(2)

        with c1:
            previous_review_sentiment = st.selectbox(
                "Previous review sentiment code",
                [0, 1, 2, 3],
            )

        with c2:
            social_channels_followed_count = st.number_input(
                "Social channels followed",
                min_value=0,
                max_value=1000,
                value=2,
            )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Run prediction",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# PREDICTION RESULT
# =========================================================
if submitted:

    row = {
        "age": age,
        "gender": str(gender),
        "annual_income": annual_income,
        "device_mobile": str(device_mobile),
        "traffic_source": str(traffic_source),
        "product_category": str(product_category),
        "current_session_duration_sec":
            current_session_duration_sec,
        "time_spent_deviation": time_spent_deviation,
        "pages_viewed": pages_viewed,
        "basket_icon_click": str(basket_icon_click),
        "cart_items_count_this_session":
            cart_items_count_this_session,
        "cart_total_value": cart_total_value,
        "scrolled_to_reviews": str(scrolled_to_reviews),
        "checked_delivery_pdp": str(checked_delivery_pdp),
        "saw_checkout": str(saw_checkout),
        "sign_in": str(sign_in),
        "is_returning_user": str(is_returning_user),
        "lifetime_purchases_count": lifetime_purchases_count,
        "customer_conversion_rate": customer_conversion_rate,
        "days_since_last_discount": days_since_last_discount,
        "previous_review_sentiment": str(
            previous_review_sentiment
        ),
        "social_channels_followed_count":
            social_channels_followed_count,
    }

    input_df = pd.DataFrame([row])

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

        if hasattr(model, "classes_"):
            classes = list(model.classes_)
        elif hasattr(model, "named_steps"):
            final_estimator = list(model.named_steps.values())[-1]
            classes = list(final_estimator.classes_)
        else:
            raise AttributeError(
                "Unable to determine the saved model's class labels."
            )

        probability_by_class = {
            int(label): float(probability)
            for label, probability in zip(classes, probabilities)
        }

        predicted_probability = probability_by_class[prediction]

        st.divider()

        st.markdown(
            '<div class="section-title">Prediction result</div>',
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Predicted dataset label</div>
                    <div class="result-value">{prediction}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Probability of predicted label</div>
                    <div class="result-value">{predicted_probability:.1%}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with r3:
            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">Classification model</div>
                    <div class="result-value">Random Forest</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("#### Probability by label")

        for class_label in sorted(probability_by_class):
            probability = probability_by_class[class_label]
            st.progress(
                probability,
                text=f"Label {class_label}: {probability:.1%}",
            )

        st.caption(
            "Probabilities are model estimates, not guarantees. "
            "The dataset label should not be interpreted as a confirmed "
            "discount recommendation until its definition is verified."
        )

        output_df = input_df.copy()
        output_df["predicted_dataset_label"] = prediction
        output_df["predicted_label_probability"] = predicted_probability

        st.download_button(
            "⬇️ Download prediction record (CSV)",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="cart_session_prediction.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as exc:
        st.error(
            "Prediction failed. Verify that the saved model uses the "
            "same feature names and preprocessing as this application."
        )
        st.caption(f"Technical details: {exc}")


# =========================================================
# MODEL EVALUATION
# =========================================================
with st.expander("📈 Model evaluation details"):

    if not metrics:
        st.write(
            "No metrics.json file was found. Predictions may still work."
        )

    else:
        m1, m2 = st.columns(2)

        with m1:
            if metrics.get("accuracy") is not None:
                st.metric(
                    "Test accuracy",
                    f"{float(metrics['accuracy']):.1%}",
                )

        with m2:
            if metrics.get("test_size") is not None:
                st.metric(
                    "Test records",
                    int(metrics["test_size"]),
                )

        if metrics.get("confusion_matrix"):
            st.markdown("#### Confusion matrix")

            st.dataframe(
                pd.DataFrame(
                    metrics["confusion_matrix"],
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                ),
                use_container_width=True,
            )

        report = metrics.get("classification_report", {})

        report_rows = [
            {"Class": label, **report[label]}
            for label in ["0", "1", "macro avg", "weighted avg"]
            if label in report
        ]

        if report_rows:
            st.markdown("#### Classification report")

            st.dataframe(
                pd.DataFrame(report_rows).set_index("Class"),
                use_container_width=True,
            )

        st.caption(
            "Evaluation measures prediction of the existing dataset "
            "labels. It does not establish causal impact, increased "
            "sales, or reduced discount spending."
        )


# =========================================================
# FOOTER
# =========================================================
st.divider()

st.caption(
    "Cart Session Intelligence · Machine Learning Case Study · "
    "Experimental predictions; validate before business use."
)
