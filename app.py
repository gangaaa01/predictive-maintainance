import streamlit as st
import pandas as pd
import joblib


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

model = joblib.load("predictive_maintenance_model.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Predictive Maintenance")

st.sidebar.markdown("### Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🔮 Prediction",
        "📊 Model Dashboard",
        "ℹ️ About"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Machine learning based predictive maintenance "
    "system for early failure detection."
)


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "🔮 Prediction":

    st.title("⚙️ Predictive Maintenance System")

    st.write(
        "Predict machine failure using Machine Learning."
    )

    st.divider()

    st.subheader("Enter Machine Parameters")

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # LEFT COLUMN
    # --------------------------------------------------------

    with col1:

        machine_type = st.selectbox(
            "Machine Type",
            ["L", "M", "H"]
        )

        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=290.0,
            max_value=310.0,
            value=300.0,
            step=0.1
        )

        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=300.0,
            max_value=320.0,
            value=310.0,
            step=0.1
        )

    # --------------------------------------------------------
    # RIGHT COLUMN
    # --------------------------------------------------------

    with col2:

        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=1000,
            max_value=3000,
            value=1500,
            step=10
        )

        torque = st.number_input(
            "Torque [Nm]",
            min_value=0.0,
            max_value=100.0,
            value=40.0,
            step=0.1
        )

        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )

    st.write("")

    predict_button = st.button(
        "🔍 Predict Machine Condition",
        type="primary"
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if predict_button:

        input_data = pd.DataFrame({
            "Type": [machine_type],
            "Air temperature [K]": [air_temperature],
            "Process temperature [K]": [process_temperature],
            "Rotational speed [rpm]": [rotational_speed],
            "Torque [Nm]": [torque],
            "Tool wear [min]": [tool_wear]
        })

        try:

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(input_data)[0][1]

            st.divider()

            st.subheader("Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ MACHINE FAILURE PREDICTED"
                )

                st.write(
                    f"Failure probability: "
                    f"**{probability * 100:.2f}%**"
                )

                st.warning(
                    "Recommended action: Inspect the machine "
                    "and schedule maintenance."
                )

            else:

                st.success(
                    "✅ MACHINE OPERATING NORMALLY"
                )

                st.write(
                    f"Failure probability: "
                    f"**{probability * 100:.2f}%**"
                )

                st.info(
                    "Recommended action: Continue monitoring "
                    "the machine."
                )

        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.code(str(e))


# ============================================================
# PAGE 2 — MODEL DASHBOARD
# ============================================================

elif page == "📊 Model Dashboard":

    st.title("📊 Model Dashboard")

    st.write(
        "Performance and analysis of the Predictive "
        "Maintenance Machine Learning model."
    )

    st.divider()

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Model Accuracy",
            "98%"
        )

    with col2:

        st.metric(
            "Test Samples",
            "2,000"
        )

    with col3:

        st.metric(
            "Algorithm",
            "Random Forest"
        )

    st.divider()

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("🔍 Feature Importance")

    st.image(
        "feature_importance.png",
        caption="Features contributing to machine failure prediction",
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("📌 Confusion Matrix")

    st.image(
        "random_forest_confusion_matrix.png",
        caption="Random Forest confusion matrix",
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader("📈 ROC Curve Comparison")

    st.image(
        "roc_curve_comparison.png",
        caption="ROC curve comparison of the models",
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------------
    # FAILURE DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("📊 Machine Failure Distribution")

    st.image(
        "machine_failure_distribution.png",
        caption="Distribution of machine failure and non-failure cases",
        width="stretch"
    )

    st.divider()

    # --------------------------------------------------------
    # OTHER ANALYSIS
    # --------------------------------------------------------

    st.subheader("🔧 Tool Wear vs Failure")

    st.image(
        "tool_wear_vs_failure.png",
        caption="Relationship between tool wear and machine failure",
        width="stretch"
    )

    st.divider()

    st.subheader("⚙️ Torque vs Failure")

    st.image(
        "torque_vs_failure.png",
        caption="Relationship between torque and machine failure",
        width="stretch"
    )


# ============================================================
# PAGE 3 — ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.write(
        """
        This project uses Machine Learning to predict whether
        an industrial machine is likely to experience a failure
        based on its operating parameters.
        """
    )

    st.write(
        "The system analyzes parameters such as:"
    )

    st.markdown(
        """
        - Air Temperature
        - Process Temperature
        - Rotational Speed
        - Torque
        - Tool Wear
        - Machine Type
        """
    )

    st.write(
        """
        A Random Forest classification model is used to predict
        the machine condition and provide a maintenance recommendation.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # PROJECT OBJECTIVE
    # --------------------------------------------------------

    st.subheader("🎯 Project Objective")

    st.write(
        "To identify potential machine failures early and "
        "support condition-based maintenance using machine learning."
    )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGIES
    # --------------------------------------------------------

    st.subheader("🛠️ Technologies Used")

    st.write(
        "Python • Pandas • NumPy • Scikit-learn • "
        "Matplotlib • Seaborn • Streamlit"
    )

    st.divider()

    # --------------------------------------------------------
    # MACHINE LEARNING MODEL
    # --------------------------------------------------------

    st.subheader("🤖 Machine Learning Model")

    st.write(
        "A Random Forest classification model is used to "
        "predict the machine condition and provide a "
        "maintenance recommendation."
    )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM WORKFLOW
    # --------------------------------------------------------

    st.subheader("🔄 System Workflow")

    st.write(
        "Machine Parameters → Data Processing → "
        "Machine Learning Model → Failure Prediction → "
        "Maintenance Recommendation"
    )

    st.divider()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.subheader("📁 Dataset")

    st.write(
        "The project uses the AI4I 2020 Predictive "
        "Maintenance Dataset."
    )

    st.write(
        "The dataset contains machine operating parameters "
        "and machine failure information used to train the model."
    )

    st.divider()

    # --------------------------------------------------------
    # FINAL NOTE
    # --------------------------------------------------------

    st.caption(
        "Predictive Maintenance System | Machine Learning"
    )