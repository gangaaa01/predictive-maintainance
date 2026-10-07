import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Electric Motor Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    padding: 28px;
    border-radius: 15px;
    background: linear-gradient(135deg, #172554, #1e40af);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    margin-top: 6px;
}

.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 15px;
    margin-bottom: 15px;
}

.result-normal {
    padding: 25px;
    border-radius: 15px;
    background-color: #ecfdf5;
    border: 2px solid #10b981;
    text-align: center;
}

.result-risk {
    padding: 25px;
    border-radius: 15px;
    background-color: #fef2f2;
    border: 2px solid #ef4444;
    text-align: center;
}

.result-title {
    font-size: 28px;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = Path("predictive_maintenance_model.pkl")

if not MODEL_PATH.exists():

    st.error(
        "Model file not found. Please run train_model.py first."
    )

    st.stop()


model = joblib.load(MODEL_PATH)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Electric Motor PM")

st.sidebar.caption(
    "Electric Motor Predictive Maintenance System"
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Motor Health",
        "📊 Model Dashboard",
        "🔧 How It Works",
        "ℹ️ About Project"
    ]
)


st.sidebar.divider()


st.sidebar.markdown(
    """
    **Final ML Model**

    🏆 XGBoost

    **Dataset**

    AI4I 2020 Predictive Maintenance

    **Records**

    10,000
    """
)


# ============================================================
# MOTOR HEALTH PAGE
# ============================================================

if page == "🏠 Motor Health":

    st.markdown("""
    <div class="hero">

    <h1>⚙️ Electric Motor Predictive Maintenance System</h1>

    <p>
    Machine learning based failure prediction for
    industrial electric motors.
    </p>

    <p>
    Enter the current motor operating parameters to
    assess the possibility of machine failure.
    </p>

    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">🔍 Motor Operating Parameters</div>',
        unsafe_allow_html=True
    )


    st.write(
        "Enter the current operating conditions of the electric motor."
    )


    # ========================================================
    # INPUT SECTION
    # ========================================================

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # TEMPERATURE
    # --------------------------------------------------------

    with col1:

        st.markdown("### 🌡️ Temperature")

        machine_type = st.selectbox(
            "Motor Type",
            ["L", "M", "H"],
            help="Machine/product type."
        )


        air_temperature = st.number_input(
            "Air Temperature [K]",
            min_value=250.0,
            max_value=350.0,
            value=298.0,
            step=0.1
        )


        process_temperature = st.number_input(
            "Process Temperature [K]",
            min_value=250.0,
            max_value=400.0,
            value=308.0,
            step=0.1
        )


    # --------------------------------------------------------
    # MOTOR OPERATION
    # --------------------------------------------------------

    with col2:

        st.markdown("### ⚡ Motor Operation")


        rotational_speed = st.number_input(
            "Rotational Speed [rpm]",
            min_value=0,
            max_value=5000,
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


    # --------------------------------------------------------
    # TOOL CONDITION
    # --------------------------------------------------------

    with col3:

        st.markdown("### 🔧 Tool Condition")


        tool_wear = st.number_input(
            "Tool Wear [min]",
            min_value=0,
            max_value=300,
            value=100,
            step=1
        )


        st.info(
            """
            **Important parameters**

            Temperature, rotational speed,
            torque and tool wear provide information
            about the machine's operating condition.
            """
        )


    st.divider()


    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "🔮 ANALYZE MOTOR HEALTH",
        use_container_width=True,
        type="primary"
    )


    if predict_button:

        # ----------------------------------------------------
        # CREATE INPUT DATA
        # ----------------------------------------------------

        input_data = pd.DataFrame({

            "Type": [
                machine_type
            ],

            "Air temperature [K]": [
                air_temperature
            ],

            "Process temperature [K]": [
                process_temperature
            ],

            "Rotational speed [rpm]": [
                rotational_speed
            ],

            "Torque [Nm]": [
                torque
            ],

            "Tool wear [min]": [
                tool_wear
            ]
        })


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )[0]


        probability = model.predict_proba(
            input_data
        )[0][1]


        st.divider()


        st.markdown(
            '<div class="section-title">📋 Motor Health Assessment</div>',
            unsafe_allow_html=True
        )


        # ====================================================
        # NORMAL CONDITION
        # ====================================================

        if prediction == 0:

            st.markdown(
                f"""
                <div class="result-normal">

                <div class="result-title">
                🟢 MOTOR OPERATING NORMALLY
                </div>

                <p>
                The model does not detect a significant
                machine failure risk for these conditions.
                </p>

                <h2>
                Failure Probability: {probability * 100:.2f}%
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.success(
                "Recommended Action: Continue normal operation "
                "and monitor the motor parameters."
            )


        # ====================================================
        # FAILURE CONDITION
        # ====================================================

        else:

            st.markdown(
                f"""
                <div class="result-risk">

                <div class="result-title">
                🔴 MOTOR FAILURE RISK DETECTED
                </div>

                <p>
                The model predicts an increased possibility
                of machine failure.
                </p>

                <h2>
                Failure Probability: {probability * 100:.2f}%
                </h2>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.error(
                "Recommended Action: Inspect the motor and "
                "schedule preventive maintenance."
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.subheader(
            "Motor Parameters Used"
        )


        display_data = pd.DataFrame({

            "Parameter": [
                "Motor Type",
                "Air Temperature",
                "Process Temperature",
                "Rotational Speed",
                "Torque",
                "Tool Wear"
            ],

            "Value": [
                machine_type,
                f"{air_temperature:.1f} K",
                f"{process_temperature:.1f} K",
                f"{rotational_speed} rpm",
                f"{torque:.1f} Nm",
                f"{tool_wear} min"
            ]
        })


        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# MODEL DASHBOARD
# ============================================================

elif page == "📊 Model Dashboard":

    st.title(
        "📊 Machine Learning Performance"
    )


    st.write(
        "Three machine learning algorithms were trained and "
        "evaluated using the same dataset."
    )


    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    results = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost ⭐"
        ],

        "Accuracy (%)": [
            82.45,
            97.95,
            98.75
        ],

        "Precision (%)": [
            14.18,
            71.43,
            93.88
        ],

        "Recall (%)": [
            82.35,
            66.18,
            67.65
        ],

        "F1 Score (%)": [
            24.19,
            68.70,
            78.63
        ],

        "ROC-AUC (%)": [
            90.70,
            97.21,
            98.00
        ]
    })


    st.subheader(
        "🏆 Model Comparison"
    )


    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )


    st.success(
        "XGBoost was selected as the final model because "
        "it achieved the best overall performance."
    )


    # ========================================================
    # KEY METRICS
    # ========================================================

    st.subheader(
        "XGBoost Final Performance"
    )


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "Accuracy",
            "98.75%"
        )


    with c2:

        st.metric(
            "Precision",
            "93.88%"
        )


    with c3:

        st.metric(
            "Recall",
            "67.65%"
        )


    with c4:

        st.metric(
            "ROC-AUC",
            "98.00%"
        )


    st.divider()


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.subheader(
        "🔎 XGBoost Feature Importance"
    )


    feature_importance = pd.DataFrame({

        "Feature": [
            "Torque",
            "Rotational Speed",
            "Tool Wear",
            "Air Temperature",
            "Process Temperature",
            "Motor Type L",
            "Motor Type M",
            "Motor Type H"
        ],

        "Importance": [
            0.277945,
            0.174229,
            0.167627,
            0.115185,
            0.100892,
            0.071809,
            0.048037,
            0.044276
        ]
    })


    fig, ax = plt.subplots(
        figsize=(9, 5)
    )


    ax.barh(
        feature_importance["Feature"][::-1],
        feature_importance["Importance"][::-1]
    )


    ax.set_xlabel(
        "Importance"
    )


    ax.set_title(
        "Features Used by XGBoost for Failure Prediction"
    )


    plt.tight_layout()


    st.pyplot(fig)


    plt.close(fig)


    st.caption(
        "Feature importance shows which input features contributed "
        "more strongly to the model's predictions. It does not by "
        "itself prove that a feature causes failure."
    )


    # ========================================================
    # CONFUSION MATRICES
    # ========================================================

    st.subheader(
        "🎯 Confusion Matrix Results"
    )


    c1, c2 = st.columns(2)


    with c1:

        st.markdown(
            "### Random Forest"
        )


        rf_matrix = pd.DataFrame(

            [
                [1914, 18],
                [23, 45]
            ],

            columns=[
                "Predicted No Failure",
                "Predicted Failure"
            ],

            index=[
                "Actual No Failure",
                "Actual Failure"
            ]
        )


        st.dataframe(
            rf_matrix,
            use_container_width=True
        )


    with c2:

        st.markdown(
            "### XGBoost ⭐"
        )


        xgb_matrix = pd.DataFrame(

            [
                [1929, 3],
                [22, 46]
            ],

            columns=[
                "Predicted No Failure",
                "Predicted Failure"
            ],

            index=[
                "Actual No Failure",
                "Actual Failure"
            ]
        )


        st.dataframe(
            xgb_matrix,
            use_container_width=True
        )


    st.info(
        "XGBoost correctly identified 46 of the 68 failure cases "
        "and produced only 3 false positive predictions."
    )


    # ========================================================
    # ROC-AUC COMPARISON
    # ========================================================

    st.subheader(
        "📈 ROC-AUC Comparison"
    )


    roc_data = pd.DataFrame({

        "Model": [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],

        "ROC-AUC": [
            90.70,
            97.21,
            98.00
        ]
    })


    fig, ax = plt.subplots(
        figsize=(8, 5)
    )


    ax.bar(
        roc_data["Model"],
        roc_data["ROC-AUC"]
    )


    ax.set_ylabel(
        "ROC-AUC (%)"
    )


    ax.set_ylim(
        0,
        100
    )


    ax.set_title(
        "Model ROC-AUC Comparison"
    )


    plt.xticks(
        rotation=10
    )


    plt.tight_layout()


    st.pyplot(fig)


    plt.close(fig)


# ============================================================
# HOW IT WORKS
# ============================================================

elif page == "🔧 How It Works":

    st.title(
        "🔧 How the System Works"
    )


    st.write(
        "The system uses machine learning to convert electric "
        "motor operating parameters into a failure prediction."
    )


    st.subheader(
        "System Workflow"
    )


    st.markdown("""
    ### 1️⃣ Motor Parameters

    The system receives:

    **Temperature → Rotational Speed → Torque → Tool Wear**

    ↓

    ### 2️⃣ Data Preprocessing

    The categorical motor type is converted into numerical
    features using one-hot encoding.

    ↓

    ### 3️⃣ Machine Learning

    Three models were compared:

    **Logistic Regression → Random Forest → XGBoost**

    ↓

    ### 4️⃣ Model Selection

    XGBoost achieved the best overall performance and was
    selected as the final model.

    ↓

    ### 5️⃣ Failure Prediction

    The trained XGBoost model predicts:

    **No Failure (0)** or **Failure (1)**

    and provides a failure probability.

    ↓

    ### 6️⃣ Maintenance Recommendation

    The application converts the prediction into a practical
    maintenance recommendation.
    """)


    st.divider()


    st.subheader(
        "🧠 Why Predictive Maintenance?"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.markdown(
            "### ⏱️ Early Detection"
        )

        st.write(
            "Identify potential failure before the motor "
            "experiences a major breakdown."
        )


    with col2:

        st.markdown(
            "### 💰 Reduced Downtime"
        )

        st.write(
            "Planned maintenance can help reduce unexpected "
            "machine downtime."
        )


    with col3:

        st.markdown(
            "### 🔧 Better Maintenance"
        )

        st.write(
            "Maintenance can be planned based on predicted "
            "machine condition."
        )


# ============================================================
# ABOUT PROJECT
# ============================================================

elif page == "ℹ️ About Project":

    st.title(
        "ℹ️ About the Project"
    )


    st.markdown("""
    ## Predictive Maintenance of Electric Motors

    This project uses machine learning to predict potential
    electric motor failure from operating parameters.

    The objective is to support **early detection and
    preventive maintenance** of industrial electric motors.
    """)


    st.subheader(
        "📂 Dataset"
    )


    st.write(
        "AI4I 2020 Predictive Maintenance Dataset"
    )


    st.write(
        "10,000 records • 14 columns • 3.39% failure rate"
    )


    st.subheader(
        "🤖 Machine Learning Models"
    )


    st.write("""
    • Logistic Regression

    • Random Forest

    • XGBoost
    """)


    st.subheader(
        "🏆 Final Model"
    )


    st.success(
        "XGBoost — Accuracy: 98.75% | "
        "Precision: 93.88% | "
        "ROC-AUC: 98.00%"
    )


    st.subheader(
        "💻 Technologies Used"
    )


    st.write("""
    Python • Pandas • Scikit-learn • XGBoost •
    Matplotlib • Streamlit • Joblib
    """)


    st.subheader(
        "🚀 Future Scope"
    )


    st.write("""
    • Real-time motor sensor integration

    • IoT-based monitoring

    • Automated maintenance alerts

    • Cloud-based monitoring

    • Continuous model improvement
    """)


    st.divider()


    st.caption(
        "Electric Motor Predictive Maintenance System | "
        "Machine Learning"
    )