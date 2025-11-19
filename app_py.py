import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- CONFIGURATION AND MODEL LOADING ---
st.set_page_config(
    page_title="Professional Medical Cost Predictor",
    page_icon="💸",
    layout="wide", # Use 'wide' layout for a more desktop-friendly, professional feel
    initial_sidebar_state="expanded"
)

# Hardcoded exchange rate for prediction demonstration (approximate rate as of late 2025)
USD_TO_INR_RATE = 83.50 

# Use Streamlit's caching mechanism (@st.cache_resource) to load the model only once.
@st.cache_resource
def load_pipeline():
    """Loads the trained ML pipeline (preprocessor + model)."""
    # CRITICAL FIX: The model file name must match the saved file name
    model_filename = 'medical_cost_stack.pkl'
    try:
        stack = joblib.load(model_filename)
        return stack
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. "
                 "Please ensure the file is in the root of your GitHub repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load the pipeline into the variable 'stack'
stack = load_pipeline()

# --- PROFESSIONAL UI COMPONENTS ---

# Header Section
st.markdown("<h1 style='text-align: center; color: #1f77b4;'>💰 Annual Insurance Cost Estimator </h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #4b4b4b;'>A data-driven tool to predict healthcare charges based on individual profiles.</h4>", unsafe_allow_html=True)
st.markdown("---")

# Main Input Container
with st.container(border=True):
    st.subheader("👤 Patient Profile Inputs")
    st.caption("Adjust the sliders and dropdowns to define the user profile for prediction.")

    # Layout inputs in 3 columns for better organization
    col_age_sex_smoker, col_bmi_children, col_region = st.columns(3)

    with col_age_sex_smoker:
        # Input 1: Age (Numeric Slider)
        age = st.slider("Age (Years)", 18, 65, 30, key='age', help="Age of the primary beneficiary.")
        
        # Input 2: Sex (Categorical Selectbox)
        sex = st.selectbox("Sex", ['male', 'female'], key='sex')
        
        # Input 3: Smoker Status (Categorical Selectbox)
        smoker = st.selectbox("Smoker Status", ['no', 'yes'], key='smoker', help="Smoking status is a major cost factor.")

    with col_bmi_children:
        # Input 4: BMI (Numeric Input)
        bmi = st.number_input("BMI (Body Mass Index)", 15.0, 55.0, 25.0, step=0.1, key='bmi', help="BMI range is typically 18.5 to 30.")
        
        # Input 5: Children (Numeric Slider)
        children = st.slider("Number of Children", 0, 5, 0, key='children', help="Number of children covered by the plan.")
        
        # Spacer for visual alignment
        st.markdown("<br>", unsafe_allow_html=True)

    with col_region:
        # Input 6: Region (Categorical Selectbox)
        region = st.selectbox("Geographical Region", ['southeast', 'southwest', 'northeast', 'northwest'], key='region')
        
        # Additional visual space
        st.markdown("<br>"*3, unsafe_allow_html=True)


# --- PREDICTION LOGIC ---
st.markdown("---")
if st.button("🚀 Calculate Estimated Charges", key='predict_button', type="primary", use_container_width=True):
    # 1. Gather all user inputs into a dictionary
    input_data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

    # 2. Convert dictionary to a DataFrame
    features_df = pd.DataFrame([input_data])

    # 3. Make the prediction
    with st.spinner('Calculating estimate using the machine learning model...'):
        try:
            # Prediction in USD
            prediction_usd = stack.predict(features_df)[0]
            
            # Prediction in INR
            prediction_inr = prediction_usd * USD_TO_INR_RATE
            
            # --- 4. Display the results in professional metrics ---
            st.success("✅ Prediction Complete!", icon="📈")
            
            # Use columns for USD and INR metrics
            col_usd, col_inr = st.columns(2)
            
            with col_usd:
                st.metric(
                    label="🇺🇸 Estimated Annual Charge (USD)",
                    value=f"${prediction_usd:,.2f}",
                    delta="Model Estimate"
                )
            
            with col_inr:
                st.metric(
                    label="🇮🇳 Estimated Annual Charge (INR)",
                    value=f"₹{prediction_inr:,.2f}",
                    delta=f"Based on {USD_TO_INR_RATE} INR/USD"
                )
            
            # Display conversion formula
            st.markdown(
                """
                <div style='background-color: #e0f7fa; padding: 15px; border-radius: 10px; margin-top: 20px; border-left: 5px solid #00bcd4;'>
                    <p style='font-weight: bold; color: #00838f;'>Conversion Formula Used:</p>
                    <p>
                        Estimated Cost (INR) = Estimated Cost (USD) $\\times$ {rate}
                    </p>
                </div>
                """.format(rate=USD_TO_INR_RATE),
                unsafe_allow_html=True
            )
            
            # st.balloons() # Optional: keep balloons for success feedback
            
        except Exception as e:
            st.error(f"An error occurred during prediction. Please check model and inputs. Error details: {e}")

# --- PROFESSIONAL SIDEBAR/FOOTER ---
st.sidebar.header("Model and Deployment Info")
st.sidebar.markdown(
    f"""
    **Current Exchange Rate:** 1 USD ≈ {USD_TO_INR_RATE} INR (Hardcoded)

    This app uses a Machine Learning Pipeline (`stack`) trained on historical insurance data. 
    The pipeline includes a **StandardScaler** and **OneHotEncoder** to ensure data integrity during prediction.
    """
)
st.sidebar.markdown("---")
st.sidebar.caption("App Version 2.0 | Deployed via Streamlit Community Cloud")
