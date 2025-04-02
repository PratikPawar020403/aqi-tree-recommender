import streamlit as st 
import numpy as np
import pickle
import json
import pandas as pd
import re  # For robust range parsing
from PIL import Image

# ---------------------------
# 1. PAGE CONFIGURATION
# ---------------------------
if __name__ == '__main__':
    st.set_page_config(
        page_title="AQI Prediction Tool 📈",
        page_icon="📈",
        layout="wide"
    )

# ---------------------------
# 2. CUSTOM THEME & STYLES
# ---------------------------
st.markdown(
    """
    <style>
        body {
            color: #333;
            background-color: #f8f9fa;
            font-family: 'Roboto', sans-serif;
        }
        .stApp {
            width: 100%;
            padding: 2rem;
        }
        .st-header {
            font-size: 2.5rem;
            color: #007BFF;
            text-align: center;
            margin-bottom: 1rem;
        }
        .st-subheader {
            font-size: 1.5rem;
            color: #007BFF;
            margin-top: 1rem;
        }
        .input-label {
            font-weight: bold;
            color: #555;
            margin-bottom: 0.2rem;
        }
        .stButton > button {
            background: linear-gradient(135deg, #007BFF, #0056b3);
            color: white;
            font-weight: bold;
            border: none;
            border-radius: 0.3rem;
            padding: 0.7rem 1.5rem;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton > button:hover {
            background: linear-gradient(135deg, #0056b3, #003f7f);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# 3. RESULT BOX STYLING
# ---------------------------
result_styles = {
    "green":  {"border": "#2e7d32", "text": "#2e7d32"},
    "orange": {"border": "#ff9800", "text": "#ff9800"},
    "red":    {"border": "#f44336", "text": "#f44336"}
}

# ---------------------------
# 4. LOAD MODEL BUNDLE
# ---------------------------
@st.cache_resource
def load_model_bundle():
    """Loads the model bundle (pipeline, label encoder, scaler)."""
    try:
        with open("final-project/model_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        return bundle
    except Exception as e:
        st.error(f"Error loading model bundle: {e}")
        return None

bundle = load_model_bundle()
if bundle is None:
    st.stop()

pipeline = bundle['pipeline']
le_pred_param = bundle['label_encoder_pred_param']
if 'scaler' in bundle:
    scaler = bundle['scaler']
else:
    st.error("Scaler not found in model bundle.")
    st.stop()

# ---------------------------
# 5. LOAD METRICS (OPTIONAL)
# ---------------------------
@st.cache_data
def load_metrics():
    """Loads model evaluation metrics from a JSON file."""
    try:
        with open("final-project/metrics.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics: {e}")
        return None

metrics = load_metrics()
if metrics:
    st.sidebar.markdown("### Model Evaluation Metrics")
    st.sidebar.write(f"**MAE:** {metrics['MAE']:.4f}")
    st.sidebar.write(f"**MSE:** {metrics['MSE']:.4f}")
    st.sidebar.write(f"**R-Squared:** {metrics['R-Squared']:.4f}")

# ---------------------------
# 6. LOAD TREE DATASET & NORMALIZE (UPDATED)
# ---------------------------
@st.cache_data
def load_tree_data():
    """
    Loads the tree dataset and normalizes the pollutant names.
    Now handles subscript characters in pollutant names.
    """
    try:
        df = pd.read_csv("treeaqi - Sheet1.csv")
        # Normalize pollutant names: replace subscripts and clean
        df["Predominant Pollutant"] = (
            df["Predominant Pollutant"]
            .str.replace("₁", "1")  # Subscript 1 → "1"
            .str.replace("₂", "2")  # Subscript 2 → "2"
            .str.replace("₃", "3")  # Subscript 3 → "3"
            .str.replace("₅", "5")  # Subscript 5 → "5"
            .str.replace("₀", "0")  # Subscript 0 → "0"
            .str.strip()
            .str.upper()
            .str.replace(" ", "")
        )
        # Clean AQI range column
        df["AQI Value Range"] = df["AQI Value Range"].astype(str).str.strip()
        return df
    except Exception as e:
        st.error(f"Error loading tree data: {e}")
        return None

# ---------------------------
# 7. HELPER: ROBUST RANGE PARSER
# ---------------------------
def parse_aqi_range(range_str):
    """
    Attempts to parse a string range (e.g., "100-150", "100 to 150", "100 – 150")
    and returns (lower_bound, upper_bound) as floats.
    Returns (None, None) if parsing fails.
    """
    # Replace common dash variants with a standard dash.
    range_clean = re.sub(r"[–—]", "-", range_str)
    # Try splitting by dash
    if "-" in range_clean:
        parts = range_clean.split("-")
    elif "to" in range_clean.lower():
        parts = range_clean.lower().split("to")
    else:
        parts = []
    if len(parts) >= 2:
        try:
            lower_bound = float(parts[0].strip())
            upper_bound = float(parts[1].strip())
            return lower_bound, upper_bound
        except Exception:
            return None, None
    return None, None

# ---------------------------
# 8. PREDICTION INTERFACE
# ---------------------------
def prediction_interface():
    """
    Displays input fields, makes the AQI prediction, and recommends a tree.
    """
    st.title("AQI Prediction Tool 📈")
    st.markdown("Enter environmental parameters to get an AQI prediction and tree recommendation.")

    # --- Input Form ---
    with st.form("aqi_prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<p class='input-label'>Geographical Coordinates</p>", unsafe_allow_html=True)
            latitude = st.number_input("Enter Latitude", value=0.0, format="%.6f", placeholder="e.g., 28.6139")
            longitude = st.number_input("Enter Longitude", value=0.0, format="%.6f", placeholder="e.g., 77.2090")
            st.markdown("<p class='input-label'>Pollutant Concentrations</p>", unsafe_allow_html=True)
            min_val = st.number_input("Minimum Concentration", value=0.0, placeholder="e.g., 10.5")
            avg_val = st.number_input("Average Concentration", value=0.0, placeholder="e.g., 20.0")
        with col2:
            max_val = st.number_input("Maximum Concentration", value=0.0, placeholder="e.g., 35.2")
            st.markdown("<p class='input-label'>Predominant Parameter</p>", unsafe_allow_html=True)
            predominant_input = st.text_input("Enter Predominant Parameter", "PM2.5", placeholder="e.g., PM2.5, OZONE, SO2")
            # Normalize user input
            predominant_clean = predominant_input.strip().upper().replace(" ", "")
        submitted = st.form_submit_button("Predict Air Quality Index")

    # --- Prediction Logic ---
    if submitted:
        with st.spinner("Predicting AQI..."):
            try:
                # Validate and encode predominant parameter
                encoded_classes = [x.upper().replace(" ", "") for x in le_pred_param.classes_]
                if predominant_clean not in encoded_classes:
                    st.error(f"Unrecognized predominant parameter: {predominant_input}. Please use one of: {', '.join(le_pred_param.classes_)}")
                    return
                index = encoded_classes.index(predominant_clean)
                true_class = le_pred_param.classes_[index]
                predominant_encoded = float(le_pred_param.transform([true_class])[0])

                # Feature engineering
                pollutant_avg = (min_val + max_val + avg_val) / 3.0
                aqi_range = max_val - min_val
                aqi_std = np.std([min_val, max_val, avg_val])
                aqi_range_avg_int = aqi_range * avg_val
                aqi_std_avg_int = aqi_std * avg_val
                aqi_range_squared = aqi_range ** 2
                aqi_std_cubed = aqi_std ** 3

                features = np.array([
                    latitude,
                    longitude,
                    min_val,
                    max_val,
                    avg_val,
                    predominant_encoded,
                    pollutant_avg,
                    aqi_range,
                    aqi_std,
                    aqi_range_avg_int,
                    aqi_std_avg_int,
                    aqi_range_squared,
                    aqi_std_cubed
                ]).reshape(1, -1)

                features_scaled = scaler.transform(features)
                scaled_prediction = pipeline.predict(features_scaled)
                aqi_index = 5  # Adjust if needed
                original_predicted_aqi = scaled_prediction[0] * scaler.scale_[aqi_index] + scaler.mean_[aqi_index]

                # Determine styling
                if original_predicted_aqi <= 50:
                    style_key = "green"
                    interpretation_text = "Air quality is satisfactory."
                elif original_predicted_aqi <= 100:
                    style_key = "orange"
                    interpretation_text = "Air quality is moderately polluted."
                else:
                    style_key = "red"
                    interpretation_text = "Air quality is unhealthy."
                style_settings = result_styles[style_key]

                # Tree Recommendation Logic
                tree_df = load_tree_data()
                recommended_tree = "No recommendation available"
                apti_value = "N/A"
                if tree_df is not None:
                    for _, row in tree_df.iterrows():
                        ds_pollutant = str(row["Predominant Pollutant"]).strip()
                        if ds_pollutant == predominant_clean:
                            range_str = str(row["AQI Value Range"]).strip()
                            lower_bound, upper_bound = parse_aqi_range(range_str)
                            if lower_bound and upper_bound and (lower_bound <= original_predicted_aqi <= upper_bound):
                                recommended_tree = row["Recommended Tree (English / Indian Name)"]
                                apti_value = row["APTI Value (avg)"]
                                break

                # Display Results
                st.markdown(
                    f"""
                    <div style="
                        padding: 1.5rem; 
                        background-color: #d9f2ff; 
                        border: 3px solid {style_settings['border']}; 
                        border-radius: 0.5rem; 
                        margin-top: 1.5rem; 
                        text-align: center; 
                        font-size: 1.3rem; 
                        font-weight: bold;
                        color: {style_settings['text']};
                    ">
                        <p style="margin: 0; font-size: 1.5rem;">
                            Predicted AQI: <strong>{original_predicted_aqi:.2f}</strong>
                        </p>
                        <p style="margin: 0; font-size: 1.2rem;">
                            {interpretation_text}
                        </p>
                        <hr style="margin: 1rem 0;">
                        <p style="margin: 0; font-size: 1.3rem;">
                            Recommended Tree: <strong>{recommended_tree}</strong>
                        </p>
                        <p style="margin: 0; font-size: 1.2rem;">
                            APTI: <strong>{apti_value}</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    st.markdown("---")
    st.info("Tip: Ensure that pollutant concentrations are in the same units as used during training. Common pollutants include PM2.5, PM10, CO, SO2, NO2, and OZONE.")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
if __name__ == '__main__':
    prediction_interface()
