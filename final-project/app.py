import streamlit as st
from PIL import Image

# --- Page Configuration (Only in the main app file) ---
st.set_page_config(
    page_title="AQI Insights Hub",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Custom CSS for a Modern, Aesthetic Look ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

    html, body {
        font-family: 'Montserrat', sans-serif;
        color: #333;
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    }
    .stApp {
        padding: 2rem;
    }
    /* Header Styles */
    .header-title {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #00796b;
    }
    .header-subtitle {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #004d40;
    }
    /* Section Title */
    .section-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00796b;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #00796b;
        padding-bottom: 0.5rem;
    }
    /* App Description */
    .app-description {
        font-size: 1rem;
        line-height: 1.6;
        text-align: center;
        color: #555;
    }
    /* Feature Block */
    .feature-block {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s ease-in-out;
    }
    .feature-block:hover {
        transform: translateY(-5px);
    }
    .feature-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #00695c;
        margin-bottom: 0.5rem;
    }
    .feature-description {
        font-size: 1rem;
        color: #555;
    }
    /* Sidebar custom info */
    .sidebar .sidebar-content {
        font-size: 1rem;
        color: #004d40;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Home", "Predict", "Explore"), index=0)

# --- Home Page ---
if page == "Home":
    st.markdown("<div class='header-title'>Welcome to AQI Insights Hub 🌳</div>", unsafe_allow_html=True)
    st.markdown("<div class='header-subtitle'>Your Central Platform for AQI Predictions and Data Exploration</div>", unsafe_allow_html=True)
    
    # Load and display the home image with caching
    @st.cache_resource
    def load_app_image(image_path):
        try:
            return Image.open(image_path)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return None

    # Use a relative path to your image file
    app_image = load_app_image("final-project/aqi.jpg")
    if app_image:
        st.image(app_image, caption="Visualize Air Quality, Understand Our Environment", width=600)
    
    st.markdown("<div class='section-title'>Key Features</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class='feature-block'>
                <div class='feature-title'>Real-Time AQI Prediction</div>
                <p class='feature-description'>
                    Utilize our state-of-the-art prediction model to receive instant Air Quality Index forecasts.
                    Simply input environmental parameters and get accurate predictions to plan your day.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            """
            <div class='feature-block'>
                <div class='feature-title'>Interactive Data Exploration</div>
                <p class='feature-description'>
                    Dive into historical AQI data through interactive charts, graphs, and maps.
                    Use dynamic filters to explore specific states, pollutants, and parameters for comprehensive insights.
                </p>
            </div>
            """, unsafe_allow_html=True
        )
    
    st.markdown("<div class='section-title'>Getting Started</div>", unsafe_allow_html=True)
    st.markdown(
        """
        - **Navigate to 'Predict':** Use the sidebar to access our AQI prediction tool and input the required parameters.
        - **Navigate to 'Explore':** Select the 'Explore' page to visualize historical AQI data with interactive charts and maps.
        """, unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.write("Developed with Streamlit, Plotly, and Scikit-learn for an engaging and insightful experience.")

# --- Predict Page ---
elif page == "Predict":
    st.title("Real-Time AQI Prediction")
    st.markdown("<div class='header-subtitle'>Input parameters and receive an instant AQI forecast</div>", unsafe_allow_html=True)
    try:
        # Call the prediction interface from predict.py
        from predict import prediction_interface
        prediction_interface()
    except Exception as e:
        st.error(f"Error in prediction module: {e}")

# --- Explore Page ---
elif page == "Explore":
    st.title("Interactive AQI Data Exploration")
    st.markdown("<div class='header-subtitle'>Visualize historical AQI data with interactive charts and maps</div>", unsafe_allow_html=True)
    try:
        # Call the exploration interface from explore.py
        from explore import exploration_interface
        exploration_interface()
    except Exception as e:
        st.error(f"Error in exploration module: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Developed with Streamlit & Plotly")
