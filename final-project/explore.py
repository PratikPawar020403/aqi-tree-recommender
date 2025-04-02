import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import os
from PIL import Image
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# --- Caching: Load AQI data ---
@st.cache_data
def load_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        # Fill missing numeric values with column means
        for col in ['Min', 'Max', 'Avg', 'AQI']:
            if col in df.columns:
                df[col].fillna(df[col].mean(), inplace=True)
        # Fill missing categorical values
        if 'Predominant Parameter' in df.columns:
            df['Predominant Parameter'].fillna('Unknown', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- Caching: Load the model bundle ---
@st.cache_resource
def load_model_bundle(bundle_path):
    try:
        with open(bundle_path, 'rb') as file:
            bundle = pickle.load(file)
        return bundle
    except FileNotFoundError:
        st.warning("Model bundle not found. Please run train.py to generate the model bundle.")
    except Exception as e:
        st.error(f"Error loading model bundle: {e}")
    return None

def exploration_interface():
    """
    This function creates the interactive data exploration interface.
    It loads the AQI CSV data and model evaluation metrics (if available),
    provides sidebar filters, and renders multiple interactive visualizations.
    """
    CSV_FILE_PATH = 'AQI.csv'
    MODEL_BUNDLE_PATH = 'model_bundle.pkl'
    
    df = load_data(CSV_FILE_PATH)
    if df is None:
        st.stop()
    else:
        st.success("AQI data loaded successfully for exploration.")
    
    model_bundle = load_model_bundle(MODEL_BUNDLE_PATH)
    if model_bundle and "metrics" in model_bundle:
        model_metrics = model_bundle["metrics"]
        st.success("Model metrics loaded.")
    else:
        model_metrics = None
        st.info("Model evaluation metrics are not available. Please run train.py to generate metrics.")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Data Filters")
    state_options = df['State'].unique() if 'State' in df.columns else []
    pollutant_options = df['Pollutant'].unique() if 'Pollutant' in df.columns else []
    param_options = df['Predominant Parameter'].unique() if 'Predominant Parameter' in df.columns else []
    
    # Default selections are all available options if any exist.
    state_filter = st.sidebar.multiselect("Select States", options=sorted(state_options), 
                                            default=sorted(state_options) if state_options.size > 0 else [])
    pollutant_filter = st.sidebar.multiselect("Select Pollutants", options=sorted(pollutant_options), 
                                                default=sorted(pollutant_options) if pollutant_options.size > 0 else [])
    parameter_filter = st.sidebar.multiselect("Select Predominant Parameters", options=sorted(param_options), 
                                                default=sorted(param_options) if param_options.size > 0 else [])
    
    # --- Error Handling for Empty Filter Selections ---
    if not state_filter:
        st.warning("Please select at least one State from the sidebar filters.")
        st.stop()
    if not pollutant_filter:
        st.warning("Please select at least one Pollutant from the sidebar filters.")
        st.stop()
    if not parameter_filter:
        st.warning("Please select at least one Predominant Parameter from the sidebar filters.")
        st.stop()
    
    # Apply filters to the dataframe
    if 'State' in df.columns:
        df = df[df['State'].isin(state_filter)]
    if 'Pollutant' in df.columns:
        df = df[df['Pollutant'].isin(pollutant_filter)]
    if 'Predominant Parameter' in df.columns:
        df = df[df['Predominant Parameter'].isin(parameter_filter)]
    
    st.title("AQI Data and Model Exploration")
    st.write("Explore historical AQI data with interactive visualizations.")
    
    # --- Model Evaluation Metrics Display ---
    st.header("Model Evaluation Metrics")
    if model_metrics:
        if all(k in model_metrics for k in ['MAE', 'MSE', 'R-Squared']):
            mae = model_metrics['MAE']
            mse = model_metrics['MSE']
            r2 = model_metrics['R-Squared']
        elif all(k in model_metrics for k in ['mae', 'mse', 'r2']):
            mae = model_metrics['mae']
            mse = model_metrics['mse']
            r2 = model_metrics['r2']
        else:
            st.error("Model evaluation metric keys not found. Available keys: " + ", ".join(model_metrics.keys()))
            mae = mse = r2 = None
        if None not in (mae, mse, r2):
            col1, col2, col3 = st.columns(3)
            col1.metric("Mean Absolute Error", f"{mae:.2f}")
            col2.metric("Mean Squared Error", f"{mse:.2f}")
            col3.metric("R-Squared", f"{r2:.2f}")
    else:
        st.info("Model evaluation metrics are not available. Please run train.py to generate metrics.")
    
    # --- Interactive Visualizations ---
    st.header("Interactive Visualizations")
    if not df.empty:
        col_left, col_right = st.columns(2)
        with col_left:
            # 1. Histogram: Distribution of AQI
            fig_hist = px.histogram(df, x="AQI", title="Distribution of AQI Values", 
                                    color_discrete_sequence=['#1f77b4'])
            fig_hist.update_layout(template="plotly_white")
            st.plotly_chart(fig_hist, use_container_width=True)
            st.write("**Insight:** Frequency distribution of AQI values.")
            
            # 2. Bar Chart: Average AQI by State
            if 'State' in df.columns:
                state_avg = df.groupby('State')['AQI'].mean().reset_index()
                fig_bar_state = px.bar(state_avg, x="State", y="AQI", title="Average AQI by State", 
                                       color_discrete_sequence=['#ff7f0e'])
                fig_bar_state.update_layout(template="plotly_white")
                st.plotly_chart(fig_bar_state, use_container_width=True)
                st.write("**Insight:** States with higher average AQI levels.")
            
            # 3. Boxplot: AQI by Predominant Parameter
            if 'Predominant Parameter' in df.columns:
                fig_box_param = px.box(df, x="Predominant Parameter", y="AQI", 
                                       title="AQI by Predominant Parameter", 
                                       color_discrete_sequence=['#2ca02c'])
                fig_box_param.update_layout(template="plotly_white")
                st.plotly_chart(fig_box_param, use_container_width=True)
                st.write("**Insight:** Distribution of AQI across predominant parameters.")
            
            # 4. Histogram: Distribution of Engineered Feature (AQI_Range)
            if 'AQI_Range' in df.columns:
                fig_range = px.histogram(df, x="AQI_Range", title="Distribution of AQI_Range (Max - Min)", 
                                         color_discrete_sequence=['#9467bd'])
                fig_range.update_layout(template="plotly_white")
                st.plotly_chart(fig_range, use_container_width=True)
                st.write("**Insight:** Variation between maximum and minimum AQI readings.")
        with col_right:
            # 5. Scatter Plot: Geographical AQI
            if "Latitude" in df.columns and "Longitude" in df.columns:
                fig_scatter_geo = px.scatter(df, x="Longitude", y="Latitude", color="AQI",
                                             title="AQI by Location", 
                                             color_continuous_scale=px.colors.diverging.RdBu)
                fig_scatter_geo.update_layout(mapbox_style="carto-positron", template="plotly_white")
                st.plotly_chart(fig_scatter_geo, use_container_width=True)
                st.write("**Insight:** Mapping AQI levels on geographical coordinates.")
            
            # 6. Donut Chart: Pollutant Distribution
            if 'Pollutant' in df.columns:
                pollutant_counts = df['Pollutant'].value_counts().reset_index()
                pollutant_counts.columns = ['Pollutant', 'Count']
                fig_donut = px.pie(pollutant_counts, names='Pollutant', values='Count',
                                   title='Pollutant Distribution', 
                                   color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)
                fig_donut.update_traces(hovertemplate='%{label}<br>Count: %{value} (%{percent})')
                st.plotly_chart(fig_donut, use_container_width=True)
                st.write("**Insight:** Share of each pollutant in the dataset.")
            
            # 7. Correlation Heatmap for Selected Variables
            corr_vars = ['AQI', 'Min', 'Max']
            if all(var in df.columns for var in ['Latitude', 'Longitude']):
                corr_vars.extend(['Latitude', 'Longitude'])
            available_corr = [col for col in corr_vars if col in df.columns]
            if available_corr:
                corr_matrix = df[available_corr].corr()
                fig_heatmap = px.imshow(corr_matrix,
                                        labels=dict(x="Variables", y="Variables", color="Correlation"),
                                        x=available_corr, y=available_corr,
                                        color_continuous_scale=px.colors.diverging.RdBu,
                                        title="Correlation Matrix")
                fig_heatmap.update_layout(template="plotly_white")
                st.plotly_chart(fig_heatmap, use_container_width=True)
                st.write("**Insight:** Correlations between key variables.")
    else:
        st.info("No data to display based on current filters.")
    
    # --- Additional Interactive Visualizations ---
    st.header("Additional Visualizations")
    if not df.empty:
        col_left2, col_right2 = st.columns(2)
        with col_left2:
            # Scatter Matrix of key features
            features_for_matrix = ["Min", "Max", "Avg", "AQI"]
            available_features = [feat for feat in features_for_matrix if feat in df.columns]
            if available_features:
                fig_matrix = px.scatter_matrix(df, dimensions=available_features,
                                               title="Scatter Matrix of Pollutant Levels", 
                                               color="AQI", template="plotly_white")
                st.plotly_chart(fig_matrix, use_container_width=True)
                st.write("**Insight:** Pairwise relationships among pollutant metrics and AQI.")
            
            # Violin Plot: AQI Distribution by Pollutant
            if 'Pollutant' in df.columns:
                fig_violin = px.violin(df, x="Pollutant", y="AQI", color="Pollutant", box=True, points=False,
                                       title="AQI Distribution by Pollutant", 
                                       color_discrete_sequence=px.colors.qualitative.Dark2)
                fig_violin.update_layout(xaxis_title="Pollutant", yaxis_title="AQI Value", showlegend=False)
                st.plotly_chart(fig_violin, use_container_width=True)
                st.write("**Insight:** Variability of AQI across pollutants.")
        with col_right2:
            # Stacked Bar Chart: Predominant Pollutants by State
            if 'State' in df.columns and 'Predominant Parameter' in df.columns:
                pred_pollutant = df.groupby(['State', 'Predominant Parameter']).size().reset_index(name='Count')
                fig_stacked = px.bar(pred_pollutant, x="State", y="Count", color="Predominant Parameter",
                                     title="Predominant Pollutants in Different States", barmode='stack',
                                     color_discrete_sequence=px.colors.qualitative.Set2)
                fig_stacked.update_layout(template="plotly_white")
                st.plotly_chart(fig_stacked, use_container_width=True)
                st.write("**Insight:** Distribution of predominant pollutants across states.")
        
        # --- Interactive Geographical Map ---
        st.header("Interactive Geographical Map")
        if "Latitude" in df.columns and "Longitude" in df.columns:
            map_data = df.dropna(subset=["AQI", "Latitude", "Longitude"])
            if not map_data.empty:
                fig_map = px.scatter_mapbox(
                    map_data,
                    lat="Latitude",
                    lon="Longitude",
                    color="AQI",
                    size="AQI",
                    hover_name="City" if "City" in map_data.columns else None,
                    zoom=3,
                    mapbox_style="carto-positron",
                    title="AQI by Location (Geographical Map)",
                    color_continuous_scale=px.colors.diverging.Portland
                )
                fig_map.update_layout(template="plotly_white")
                st.plotly_chart(fig_map, use_container_width=True)
                st.write("**Insight:** Interactive view of AQI distribution across cities.")
            else:
                st.warning("Filtered data does not contain valid geographical information.")
        else:
            st.warning("Geographical data (Latitude/Longitude) not available in the dataset.")
    else:
        st.info("No additional visualizations available based on current filters.")

if __name__ == "__main__":
    exploration_interface()
