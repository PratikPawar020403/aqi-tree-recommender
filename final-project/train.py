import os
import logging
import json
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# ---------------------------
# Setup Logging and Directories
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CHARTS_DIR = "charts"
if not os.path.exists(CHARTS_DIR):
    os.makedirs(CHARTS_DIR)
    logging.info(f"Created directory for charts: {CHARTS_DIR}")

# ---------------------------
# Helper Functions for Saving Artifacts
# ---------------------------
def save_model_bundle(pipeline, le_pollutant, le_pred_param, scaler, metrics, bundle_path="model_bundle.pkl"):
    """
    Saves the model bundle containing the pipeline, label encoders, scaler, and evaluation metrics.
    """
    bundle = {
        "pipeline": pipeline,
        "label_encoder_pollutant": le_pollutant,
        "label_encoder_pred_param": le_pred_param,
        "scaler": scaler,  # Added scaler to bundle
        "metrics": metrics
    }
    try:
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        logging.info(f"Model bundle saved successfully as '{bundle_path}'.")
    except Exception as e:
        logging.error(f"Error saving the model bundle: {e}")

def save_metrics(metrics, metrics_path="metrics.json"):
    """
    Saves evaluation metrics to a JSON file.
    """
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        logging.info(f"Evaluation metrics saved to '{metrics_path}'.")
    except Exception as e:
        logging.error(f"Error saving metrics: {e}")

# ---------------------------
# Data Loading and Preprocessing
# ---------------------------
def load_and_preprocess_data(csv_path):
    """
    Loads the AQI dataset, replaces "NA" with NaN, converts specified columns to numeric,
    imputes missing values for 'Min', 'Max', and 'Avg' with column means,
    and drops rows with missing 'AQI'.
    """
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"CSV file '{csv_path}' loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading CSV file '{csv_path}': {e}")
        raise

    # Replace "NA" strings with NaN
    df.replace("NA", np.nan, inplace=True)
    logging.info("Replaced 'NA' with NaN.")

    # Convert relevant columns to numeric
    numeric_cols = ['Latitude', 'Longitude', 'Min', 'Max', 'Avg', 'AQI']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    logging.info("Converted specified columns to numeric types.")

    # Impute missing values for numeric columns and drop rows with missing AQI
    for col in ['Min', 'Max', 'Avg']:
        df[col].fillna(df[col].mean(), inplace=True)
        logging.info(f"Filled missing values in '{col}' with column mean.")
    df = df.dropna(subset=['AQI'])
    logging.info("Dropped rows with missing AQI values.")

    return df

# ---------------------------
# Feature Engineering
# ---------------------------
def feature_engineering(df):
    """
    Encodes 'Pollutant' and 'Predominant Parameter' using LabelEncoder,
    creates a basic feature 'Pollutant_Avg', and additional engineered features.
    """
    le_pollutant = LabelEncoder()
    le_pred_param = LabelEncoder()
    df['Pollutant'] = le_pollutant.fit_transform(df['Pollutant'].astype(str))
    df['Predominant Parameter'] = le_pred_param.fit_transform(df['Predominant Parameter'].astype(str))
    logging.info("Encoded 'Pollutant' and 'Predominant Parameter'.")

    # Basic feature: average of Min, Max, and Avg
    df['Pollutant_Avg'] = df[['Min', 'Max', 'Avg']].mean(axis=1)
    logging.info("Created feature 'Pollutant_Avg'.")

    # Additional engineered features
    df['AQI_Range'] = df['Max'] - df['Min']
    df['AQI_StdDev'] = df[['Min', 'Max', 'Avg']].std(axis=1)
    df['AQI_Range_Avg_Interaction'] = df['AQI_Range'] * df['Avg']
    df['AQI_StdDev_Avg_Interaction'] = df['AQI_StdDev'] * df['Avg']
    df['AQI_Range_Squared'] = df['AQI_Range'] ** 2
    df['AQI_StdDev_Cubed'] = df['AQI_StdDev'] ** 3
    logging.info("Created additional engineered features.")

    return df, le_pollutant, le_pred_param

# ---------------------------
# Outlier Handling
# ---------------------------
def handle_outliers_iqr(y):
    """
    Returns a boolean mask for values in y that are not considered outliers based on the IQR method.
    """
    Q1 = np.percentile(y, 25)
    Q3 = np.percentile(y, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    logging.info(f"Outlier detection (IQR): lower_bound={lower_bound:.2f}, upper_bound={upper_bound:.2f}")
    return (y >= lower_bound) & (y <= upper_bound)

# ---------------------------
# Feature Scaling
# ---------------------------
def scale_features(df, numerical_cols):
    """
    Scales the numerical features in df using StandardScaler.
    """
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    logging.info("Scaled numerical features using StandardScaler.")
    return df, scaler

# ---------------------------
# Plotly Visualization Functions
# ---------------------------
def plot_feature_importance_plotly(importances, feature_names):
    """
    Creates an interactive Plotly bar chart for feature importance.
    """
    indices = np.argsort(importances)[::-1]
    sorted_importances = importances[indices]
    sorted_features = [feature_names[i] for i in indices]
    fig = go.Figure(data=[go.Bar(
        x=sorted_features,
        y=sorted_importances,
        marker_color=px.colors.qualitative.Set2
    )])
    fig.update_layout(title="Feature Importance (Optimized RF)",
                      xaxis_title="Features",
                      yaxis_title="Importance",
                      template="plotly_white")
    fig.show()

def plot_learning_curve_plotly(estimator, X_train, y_train):
    """
    Creates an interactive Plotly learning curve.
    """
    train_sizes, train_scores, val_scores = learning_curve(estimator, X_train, y_train,
                                                           cv=5, scoring='r2',
                                                           train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_scores_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='crimson')
    ))
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_scores_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='darkgreen')
    ))
    # Shaded areas for standard deviation
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([train_scores_mean - train_scores_std, (train_scores_mean + train_scores_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(220,20,60,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.concatenate([train_sizes, train_sizes[::-1]]),
        y=np.concatenate([val_scores_mean - val_scores_std, (val_scores_mean + val_scores_std)[::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,0,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    ))
    fig.update_layout(title="Learning Curve (Optimized RF)",
                      xaxis_title="Training Examples",
                      yaxis_title="R-squared",
                      template="plotly_white")
    fig.show()

def plot_residuals_plotly(y_test, y_pred):
    """
    Creates an interactive Plotly scatter plot of residuals vs. predicted values.
    """
    residuals = y_test - y_pred
    fig = px.scatter(x=y_pred, y=residuals,
                     labels={'x': 'Predicted AQI', 'y': 'Residuals'},
                     title="Residuals vs. Predicted Values (Test Set)",
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Zero Error")
    fig.show()

def plot_predicted_vs_actual(y_test, y_pred):
    """
    Creates an interactive Plotly scatter plot of actual vs. predicted AQI values,
    adds a line of perfect prediction, saves the chart to HTML, and displays it.
    """
    fig = px.scatter(x=y_test, y=y_pred,
                     labels={'x': 'Actual AQI', 'y': 'Predicted AQI'},
                     title="Predicted vs Actual AQI",
                     template="plotly_white",
                     color_discrete_sequence=px.colors.qualitative.Set1)
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode="lines",
        name="Ideal",
        line=dict(color="red", dash="dash")
    ))
    chart_path_html = os.path.join(CHARTS_DIR, "predicted_vs_actual.html")
    fig.write_html(chart_path_html)
    logging.info(f"Interactive Predicted vs Actual AQI chart saved to {chart_path_html}")
    fig.show()

# ---------------------------
# Main Workflow
# ---------------------------
def main():
    csv_path = 'AQI.csv'

    # Load and preprocess the dataset
    df = load_and_preprocess_data(csv_path)

    # Print basic dataset information to the console (for debugging)
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nDescriptive Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # ---------------------------
    # Interactive EDA with Plotly
    # ---------------------------
    # Plot distributions for numeric columns
    for col in df.select_dtypes(include=['number']).columns:
        if df[col].notnull().any():
            fig = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}",
                               template="plotly_white",
                               color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.show()

    # Plot frequency for categorical columns using Plotly
    for col in df.select_dtypes(include=['object']).columns:
        vc_df = df[col].value_counts().reset_index()
        vc_df.columns = [col, 'count']
        fig = px.bar(vc_df, x=col, y='count', title=f"Frequency of {col}",
                     template="plotly_white",
                     color=col,
                     color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(xaxis_title=col, yaxis_title="Count")
        fig.show()

    # Correlation heatmap using Plotly
    if not df.select_dtypes(include=[np.number]).empty:
        corr_matrix = df.select_dtypes(include=[np.number]).corr()
        fig = px.imshow(corr_matrix,
                        labels=dict(x="Variables", y="Variables", color="Correlation"),
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        color_continuous_scale=px.colors.diverging.RdBu,
                        title="Correlation Matrix of Numerical Features")
        fig.update_layout(template="plotly_white")
        fig.show()

    # ---------------------------
    # Data Cleaning: Winsorize and Remove Duplicates
    # ---------------------------
    for col in ['Min', 'Max', 'Avg', 'AQI']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"Winsorized '{col}': lower: {lower_bound}, upper: {upper_bound}")
    num_duplicates = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"Removed {num_duplicates} duplicate rows.")
    if df['AQI'].dtype != 'int64':
        df['AQI'] = df['AQI'].astype('int64')
        print("Converted AQI to int64.")

    # ---------------------------
    # Feature Engineering and Encoding
    # ---------------------------
    df, le_pollutant, le_pred_param = feature_engineering(df)

    # Recreate additional engineered features for consistency
    df['AQI_Range'] = df['Max'] - df['Min']
    df['AQI_StdDev'] = df[['Min', 'Max', 'Avg']].std(axis=1)
    df['AQI_Range_Avg_Interaction'] = df['AQI_Range'] * df['Avg']
    df['AQI_StdDev_Avg_Interaction'] = df['AQI_StdDev'] * df['Avg']
    df['AQI_Range_Squared'] = df['AQI_Range'] ** 2
    df['AQI_StdDev_Cubed'] = df['AQI_StdDev'] ** 3
    logging.info("Created additional engineered features.")

    # Scale numerical features
    numerical_features = ['Latitude', 'Longitude', 'Min', 'Max', 'Avg', 'AQI',
                          'Pollutant_Avg', 'AQI_Range', 'AQI_StdDev',
                          'AQI_Range_Avg_Interaction', 'AQI_StdDev_Avg_Interaction',
                          'AQI_Range_Squared', 'AQI_StdDev_Cubed']
    df, scaler = scale_features(df, numerical_features)

    # ---------------------------
    # Data Splitting (70% Train, 15% Validation, 15% Test)
    # ---------------------------
    X = df.drop(['Country', 'State', 'City', 'Station', 'Last Update', 'Pollutant', 'AQI'], axis=1, errors='ignore')
    y = df['AQI']
    X = X.dropna()
    y = y[X.index]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # ---------------------------
    # Model Training: Train Multiple Models
    # ---------------------------
    linear_reg = LinearRegression()
    rf_reg = RandomForestRegressor(random_state=42)
    gb_reg = GradientBoostingRegressor(random_state=42)

    # Train models on the training set
    linear_reg.fit(X_train, y_train)
    rf_reg.fit(X_train, y_train)
    gb_reg.fit(X_train, y_train)

    # Predictions on validation set
    linear_pred = linear_reg.predict(X_val)
    rf_pred = rf_reg.predict(X_val)
    gb_pred = gb_reg.predict(X_val)

    # Evaluate models on the validation set
    def evaluate_model(y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        return r2, rmse, mae

    r2_linear, rmse_linear, mae_linear = evaluate_model(y_val, linear_pred)
    r2_rf, rmse_rf, mae_rf = evaluate_model(y_val, rf_pred)
    r2_gb, rmse_gb, mae_gb = evaluate_model(y_val, gb_pred)

    model_performance = {
        'LinearRegression': {'R2': r2_linear, 'RMSE': rmse_linear, 'MAE': mae_linear},
        'RandomForestRegressor': {'R2': r2_rf, 'RMSE': rmse_rf, 'MAE': mae_rf},
        'GradientBoostingRegressor': {'R2': r2_gb, 'RMSE': rmse_gb, 'MAE': mae_gb}
    }

    print("\nInitial Model Performance on Validation Set:")
    for model_name, metrics in model_performance.items():
        print(f"{model_name} => R2: {metrics['R2']:.3f}, RMSE: {metrics['RMSE']:.3f}, MAE: {metrics['MAE']:.3f}")

    # ---------------------------
    # Model Optimization: Hyperparameter Tuning for Random Forest
    # ---------------------------
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    optimized_rf = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(estimator=optimized_rf, param_distributions=param_grid,
                                       n_iter=10, scoring='neg_mean_squared_error',
                                       cv=5, verbose=1, random_state=42)
    random_search.fit(X_train, y_train)
    print("\nBest hyperparameters:", random_search.best_params_)
    print("Best (negative) MSE score:", random_search.best_score_)

    best_rf_reg = RandomForestRegressor(**random_search.best_params_, random_state=42)
    best_rf_reg.fit(X_train, y_train)

    # Evaluate optimized model on validation set
    y_val_pred = best_rf_reg.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))
    mae_val = mean_absolute_error(y_val, y_val_pred)

    print("\nOptimized Random Forest Performance on Validation Set:")
    print(f"R-squared: {r2_val:.3f}")
    print(f"RMSE: {rmse_val:.3f}")
    print(f"MAE: {mae_val:.3f}")

    # Evaluate optimized model on test set
    y_test_pred = best_rf_reg.predict(X_test)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)  # For metrics saving

    print("\nTest Set Performance:")
    print(f"R-squared: {r2_test:.3f}")
    print(f"RMSE: {rmse_test:.3f}")
    print(f"MAE: {mae_test:.3f}")

    print("\nComparison with Validation Set Performance:")
    print(f"R-squared (Validation): {r2_val:.3f}")
    print(f"RMSE (Validation): {rmse_val:.3f}")
    print(f"MAE (Validation): {mae_val:.3f}")

    if abs(r2_val - r2_test) > 0.1 or abs(rmse_val - rmse_test) > 0.1 or abs(mae_val - mae_test) > 0.1:
        print("\nSignificant differences observed between validation and test performance.")
    else:
        print("\nNo significant differences observed between validation and test performance.")

    # ---------------------------
    # Plotly Visualizations for Model Diagnostics
    # ---------------------------
    plot_feature_importance_plotly(best_rf_reg.feature_importances_, X_train.columns)
    plot_learning_curve_plotly(best_rf_reg, X_train, y_train)
    plot_residuals_plotly(y_test, y_test_pred)
    plot_predicted_vs_actual(y_test, y_test_pred)

    # ---------------------------
    # Save Model Bundle and Metrics
    # ---------------------------
    metrics_dict = {'MSE': mse_test, 'MAE': mae_test, 'R-Squared': r2_test}
    save_model_bundle(best_rf_reg, le_pollutant, le_pred_param, scaler, metrics_dict)
    save_metrics(metrics_dict)

if __name__ == '__main__':
    main()
