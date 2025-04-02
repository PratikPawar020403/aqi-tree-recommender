# AQI Insights Hub 🌳

An intelligent environmental platform that combines Air Quality Index (AQI) prediction with smart tree recommendations for urban air pollution management.

## 🌟 Key Features

### AQI Analysis & Prediction
- Real-time AQI predictions based on environmental parameters
- Interactive visualization of historical AQI data
- Geographical coordinate-based analysis
- Advanced pollutant concentration tracking

### Smart Tree Recommendations 🌱
- Intelligent tree species suggestions based on:
  - Predicted AQI levels
  - Predominant pollutants
  - Air Pollution Tolerance Index (APTI)
- Recommendations for both English and Indian tree names
- APTI value display for recommended trees

### Data Exploration
- Interactive charts and visualizations
- Historical AQI trend analysis
- Pollutant concentration patterns
- Geographical distribution of air quality

## 🛠️ Tech Stack

- **Frontend**: Streamlit with custom CSS for modern UI
- **Backend**: Python
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly, Seaborn
- **Data Storage**: CSV, XML formats

## 📊 Features in Detail

1. **AQI Prediction**
   - Input geographical coordinates
   - Specify pollutant concentrations (min, max, average)
   - Select predominant pollutant type
   - Get instant AQI predictions with interpretation

2. **Tree Recommendation System**
   - Automatic tree species suggestion based on AQI prediction
   - APTI (Air Pollution Tolerance Index) values for each recommendation
   - Specific tree recommendations for different pollutant types
   - Both local and scientific tree names provided

3. **Data Exploration Tools**
   - Interactive data visualization
   - Historical trend analysis
   - Pollutant concentration patterns
   - Geographical distribution maps

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/aqi-insights-hub.git
cd aqi-insights-hub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Navigate through the application:
   - **Home**: Overview and key features
   - **Predict**: Get AQI predictions and tree recommendations
   - **Explore**: Analyze historical data and patterns

## 📁 Project Structure

```
aqi-insights-hub/
├── app.py              # Main Streamlit application
├── predict.py          # AQI prediction and tree recommendation logic
├── explore.py          # Data exploration functionality
├── train.py           # Model training script
├── requirements.txt    # Project dependencies
├── data/
│   ├── AQI.csv        # Historical AQI data
│   ├── data_aqi_cpcb.xml  # CPCB AQI data
│   └── treeaqi.csv    # Tree recommendation data
└── model/
    └── model_bundle.pkl  # Trained ML model
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Central Pollution Control Board (CPCB) for AQI data
- Environmental research papers for tree APTI values
- Streamlit community for UI components
