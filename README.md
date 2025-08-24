# 🧀 TabularML - Advanced ML Pipeline with Streamlit UI

A comprehensive machine learning pipeline for tabular data with a beautiful Streamlit web interface and automated UV environment setup.

## ✨ Features

### 🚀 Machine Learning Pipeline
- **Automated Data Processing**: Handles numeric and categorical features automatically
- **Smart Feature Selection**: Uses Random Forest for intelligent feature selection
- **LightGBM Integration**: Fast and efficient gradient boosting algorithm
- **Hyperparameter Tuning**: Automated model optimization with GridSearchCV
- **Comprehensive Evaluation**: Detailed performance metrics and visualizations
- **Model Persistence**: Saves trained models for deployment

### 🎨 Streamlit Web Interface
- **Interactive Dashboard**: Beautiful, responsive web interface
- **Data Exploration**: Comprehensive data analysis and visualizations
- **Real-time Training**: Live progress tracking during model training
- **Model Evaluation**: Detailed performance metrics with interactive charts
- **Prediction Interface**: Make predictions on new data with confidence intervals
- **Batch Processing**: Upload CSV files for batch predictions

### ⚡ UV Package Management
- **Fast Environment Setup**: Automated dependency management with UV
- **Cross-platform Scripts**: Works on Windows, macOS, and Linux
- **Reproducible Builds**: Locked dependencies for consistent environments

## 🛠️ Quick Start

### Option 1: Automatic Setup (Recommended)

#### Linux/macOS:
```bash
# Make setup script executable and run
chmod +x setup.sh
./setup.sh
```

#### Windows:
```cmd
# Run the setup batch file
setup.bat
```

### Option 2: Manual Setup

1. **Install UV** (if not already installed):
```bash
pip install uv
```

2. **Initialize the environment**:
```bash
uv sync
```

3. **Run the application**:
```bash
# Run the Streamlit UI
uv run streamlit run ui.py

# Or run the pipeline directly
uv run python pipeline.py
```

## 🎮 Using the Application

### 1. Launch the Web Interface
```bash
uv run streamlit run ui.py
```
Then open your browser to `http://localhost:8501`

### 2. Navigate Through the Pipeline

#### 🏠 Home Page
- Initialize the pipeline
- Load sample data
- View system status

#### 📊 Data Exploration
- View dataset statistics and metrics
- Explore data distributions and correlations
- Analyze feature relationships with interactive plots

#### 🔧 Model Training
- Configure training parameters
- Start model training with live progress tracking
- View training logs and results

#### 📈 Model Evaluation
- Detailed performance metrics (R², RMSE, MAE, MSE)
- Predictions vs Actual scatter plots
- Residuals distribution analysis
- Feature importance charts
- Model parameter inspection

#### 🔮 Predictions
- **Single Predictions**: Enter feature values for individual predictions
- **Batch Predictions**: Upload CSV files for bulk processing
- **Confidence Intervals**: Get prediction uncertainty estimates

#### ⚙️ Settings
- Configure model parameters
- Adjust preprocessing options
- System information and controls

## 📊 Pipeline Architecture

The ML pipeline follows these steps:

1. **Data Loading**: Loads dataset (with fallback to synthetic data)
2. **Data Preprocessing**: Handles missing values, scaling, and encoding
3. **Train-Test Split**: Divides data into training and testing sets
4. **Feature Selection**: Identifies top features using Random Forest
5. **Model Building**: Trains LightGBM with hyperparameter tuning
6. **Model Evaluation**: Comprehensive performance assessment
7. **Deployment**: Saves model for production use

## 🔧 Configuration

### Dependencies (pyproject.toml)
- **Core ML**: pandas, scikit-learn, lightgbm, numpy
- **Visualization**: matplotlib, plotly, seaborn
- **Web Interface**: streamlit
- **Utilities**: joblib for model persistence

### UV Scripts
The setup includes pre-configured UV scripts for common tasks:
- Environment initialization
- Dependency installation  
- Application launching

## 🎯 Sample Dataset

The application includes a synthetic housing dataset with:
- **1000 samples** with 12 features
- **Numeric features**: Income, house age, rooms, location, etc.
- **Categorical features**: Property type, year built
- **Target**: House price prediction

## 📈 Performance

The pipeline achieves excellent performance on the sample dataset:
- **R² Score**: ~0.97 (97% variance explained)
- **RMSE**: ~0.99 (low prediction error)
- **Training Time**: ~15 seconds for full pipeline

## 🔍 Advanced Features

### Interactive Visualizations
- Target distribution histograms
- Feature correlation heatmaps
- Scatter plot matrices
- Predictions vs actual charts
- Residuals analysis

### Model Insights
- Feature importance rankings
- Model parameter inspection
- Training progress tracking
- Comprehensive evaluation metrics

### Production Ready
- Model serialization with joblib
- Batch prediction capabilities
- Error handling and validation
- Scalable architecture

## 🚀 Extending the Pipeline

### Adding New Datasets
1. Modify the `fetch_data()` method in `pipeline.py`
2. Ensure your data has a 'label' column for the target
3. The pipeline automatically handles numeric/categorical features

### Customizing Models
1. Update the `model_building()` method
2. Modify hyperparameter grids in the training configuration
3. Add new evaluation metrics as needed

### UI Customization
1. Modify `ui.py` to add new pages or features
2. Update the navigation and styling
3. Add new visualization types

## 📚 Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning toolkit
- **lightgbm**: Gradient boosting framework
- **streamlit**: Web application framework
- **plotly**: Interactive visualizations

### Development Tools
- **uv**: Fast Python package manager
- **pytest**: Testing framework (optional)
- **black**: Code formatting (optional)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🎉 Acknowledgments

- Built with modern Python ML stack
- Inspired by best practices in MLOps
- Designed for both beginners and experts
- Emphasis on user experience and visualization

---

**Ready to explore your data? Start with `./setup.sh` and launch the Streamlit interface!** 🚀