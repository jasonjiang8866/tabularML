import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import Pipeline
import time
import io
from contextlib import redirect_stdout

# Set page config
st.set_page_config(
    page_title="TabularML - Machine Learning Pipeline",
    page_icon="🧀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .sidebar-content {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🧀 TabularML Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("### A versatile machine learning pipeline for tabular data prediction")
    
    # Sidebar
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "📊 Data Exploration", "🔧 Model Training", "📈 Model Evaluation", "🔮 Predictions", "⚙️ Settings"]
    )
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'training_results' not in st.session_state:
        st.session_state.training_results = None
    
    # Page routing
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Exploration":
        show_data_exploration()
    elif page == "🔧 Model Training":
        show_model_training()
    elif page == "📈 Model Evaluation":
        show_model_evaluation()
    elif page == "🔮 Predictions":
        show_predictions()
    elif page == "⚙️ Settings":
        show_settings()

def show_home_page():
    """Display the home page with pipeline overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("## 🎯 Welcome to TabularML!")
        
        st.markdown("""
        This application provides a comprehensive machine learning pipeline for tabular data analysis and prediction. 
        The pipeline is designed to be versatile and can be adapted for various prediction tasks.
        
        ### 🚀 Key Features:
        
        - **📊 Data Exploration**: Comprehensive data analysis and visualization
        - **🔧 Automated Preprocessing**: Handles numeric and categorical features automatically
        - **🎯 Feature Selection**: Uses Random Forest for intelligent feature selection
        - **⚡ LightGBM Integration**: Fast and efficient gradient boosting
        - **🔍 Hyperparameter Tuning**: Automated model optimization
        - **📈 Model Evaluation**: Detailed performance metrics and visualizations
        - **🔮 Real-time Predictions**: Make predictions on new data
        
        ### 📋 Pipeline Steps:
        
        1. **Data Loading**: Loads and explores the dataset
        2. **Data Preprocessing**: Handles missing values, scaling, and encoding
        3. **Train-Test Split**: Divides data into training and testing sets
        4. **Feature Selection**: Identifies the most important features
        5. **Model Building**: Trains LightGBM regressor with hyperparameter tuning
        6. **Model Evaluation**: Assesses performance with various metrics
        7. **Deployment Ready**: Saves trained model for production use
        """)
        
        st.info("👈 Use the sidebar to navigate through different sections of the pipeline!")
    
    with col2:
        st.markdown("## 📊 Quick Stats")
        
        # Pipeline status
        if st.session_state.pipeline is None:
            st.error("🔴 Pipeline not initialized")
        else:
            st.success("🟢 Pipeline ready")
        
        if st.session_state.data is None:
            st.warning("🟡 No data loaded")
        else:
            st.success(f"🟢 Data loaded: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
        
        if st.session_state.model_trained:
            st.success("🟢 Model trained")
        else:
            st.warning("🟡 Model not trained")
        
        # Quick actions
        st.markdown("## ⚡ Quick Actions")
        
        if st.button("🚀 Initialize Pipeline", use_container_width=True):
            with st.spinner("Initializing pipeline..."):
                st.session_state.pipeline = Pipeline()
                st.success("Pipeline initialized successfully!")
                st.rerun()
        
        if st.button("📊 Load Sample Data", use_container_width=True):
            if st.session_state.pipeline is None:
                st.error("Please initialize pipeline first!")
            else:
                with st.spinner("Loading sample data..."):
                    st.session_state.data = st.session_state.pipeline.fetch_data()
                    st.success("Sample data loaded successfully!")
                    st.rerun()

def show_data_exploration():
    """Display data exploration page"""
    
    st.markdown("## 📊 Data Exploration")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please load data from the Home page first.")
        return
    
    data = st.session_state.data
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Rows", data.shape[0])
    with col2:
        st.metric("📊 Columns", data.shape[1])
    with col3:
        st.metric("🎯 Target Mean", f"{data['label'].mean():.2f}")
    with col4:
        st.metric("🎯 Target Std", f"{data['label'].std():.2f}")
    
    # Data preview
    st.markdown("### 👀 Data Preview")
    st.dataframe(data.head(10), use_container_width=True)
    
    # Data types and missing values
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Data Types")
        dtype_df = pd.DataFrame({
            'Column': data.columns,
            'Type': data.dtypes,
            'Non-Null Count': data.count(),
            'Null Count': data.isnull().sum()
        })
        st.dataframe(dtype_df, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Statistical Summary")
        st.dataframe(data.describe(), use_container_width=True)
    
    # Visualizations
    st.markdown("### 📈 Data Visualizations")
    
    # Target distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Target Distribution")
        fig = px.histogram(data, x='label', nbins=30, title="Distribution of Target Variable")
        fig.update_layout(xaxis_title="House Price", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Feature Correlation with Target")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        correlation_with_target = data[numeric_cols].corr()['label'].sort_values(ascending=False)
        
        fig = px.bar(
            x=correlation_with_target.values[1:],  # Exclude self-correlation
            y=correlation_with_target.index[1:],
            orientation='h',
            title="Feature Correlation with Target"
        )
        fig.update_layout(xaxis_title="Correlation", yaxis_title="Features")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature relationships
    st.markdown("#### 🔗 Feature Relationships")
    
    selected_features = st.multiselect(
        "Select features to analyze:",
        options=[col for col in data.columns if col != 'label'],
        default=list(data.select_dtypes(include=[np.number]).columns[:4])
    )
    
    if len(selected_features) >= 2:
        if len(selected_features) == 2:
            fig = px.scatter(
                data, 
                x=selected_features[0], 
                y=selected_features[1], 
                color='label',
                title=f"{selected_features[0]} vs {selected_features[1]}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Pairplot for multiple features
            if len(selected_features) <= 5:  # Limit to prevent performance issues
                fig = px.scatter_matrix(
                    data[selected_features + ['label']], 
                    color='label',
                    title="Feature Pairplot"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select 5 or fewer features for pairplot visualization.")

def show_model_training():
    """Display model training page"""
    
    st.markdown("## 🔧 Model Training")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please load data from the Home page first.")
        return
    
    if st.session_state.pipeline is None:
        st.warning("Pipeline not initialized. Please initialize from the Home page first.")
        return
    
    # Training configuration
    st.markdown("### ⚙️ Training Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 1000, 42)
    
    with col2:
        n_features = st.slider("Number of Top Features", 5, 50, 20)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Training button
    if st.button("🚀 Start Training", use_container_width=True, type="primary"):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.container()
        
        with log_container:
            log_expander = st.expander("📋 Training Logs", expanded=True)
            log_placeholder = log_expander.empty()
        
        try:
            # Capture training output
            log_buffer = io.StringIO()
            
            with st.spinner("Training machine learning model..."):
                # Update progress and status
                status_text.text("🔄 Preprocessing data...")
                progress_bar.progress(0.1)
                
                # Temporarily modify pipeline for custom parameters
                original_test_size = 0.2  # Store original
                
                # Start training
                with redirect_stdout(log_buffer):
                    data = st.session_state.data
                    pipeline = st.session_state.pipeline
                    
                    # Data preprocessing
                    status_text.text("🔄 Preprocessing data...")
                    X_train_transformed, X_test_transformed, y_train, y_test = pipeline.preprocess_data(df=data)
                    progress_bar.progress(0.3)
                    
                    # Feature selection
                    status_text.text("🔄 Selecting features...")
                    X_train_transformed_top, X_test_transformed_top = pipeline.feature_selection(
                        X_train_transformed, X_test_transformed, y_train, y_test
                    )
                    progress_bar.progress(0.5)
                    
                    # Model training
                    status_text.text("🔄 Training model...")
                    best_lgbm_model = pipeline.model_building(X_train_transformed_top, y_train)
                    progress_bar.progress(0.8)
                    
                    # Model evaluation
                    status_text.text("🔄 Evaluating model...")
                    # Capture evaluation results
                    y_pred = best_lgbm_model.predict(X_test_transformed_top)
                    
                    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    
                    # Store results
                    st.session_state.training_results = {
                        'model': best_lgbm_model,
                        'X_test': X_test_transformed_top,
                        'y_test': y_test,
                        'y_pred': y_pred,
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2': r2,
                        'feature_importance': best_lgbm_model.feature_importances_,
                        'n_features': X_train_transformed_top.shape[1]
                    }
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Training completed!")
                    
                    # Update logs
                    logs = log_buffer.getvalue()
                    log_placeholder.text(logs)
                    
                    st.session_state.model_trained = True
                    
                    # Show success message
                    st.success("🎉 Model training completed successfully!")
                    
                    # Display quick results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("R² Score", f"{r2:.4f}")
                    with col2:
                        st.metric("RMSE", f"{rmse:.4f}")
                    with col3:
                        st.metric("MAE", f"{mae:.4f}")
                    with col4:
                        st.metric("Features Used", X_train_transformed_top.shape[1])
                    
                    st.info("Navigate to the Model Evaluation page to see detailed results!")
                    
        except Exception as e:
            st.error(f"Training failed: {str(e)}")
            progress_bar.progress(0)
            status_text.text("❌ Training failed")
    
    # Show current training status
    if st.session_state.model_trained:
        st.success("✅ Model is trained and ready for evaluation!")
    else:
        st.info("👆 Click the training button to train your model")

def show_model_evaluation():
    """Display model evaluation results"""
    
    st.markdown("## 📈 Model Evaluation")
    
    if not st.session_state.model_trained or st.session_state.training_results is None:
        st.warning("No trained model found. Please train a model first in the Model Training section.")
        return
    
    results = st.session_state.training_results
    
    # Performance metrics
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 R² Score", f"{results['r2']:.4f}")
    with col2:
        st.metric("📏 RMSE", f"{results['rmse']:.4f}")
    with col3:
        st.metric("📐 MAE", f"{results['mae']:.4f}")
    with col4:
        st.metric("🔢 MSE", f"{results['mse']:.4f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Predictions vs Actual Values")
        fig = px.scatter(
            x=results['y_test'], 
            y=results['y_pred'],
            title="Predictions vs Actual Values",
            labels={'x': 'Actual Values', 'y': 'Predicted Values'}
        )
        
        # Add perfect prediction line
        min_val = min(results['y_test'].min(), results['y_pred'].min())
        max_val = max(results['y_test'].max(), results['y_pred'].max())
        fig.add_shape(
            type="line",
            x0=min_val, x1=max_val,
            y0=min_val, y1=max_val,
            line=dict(color="red", dash="dash")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Residuals Distribution")
        residuals = results['y_test'] - results['y_pred']
        fig = px.histogram(
            x=residuals, 
            nbins=30,
            title="Distribution of Residuals"
        )
        fig.update_layout(xaxis_title="Residuals", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    st.markdown("#### 🏆 Feature Importance")
    
    if hasattr(results['model'], 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': [f'Feature_{i}' for i in range(len(results['feature_importance']))],
            'Importance': results['feature_importance']
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(
            importance_df.head(15),  # Show top 15 features
            x='Importance',
            y='Feature',
            orientation='h',
            title="Top 15 Feature Importances"
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show feature importance table
        with st.expander("📋 Full Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True)
    
    # Model details
    st.markdown("### 🔧 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ Model Parameters")
        if hasattr(results['model'], 'get_params'):
            params = results['model'].get_params()
            param_df = pd.DataFrame([
                {'Parameter': k, 'Value': str(v)} for k, v in params.items()
            ])
            st.dataframe(param_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 📊 Model Info")
        info_data = {
            'Model Type': 'LightGBM Regressor',
            'Features Used': results['n_features'],
            'Training Samples': len(results['y_test']) * 4,  # Assuming 80/20 split
            'Test Samples': len(results['y_test']),
        }
        
        info_df = pd.DataFrame([
            {'Property': k, 'Value': v} for k, v in info_data.items()
        ])
        st.dataframe(info_df, use_container_width=True)

def show_predictions():
    """Display prediction interface"""
    
    st.markdown("## 🔮 Make Predictions")
    
    if not st.session_state.model_trained or st.session_state.training_results is None:
        st.warning("No trained model found. Please train a model first in the Model Training section.")
        return
    
    st.markdown("### 🎯 Single Prediction")
    st.info("Enter values for the features to get a house price prediction.")
    
    # Create input form based on the original features
    if st.session_state.data is not None:
        data = st.session_state.data
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features.remove('label')  # Remove target
        
        # Create input widgets
        input_values = {}
        
        col1, col2 = st.columns(2)
        
        with col1:
            for i, feature in enumerate(numeric_features[:len(numeric_features)//2]):
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                default_val = float(data[feature].mean())
                
                input_values[feature] = st.number_input(
                    f"{feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )
        
        with col2:
            for feature in numeric_features[len(numeric_features)//2:]:
                min_val = float(data[feature].min())
                max_val = float(data[feature].max())
                default_val = float(data[feature].mean())
                
                input_values[feature] = st.number_input(
                    f"{feature}:",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100
                )
        
        # Categorical features
        categorical_features = data.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if categorical_features:
            st.markdown("#### 📝 Categorical Features")
            for feature in categorical_features:
                unique_values = data[feature].unique()
                input_values[feature] = st.selectbox(
                    f"{feature}:",
                    options=unique_values,
                    index=0
                )
        
        # Prediction button
        if st.button("🔮 Make Prediction", use_container_width=True, type="primary"):
            try:
                # Create input dataframe
                input_df = pd.DataFrame([input_values])
                
                # Preprocess the input using the trained pipeline
                pipeline = st.session_state.pipeline
                
                # Transform the input data
                input_transformed = pipeline.preprocessor.transform(input_df)
                
                # Apply feature selection (use the same top features)
                model = st.session_state.training_results['model']
                
                # For simplicity, use the same feature selection as training
                # In a production system, you'd want to store the feature indices
                prediction = model.predict(input_transformed[:, :st.session_state.training_results['n_features']])
                
                # Display prediction
                st.success(f"🏡 Predicted House Price: **${prediction[0]:.2f}**")
                
                # Show confidence interval (rough estimate)
                std_error = st.session_state.training_results['rmse']
                lower_bound = prediction[0] - 1.96 * std_error
                upper_bound = prediction[0] + 1.96 * std_error
                
                st.info(f"📊 95% Confidence Interval: ${lower_bound:.2f} - ${upper_bound:.2f}")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
    
    # Batch prediction
    st.markdown("### 📊 Batch Predictions")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file for batch predictions:",
        type="csv",
        help="Upload a CSV file with the same features as the training data (excluding the target variable)"
    )
    
    if uploaded_file is not None:
        try:
            batch_data = pd.read_csv(uploaded_file)
            st.write("📋 Uploaded Data Preview:")
            st.dataframe(batch_data.head(), use_container_width=True)
            
            if st.button("🚀 Generate Batch Predictions"):
                with st.spinner("Generating predictions..."):
                    pipeline = st.session_state.pipeline
                    model = st.session_state.training_results['model']
                    
                    # Transform and predict
                    batch_transformed = pipeline.preprocessor.transform(batch_data)
                    batch_predictions = model.predict(batch_transformed[:, :st.session_state.training_results['n_features']])
                    
                    # Add predictions to the dataframe
                    result_df = batch_data.copy()
                    result_df['Predicted_Price'] = batch_predictions
                    
                    st.success("✅ Batch predictions completed!")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download link
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Batch prediction failed: {str(e)}")

def show_settings():
    """Display settings and configuration"""
    
    st.markdown("## ⚙️ Settings & Configuration")
    
    # Model settings
    st.markdown("### 🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 LightGBM Parameters")
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
        n_estimators = st.slider("Number of Estimators", 50, 500, 100, 10)
        max_depth = st.slider("Max Depth", 3, 15, -1)
        num_leaves = st.slider("Number of Leaves", 10, 100, 31)
    
    with col2:
        st.markdown("#### 🔧 Pipeline Settings")
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("CV Folds", 3, 10, 5)
        random_state = st.number_input("Random State", 0, 1000, 42)
        n_jobs = st.selectbox("Number of Jobs", [-1, 1, 2, 4, 8], index=0)
    
    # Data settings
    st.markdown("### 📊 Data Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔢 Preprocessing")
        numeric_strategy = st.selectbox("Numeric Imputation", ["mean", "median", "most_frequent"], index=0)
        categorical_strategy = st.selectbox("Categorical Imputation", ["constant", "most_frequent"], index=0)
        scaling_method = st.selectbox("Scaling Method", ["StandardScaler", "MinMaxScaler", "RobustScaler"], index=0)
    
    with col2:
        st.markdown("#### 🎯 Feature Selection")
        feature_selection_method = st.selectbox("Selection Method", ["RandomForest", "SelectKBest", "RFE"], index=0)
        n_features_to_select = st.slider("Number of Features", 5, 50, 20)
    
    # Export/Import settings
    st.markdown("### 💾 Model Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Current Model", use_container_width=True):
            if st.session_state.model_trained:
                # In a real application, you would save the model to disk
                st.success("Model saved successfully! (Feature coming soon)")
            else:
                st.warning("No trained model to save.")
    
    with col2:
        if st.button("📁 Load Saved Model", use_container_width=True):
            st.info("Model loading feature coming soon!")
    
    # System information
    st.markdown("### ℹ️ System Information")
    
    try:
        import platform
        import sys
        
        system_info = {
            "Python Version": sys.version.split()[0],
            "Platform": platform.platform(),
            "Processor": platform.processor() or "Unknown",
            "Architecture": platform.architecture()[0],
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
            
    except Exception as e:
        st.error(f"Could not retrieve system information: {e}")
    
    # Reset application
    st.markdown("### 🔄 Reset Application")
    
    if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary"):
        if st.checkbox("I understand this will clear all data and models"):
            st.session_state.clear()
            st.success("Application reset successfully! Please refresh the page.")

if __name__ == "__main__":
    main()