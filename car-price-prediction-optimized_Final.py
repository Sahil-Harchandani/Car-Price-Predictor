import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import gc
import psutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E3A8A !important;
        margin-bottom: 1rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #1E3A8A !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        text-align: center;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 0.3rem;
        padding: 0.5rem 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
    .st-emotion-cache-16txtl3 {
        padding-top: 2rem;
    }
    .prediction-result {
        background-color: #F0F9FF;
        border-left: 5px solid #0EA5E9;
        padding: 1rem;
        border-radius: 0.3rem;
        margin-top: 1rem;
    }
    .sidebar-content {
        padding: 1rem;
    }
    .stSlider>div>div>div {
        background-color: #2563EB !important;
    }
</style>
""", unsafe_allow_html=True)

# Function to check memory usage
def check_memory():
    memory = psutil.virtual_memory()
    return {
        'total': memory.total / (1024 ** 3),  # GB
        'available': memory.available / (1024 ** 3),  # GB
        'percent': memory.percent
    }

# Function to train model using pandas and scikit-learn
def train_model_pandas(df):
    """Train a model using pandas and scikit-learn"""
    # Extract X and y
    X = df.drop(['AskPrice', 'BrandModel'], axis=1, errors='ignore')
    y = df['AskPrice']
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Create preprocessor for categorical and numeric data
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ])
    
    # Create pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Free memory
    del X, y
    gc.collect()
    
    # Train model
    model_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model_pipeline.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # For visualization
    vis_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    vis_df = vis_df.sample(n=min(500, len(vis_df)))
    
    return model_pipeline, vis_df, rmse, r2, mae

# Function to predict using pandas model
def predict_price_pandas(model, car_features):
    # Convert single feature dict to DataFrame
    input_df = pd.DataFrame([car_features])
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    
    return prediction

# Function to load and preprocess data (no memory reporting)
def load_and_preprocess_data(file_path):
    try:
        # Use pandas with low_memory mode
        pdf = pd.read_csv(file_path, usecols=lambda x: x not in ['PostedDate', 'AdditionInfo'])
        
        # Clean data in pandas
        pdf['AskPrice'] = pdf['AskPrice'].astype(str).str.replace('[\₹,]', '', regex=True).astype(float)
        pdf['kmDriven'] = pdf['kmDriven'].astype(str).str.replace('[ km,]', '', regex=True).astype(float)
        
        # Filter rows in pandas to reduce data
        pdf = pdf[(pdf['AskPrice'] > 50000) & (pdf['AskPrice'] < 10000000)]
        pdf = pdf[(pdf['kmDriven'] > 1000) & (pdf['kmDriven'] < 500000)]
        
        # Calculate Age if missing
        if 'Age' not in pdf.columns:
            current_year = 2025
            pdf['Age'] = current_year - pdf['Year'].astype(int)
        
        # Convert types
        pdf['Year'] = pdf['Year'].astype(int)
        pdf['Age'] = pdf['Age'].astype(int)
        
        # Create BrandModel column
        pdf['BrandModel'] = pdf['Brand'] + '_' + pdf['model']
        
        # Aggressively reduce dataset size
        if len(pdf) > 10000:  # Extreme reduction for 4GB RAM
            pdf = pdf.sample(n=10000, random_state=42)
            st.info("Dataset reduced to optimize performance", icon="ℹ️")
        
        return pdf
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to visualize predictions using plotly
def visualize_predictions_plotly(vis_df):
    # Create a figure with plotly
    fig = px.scatter(vis_df, x='Actual', y='Predicted', 
                     opacity=0.5,
                     labels={'Actual': 'Actual Price (₹)', 'Predicted': 'Predicted Price (₹)'},
                     title='Actual vs Predicted Prices')
    
    # Add diagonal line
    min_val = min(vis_df['Actual'].min(), vis_df['Predicted'].min())
    max_val = max(vis_df['Actual'].max(), vis_df['Predicted'].max())
    
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        template='plotly_white',
        height=500,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(t=50, l=0, r=0, b=0)
    )
    
    return fig

# Function to create a colorful gauge chart for model R² score
def create_r2_gauge(r2_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r2_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model R² Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.5], 'color': '#FF4B4B'},
                {'range': [0.5, 0.7], 'color': '#FFA500'},
                {'range': [0.7, 0.9], 'color': '#90EE90'},
                {'range': [0.9, 1], 'color': '#00CC96'}
            ],
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

# Function to create summary card of the prediction
def create_prediction_card(predicted_price, car_features):
    # Create card content
    card_html = f"""
    <div style="background-color: #F0F9FF; border-radius: 10px; padding: 20px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);">
        <h3 style="color: #1E3A8A; margin-bottom: 15px;">Prediction Result</h3>
        <div style="display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
            <div style="font-size: 2rem; font-weight: bold; color: #1E3A8A;">₹ {predicted_price:,.2f}</div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="background-color: #DBEAFE; padding: 10px; border-radius: 5px;">
                <strong>Brand:</strong> {car_features['Brand']}
            </div>
            <div style="background-color: #DBEAFE; padding: 10px; border-radius: 5px;">
                <strong>Model:</strong> {car_features['model']}
            </div>
            <div style="background-color: #DBEAFE; padding: 10px; border-radius: 5px;">
                <strong>Year:</strong> {car_features['Year']} ({car_features['Age']} years old)
            </div>
            <div style="background-color: #DBEAFE; padding: 10px; border-radius: 5px;">
                <strong>Kilometers:</strong> {car_features['kmDriven']:,.0f} km
            </div>
            <div style="background-color: #DBEAFE; padding: 10px; border-radius: 5px;">
                <strong>Transmission:</strong> {car_features['Transmission']}
            </div>
            <div style="background-color: #DBEAFE; padding: 10px; border-radius: 5px;">
                <strong>Fuel Type:</strong> {car_features['FuelType']}
            </div>
        </div>
    </div>
    """
    return card_html

# Main app function
def main():
    # Force garbage collection at start
    gc.collect()
    
    # Sidebar
    with st.sidebar:
        st.image("https://www.svgrepo.com/show/530438/machine-learning.svg", width=80)
        st.markdown("<h2 style='text-align: center;'>Car Price Predictor</h2>", unsafe_allow_html=True)
        st.markdown("---")
        
        # System info (simplified)
        memory_status = check_memory()
        st.markdown(f"""
        <div class='sidebar-content'>
            <h4>System Status</h4>
            <div class='metric-card'>
                <div class='metric-label'>Memory Usage</div>
                <div class='metric-value'>{memory_status['percent']}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # File upload
        st.markdown("<h4>Upload Data</h4>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your own dataset", type="csv")
        
        # Clear cache button
        st.markdown("---")
        if st.button("Clear Cache & Retrain"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            gc.collect()
            st.experimental_rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown("This app predicts used car prices based on various features like brand, model, year, and kilometers driven.")
        st.caption("© 2025 Car Price Predictor")
    
    # Main content
    st.markdown("<h1 class='main-header'>Used Car Price Predictor</h1>", unsafe_allow_html=True)
    
    # Path to the demo dataset
    demo_data_path = "used_car_dataset.csv"
    
    # Use uploaded file if available
    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as f:
            f.write(uploaded_file.getbuffer())
            temp_path = f.name
        demo_data_path = temp_path
    
    # Load and process data
    try:
        # Check if model exists in session state
        if 'model' not in st.session_state:
            with st.spinner("Loading data and training model..."):
                # Free memory before processing
                gc.collect()
                
                # Load data with pandas
                df = load_and_preprocess_data(demo_data_path)
                
                if df is None:
                    st.error("Failed to load data. Please check your CSV file format.")
                    return
                
                # Store valuable information for UI
                st.session_state.brands = sorted(df['Brand'].unique().tolist())
                st.session_state.transmissions = sorted(df['Transmission'].unique().tolist())
                st.session_state.fuel_types = sorted(df['FuelType'].unique().tolist())
                st.session_state.year_range = (df['Year'].min(), df['Year'].max())
                st.session_state.km_range = (df['kmDriven'].min(), df['kmDriven'].max())
                
                # Get available models for each brand
                models_by_brand = {}
                for brand in st.session_state.brands:
                    models = sorted(df[df['Brand'] == brand]['model'].unique().tolist())
                    models_by_brand[brand] = models
                
                st.session_state.models_by_brand = models_by_brand
                
                # Train model using pandas and sklearn
                model, vis_df, rmse, r2, mae = train_model_pandas(df)
                
                # Clear dataframe reference to free memory
                del df
                gc.collect()
                
                # Create visualizations
                pred_plot = visualize_predictions_plotly(vis_df)
                r2_gauge = create_r2_gauge(r2)
                
                # Clear visualization data reference
                del vis_df
                gc.collect()
                
                # Store everything in session state
                st.session_state.model = model
                st.session_state.rmse = rmse
                st.session_state.r2 = r2
                st.session_state.mae = mae
                st.session_state.pred_plot = pred_plot
                st.session_state.r2_gauge = r2_gauge
                
                # Free up memory
                gc.collect()
        
        # Create tabs for different sections
        tab1, tab2 = st.tabs(["Car Price Prediction", "Model Performance"])
        
        with tab1:
            st.markdown("<h2 class='sub-header'>Predict Car Price</h2>", unsafe_allow_html=True)
            
            # Create input form using columns
            col1, col2 = st.columns(2)
            
            with col1:
                brand = st.selectbox("Brand", st.session_state.brands)
                
                # Add model selection based on the selected brand
                available_models = st.session_state.models_by_brand.get(brand, [])
                if available_models:
                    model_name = st.selectbox("Model", available_models)
                else:
                    model_name = "Unknown"
                
                year = st.slider("Year", 
                                int(st.session_state.year_range[0]), 
                                int(st.session_state.year_range[1]), 
                                int((st.session_state.year_range[0] + st.session_state.year_range[1]) / 2))
                
                age = 2025 - year
                
            with col2:
                transmission = st.selectbox("Transmission", st.session_state.transmissions)
                fuel_type = st.selectbox("Fuel Type", st.session_state.fuel_types)
                
                kmDriven = st.slider("Kilometers Driven", 
                                   int(st.session_state.km_range[0]), 
                                   int(st.session_state.km_range[1]), 
                                   int((st.session_state.km_range[0] + st.session_state.km_range[1]) / 2),
                                   step=1000,
                                   format="%d km")
            
            # Make prediction when button is clicked
            predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
            with predict_col2:
                predict_button = st.button("Predict Price", use_container_width=True)
            
            if predict_button:
                # Create feature dictionary
                car_features = {
                    "Brand": brand,
                    "model": model_name,
                    "Year": year,
                    "Age": age,
                    "kmDriven": kmDriven,
                    "Transmission": transmission,
                    "Owner": "first",  # Default value
                    "FuelType": fuel_type
                }
                
                # Make prediction
                with st.spinner("Predicting..."):
                    predicted_price = predict_price_pandas(st.session_state.model, car_features)
                    
                    # Display prediction card
                    st.markdown(create_prediction_card(predicted_price, car_features), unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
            
            # Display metrics in nice cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Root Mean Square Error</div>
                    <div class="metric-value">₹{st.session_state.rmse:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value">{st.session_state.r2:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Mean Absolute Error</div>
                    <div class="metric-value">₹{st.session_state.mae:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display R² gauge
            st.plotly_chart(st.session_state.r2_gauge, use_container_width=True)
            
            # Display prediction scatter plot
            st.markdown("<h3 class='sub-header'>Prediction Accuracy</h3>", unsafe_allow_html=True)
            st.plotly_chart(st.session_state.pred_plot, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please ensure your CSV file is in the correct format.")
        with st.expander("Technical Details"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    # Force garbage collection before starting app
    gc.collect()
    main()
