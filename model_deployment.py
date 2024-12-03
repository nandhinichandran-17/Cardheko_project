import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessing steps
model = joblib.load("C:/Users/nandh/OneDrive/Desktop/cardheko/Car_Price_Prediction_Model.pkl")
label_encoders = joblib.load("C:/Users/nandh/OneDrive/Desktop/cardheko/Label_Encoders.pkl")
scalers = joblib.load("C:/Users/nandh/OneDrive/Desktop/cardheko/Scalers.pkl")

# Load dataset for filtering and identifying similar data
data = pd.read_csv("C:/Users/nandh/OneDrive/Desktop/cardheko/Car_Dekho_Cleaned_Dataset_Raw.csv")

# Set pandas option to handle future downcasting behavior
pd.set_option('future.no_silent_downcasting', True)

# Features used for training
features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']

# Function to filter data based on user selections
def filter_data(oem=None, model=None, body_type=None, fuel_type=None, seats=None):
    filtered_data = data.copy()
    if oem:
        filtered_data = filtered_data[filtered_data['oem'] == oem]
    if model:
        filtered_data = filtered_data[filtered_data['model'] == model]
    if body_type:
        filtered_data = filtered_data[filtered_data['bt'] == body_type]
    if fuel_type:
        filtered_data = filtered_data[filtered_data['ft'] == fuel_type]
    if seats:
        filtered_data = filtered_data[filtered_data['Seats'] == seats]
    return filtered_data

# Preprocessing function for user input
def preprocess_input(df):
    df['car_age'] = 2024 - df['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    df['brand_popularity'] = df['oem'].map(brand_popularity)
    df['mileage_normalized'] = df['mileage'] / df['car_age']

    # Apply label encoding
    for column in ['ft', 'bt', 'transmission', 'oem', 'model', 'variantName', 'City']:
        if column in df.columns and column in label_encoders:
            df[column] = df[column].apply(lambda x: label_encoders[column].transform([x])[0])

    # Apply min-max scaling
    for column in ['km', 'ownerNo', 'modelYear']:
        if column in df.columns and column in scalers:
            df[column] = scalers[column].transform(df[[column]])

    return df

# Streamlit Application
st.set_page_config(page_title="CarDekho Price Prediction", page_icon="🚗", layout="wide")


# Custom CSS to add animations
st.markdown(
    """
    <style>
    /* Custom background animation */
    @keyframes gradientBackground {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .reportview-container .main {
        background: linear-gradient(-45deg, #ff9a9e, #fad0c4, #fbc2eb, #f6d365, #fda085);
        background-size: 400% 400%;
        animation: gradientBackground 15s ease infinite;
        color: black;
        padding: 20px;
    }

    /* Floating Logo Animation */
    .logo-container img {
        width: 200px;
        margin-bottom: 10px;
        animation: floatLogo 2s ease-in-out infinite;
    }

    @keyframes floatLogo {
        0% { transform: translatey(0px); }
        50% { transform: translatey(-10px); }
        100% { transform: translatey(0px); }
    }

    /* Button Hover Effect */
    .stButton>button {
        background-color: #f67280;
        border: none;
        color: white;
        font-size: 1.2em;
        padding: 10px 20px;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }

    .stButton>button:hover {
        background-color: #6c5b7b;
        transform: scale(1.1);
        box-shadow: 0 6px 15px rgba(0, 0, 0, 0.4);
    }

    .stTextInput>div>div>input:focus {
        border-color: #f67280;
        box-shadow: 0 0 5px rgba(246, 114, 128, 0.5);
        transform: scale(1.05);
    }

    /* Input Field Focus Effect */
    .stSelectbox select:focus {
        border-color: #f67280;
        box-shadow: 0 0 5px rgba(246, 114, 128, 0.5);
    }

    /* Custom Styling for Results */
.result-container {
    background: rgba(255, 255, 255, 0.8);
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
    text-align: center;
    margin-top: 20px;
    opacity: 0;
    animation: fadeInUp 1s ease-in-out forwards;
}

@keyframes fadeInUp {
    0% {
        transform: translateY(20px);
        opacity: 0;
    }
    100% {
        transform: translateY(0);
        opacity: 1;
    }
}

/* Title Animation */
.result-title {
    font-size: 1.5em;
    color: #6a0572;
    margin-bottom: 10px;
    animation: textColorChange 3s ease-in-out infinite;
}

@keyframes textColorChange {
    0% {
        color: #6a0572;
    }
    50% {
        color: #f67280;
    }
    100% {
        color: #6a0572;
    }
}

/* Result Value Animation */
.result-value {
    font-size: 2em;
    font-weight: bold;
    color: #f67280;
    animation: scaleUp 1s ease-in-out infinite;
}

@keyframes scaleUp {
    0% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.1);
    }
    100% {
        transform: scale(1);
    }
}
/* Custom Styling for Car Age */
.car-age {
    font-size: 1.2em;
    color: #6c5b7b;
    margin-top: 10px;
    animation: scaleUp 1s ease-in-out forwards;
}
/* Hover Effect on Car Age */
.car-age:hover {
    color: #f67280;
    transform: scale(1.1);
    transition: all 0.3s ease-in-out;
}

/* Hover Effect on Result Container */
.result-container:hover {
    box-shadow: 0px 8px 25px rgba(0, 0, 0, 0.3);
    transform: scale(1.05);
    transition: all 0.3s ease-in-out;
}

/* Hover Effect on Result Title */
.result-title:hover {
    color: #f67280;
    transform: translateY(-5px);
    transition: all 0.3s ease-in-out;
}

/* Hover Effect on Result Value */
.result-value:hover {
    color: #6a0572;
    transform: scale(1.1);
    transition: all 0.3s ease-in-out;
}
<style>
    """,
    unsafe_allow_html=True
)

# Ensure the logo file is in the same directory or provide the correct path

with st.sidebar:
    st.image("C:/Users/nandh/OneDrive/Desktop/cardheko_logo.jpg", caption="CarDekho", use_container_width=True)

# Display the logo
# Embed the logo using HTML for more customization
# FontAwesome CDN for icons
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """,
    unsafe_allow_html=True,
)

st.title("🚘 CarDekho Used Car Price Prediction")

# User input tabs
tab1, tab2, tab3 = st.tabs(["Basic Details", "Car Features", "Usage Information"])

# Basic Details Tab
with tab1:
    st.header("Step 1: Basic Car Details")
    st.markdown("<i class='fas fa-industry' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    selected_oem = st.selectbox("Original Equipment Manufacturer (OEM)", data['oem'].unique())
    filtered_data = filter_data(oem=selected_oem)
    st.markdown("<i class='fas fa-car' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    selected_model = st.selectbox("Car Model", filtered_data['model'].unique())
    filtered_data = filter_data(oem=selected_oem, model=selected_model)
    st.markdown("<i class='fas fa-car-side' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    body_type = st.selectbox("Body Type", filtered_data['bt'].unique())

# Car Features Tab
with tab2:
    st.header("Step 2: Car Features")
    filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type)
    st.markdown("<i class='fas fa-gas-pump' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    fuel_type = st.selectbox("Fuel Type", filtered_data['ft'].unique())
    filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
    st.markdown("<i class='fas fa-cogs' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    transmission = st.selectbox("Transmission Type", filtered_data['transmission'].unique())
    filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type)
    st.markdown("<i class='fas fa-users' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    seat_count = st.selectbox("Seats", filtered_data['Seats'].unique())
    filtered_data = filter_data(oem=selected_oem, model=selected_model, body_type=body_type, fuel_type=fuel_type, seats=seat_count)
    st.markdown("<i class='fas fa-th-large' style='font-size: 35px;color:#f67280;'></i> Variant Name", unsafe_allow_html=True)
    selected_variant = st.selectbox("Variant Name", filtered_data['variantName'].unique())

# Usage Information Tab
with tab3:
    st.header("Step 3: Usage Information")
    st.markdown("<i class='fas fa-calendar' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True) 
    modelYear = st.number_input("Year of Manufacture", min_value=1980, max_value=2024, value=2015)
    st.markdown("<i class='fas fa-user-tie' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    ownerNo = st.number_input("Number of Previous Owners", min_value=0, max_value=10, value=1)
    st.markdown("<i class='fas fa-tachometer-alt' style='font-size: 35px;color:#f67280;'></i>", unsafe_allow_html=True)
    km = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=10000)
    # Adjust mileage slider
    min_mileage = np.floor(filtered_data['mileage'].min())
    max_mileage = np.ceil(filtered_data['mileage'].max())

    # Ensure mileage slider has an interval of 0.5
    min_mileage = float(min_mileage)
    max_mileage = float(max_mileage)
    st.markdown("<i class='fas fa-road' style='font-size: 35px;color:#f67280;'></i> Mileage", unsafe_allow_html=True)
    mileage = st.slider("Mileage (kmpl)", min_value=float(filtered_data['mileage'].min()), max_value=float(filtered_data['mileage'].max()), step=0.5)
    st.markdown("<i class='fas fa-city' style='font-size: 35px;color:#f67280;'></i> City", unsafe_allow_html=True)
    city = st.selectbox("City", data['City'].unique())

# Prediction Button in Sidebar
st.sidebar.header("🚀Ready to Predict?")


if st.sidebar.button("Predict"):
    user_input_data = {
        'ft': [fuel_type],
        'bt': [body_type],
        'km': [km],
        'transmission': [transmission],
        'ownerNo': [ownerNo],
        'oem': [selected_oem],
        'model': [selected_model],
        'modelYear': [modelYear],
        'variantName': [selected_variant],
        'City': [city],
        'mileage': [mileage],
        'Seats': [seat_count],
        'car_age': [2024 - modelYear],
        'brand_popularity': [data.groupby('oem')['price'].mean().to_dict().get(selected_oem)],
        'mileage_normalized': [mileage / (2024 - modelYear)]
    }

    user_df = pd.DataFrame(user_input_data)
    user_df = user_df[features]
    user_df = preprocess_input(user_df)

    try:
        prediction = model.predict(user_df)
        st.markdown(
            f"""
            <div class="result-container">
                <p class="result-title">Predicted Car Price:</p>
                <p class="result-value">₹{prediction[0]:,.2f}</p>
                <p class="info">Car Age: {user_df['car_age'][0]} years</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Error in prediction: {e}")

# Footer
#st.markdown("<p style='text-align: center;'>Application Designed by <b>Nandhini</b></p>", unsafe_allow_html=True)
