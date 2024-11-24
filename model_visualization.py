import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import base64
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")
# Function to set a background image
def set_background_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            color: white;
        }}
        .stSidebar {{
            background-color: cherryred;
            color: black;
        }}
        .stButton button {{
            background-color: red;
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Define the main function
def main():

    # Set the background image
    set_background_image("C:/Users/nandh/Downloads/carimg.jpeg")  

    # File paths
    csv_path = "C:/Users/nandh/OneDrive/Desktop/cardheko2/final_cleaned_data.csv"
    model_path = "C:/Users/nandh/OneDrive/Desktop/cardheko2/gradient_boosting_model.pkl"

    # Check file existence
    if not os.path.exists(csv_path):
        st.error(f"File not found: {csv_path}")
        return
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return

    # Load the model
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return

    # Load the dataset
    data = pd.read_csv(csv_path)

    selected_page = option_menu(
        menu_title="Car Dheko - Used Car Price Prediction",
        options=["Home", "Features"],
        icons=["house", "info-circle"],
        orientation="horizontal"
    )
    if selected_page == "Home":
        st.title("🚗 Car Price Prediction")
        st.subheader("**Domain**: Automotive Industry, Data Science, Machine Learning")
        st.markdown("""
            This project leverages **Machine Learning** to predict the prices of used cars.  
            It is an interactive tool designed for both customers and sales representatives, 
            providing accurate car price predictions based on various features.
        """)
        st.subheader("**Problem Statement**:")
        st.markdown("The primary objective of is project is to create a data science solution for predicting used car prices accurately by analyzing a diverse dataset including car model, no. of owners, age, mileage, fuel type, kilometers driven, features and location. The aim is to build a machine learning model that offers users to find current valuations for used cars.")
        st.subheader("**Result**:")
        st.markdown("The culmination of this project will be a robust and user-friendly data science solution that leverages advanced machine learning techniques to predict used car prices with a high degree of accuracy. The end result will empower users to make informed decisions when buying or selling used cars, enhancing their overall experience in the automotive market.")
        st.subheader("**Skills Learnt**:")
        st.markdown("""
        - Data Cleaning and Preprocessing  
        - Exploratory Data Analysis  
        - Machine Learning Model Development  
        - Streamlit Application Development  
        - Model Deployment and Evaluation  
    """)
        st.subheader("**Developed By**: ")
        st.markdown("Nandhini")


    if selected_page == "Features":
        st.subheader("**Feature Explanation**:")
        
        st.markdown("""            
        - FT (Fuel Type):
            The type of fuel the car uses.
                    
            Examples:
                Petrol: Honda City petrol variant.
                Diesel: Mahindra Scorpio diesel model.
                Electric: Tesla Model 3 (runs on electricity).
                Hybrid: Toyota Prius (combines petrol and electricity).
                    
        - BT (Body Type):
            The overall structure or shape of the car.
                    
            Examples:
                Sedan: Toyota Camry.
                SUV: Hyundai Creta.
                Hatchback: Maruti Suzuki Swift.
                Coupe: Ford Mustang.
                    
        - Transmission:
            The type of gearbox the car has.
                    
            Examples:
                Manual: Mahindra Bolero (driver shifts gears manually).
                Automatic: Kia Seltos (automatic gear shifts).
                CVT (Continuously Variable Transmission): Honda Jazz (smooth gearless transitions).
                AMT (Automated Manual Transmission): Tata Tiago AMT.
                    
        - Model:
            The specific variant or version of a car.
                    
            Examples:
                Toyota Corolla Altis (Corolla is the series, Altis is the model).
                Maruti Suzuki Baleno Alpha (Baleno is the series, Alpha is the trim).
              
        - Insurance Validity:
            The period until the car's insurance policy remains active.
                     
            Examples:
                A new car might come with 1-year comprehensive insurance.
                Check the insurance certificate to see if the policy is active, e.g., valid till December 2025.
                    
        - Engine Displacement:
            The volume of the engine's cylinders in cubic centimeters (cc).
                    
            Examples:
                1000cc (1.0L): Renault Kwid.
                2000cc (2.0L): Jeep Compass.
                Larger displacement engines usually offer more power.
                
        - Drive Type:
            Indicates which wheels receive power from the engine.
                    
            Examples:
                FWD (Front Wheel Drive): Honda Civic.
                RWD (Rear Wheel Drive): BMW 3 Series.
                AWD (All Wheel Drive): Subaru Forester.
                4WD (Four Wheel Drive): Toyota Fortuner (suitable for off-roading).
              
        - City:
            Refers to the car's location, either where it is registered or primarily used.
                    
            Example: 
                    A car registered in Mumbai may have different tax and insurance costs compared to Delhi.
                           
        - Mileage:
            The distance a car can travel per unit of fuel, typically measured in kilometers per liter (km/l) or miles per gallon (mpg).
           
            Examples:
                Petrol car: Maruti Suzuki Alto (22 km/l).
                Diesel car: Hyundai Verna Diesel (25 km/l).
                    
        - Max Power:
            The peak power output of the engine, measured in horsepower (hp) or kilowatts (kW).
                    
            Examples:
                Maruti Suzuki Swift: 89 hp.
                BMW M5: 617 hp.

        - Torque:
            The rotational force of the engine, measured in Newton-meters (Nm).
                    
            Examples:
                Tata Harrier: 350 Nm (good for off-roading and towing).
                Honda City: 145 Nm (suitable for city driving).

        - Number of Cylinders:
            The count of cylinders in the engine that generate power.
                    
            Examples:
                3-cylinder: Renault Triber.
                4-cylinder: Hyundai i20.
                6-cylinder: Ford Mustang GT.      
                    
        - Car Age:
            The time elapsed since the car's manufacturing date.
                    
            Examples:
                A car manufactured in 2018 would be 6 years old in 2024.
                Older cars might face depreciation and lower resale value.

""")
    # Initialize scalers
    scalers = {}

    # Define mappings for categorical columns
    model_unique_values = data['model'].dropna().unique()
    model_mapping = {value: index + 1 for index, value in enumerate(model_unique_values)}

    # Create mappings for all categorical columns
    categorical_columns = ['ft', 'bt', 'transmission', 'model', 'Insurance Validity', 
                           'Engine Displacement', 'Drive Type', 'City']

    mappings = {}
    for column in categorical_columns:
        unique_values = data[column].dropna().unique()
        mappings[column] = {value: index + 1 for index, value in enumerate(unique_values)}

    # Define feature categories
    categorical_features = ['ft', 'bt', 'transmission', 'model', 'Insurance Validity', 
                            'Engine Displacement', 'Drive Type', 'City']
    numerical_features = ['Mileage', 'Max Power', 'Torque', 'No of Cylinder',  
                            'Car_age']
    
    # Define the list of feature names the model expects
    model_feature_names = [
        'ft', 'bt', 'transmission', 'model', 'Insurance Validity',  
        'Engine Displacement', 'Drive Type', 'City', 'Mileage', 'Max Power', 
        'Torque', 'No of Cylinder','Car_age'
    ]

    # Process numerical features
    try:
        data['Engine Displacement'] = pd.to_numeric(data['Engine Displacement'], errors='coerce')
        data['No of Cylinder'] = pd.to_numeric(data['No of Cylinder'], errors='coerce')

        # Replace zero values with NaN for meaningful division
        data['Engine Displacement'].replace(0, np.nan, inplace=True)
        data['No of Cylinder'].replace(0, np.nan, inplace=True)

        # Calculate the new feature
        data['engine_size_per_cylinder'] = data['Engine Displacement'] / data['No of Cylinder']
        data['engine_size_per_cylinder'].fillna(0, inplace=True)

    except Exception as e:
        st.error(f"Error processing 'engine_size_per_cylinder': {e}")
        return

# Check numerical features
    missing_features = [feature for feature in numerical_features if feature not in data.columns]
    if missing_features:
        st.error(f"Missing numerical features: {', '.join(missing_features)}")
        return

    # Scale numerical features
    scalers['numerical'] = MinMaxScaler()
    scalers['numerical'].fit(data[numerical_features])


    # Sidebar inputs for Car Price Prediction
    with st.sidebar:
        st.header("Input Features")
        inputs = {}
        for feature in categorical_features:
            if feature in data.columns:
                options = sorted(data[feature].dropna().unique())
                inputs[feature] = st.selectbox(feature, options=options)

        for feature in numerical_features:
            if feature in data.columns:
                min_val, max_val = data[feature].min(), data[feature].max()
                inputs[feature] = st.slider(feature, float(min_val), float(max_val), float(min_val))

    # Add engineered inputs
    #st.markdown("## 📊 **Input Summary and Prediction**")
    # Prepare DataFrame
    input_df = pd.DataFrame([inputs])

    # Apply the mappings to the input DataFrame (without displaying them)
    input_df['ft'] = input_df['ft'].map(mappings['ft'])
    input_df['bt'] = input_df['bt'].map(mappings['bt'])
    input_df['transmission'] = input_df['transmission'].map(mappings['transmission'])
    input_df['model'] = input_df['model'].map(mappings['model'])
    input_df['Insurance Validity'] = input_df['Insurance Validity'].map(mappings['Insurance Validity'])
    #input_df['Seats'] = input_df['Seats'].map(mappings['Seats'])
    input_df['Engine Displacement'] = input_df['Engine Displacement'].map(mappings['Engine Displacement'])
    input_df['Drive Type'] = input_df['Drive Type'].map(mappings['Drive Type'])
    input_df['City'] = input_df['City'].map(mappings['City'])

    # Scale numerical features
    input_df[numerical_features] = scalers['numerical'].transform(input_df[numerical_features])

    # Align input_df with the model's expected features
    missing_features = set(model_feature_names) - set(input_df.columns)
    extra_features = set(input_df.columns) - set(model_feature_names)

    # Add missing features with default values (0 or np.nan)
    for feature in missing_features:
        input_df[feature] = 0  # Replace with an appropriate default value

    # Ensure correct feature order
    input_df = input_df[model_feature_names]

    # Prediction button
    if st.button("Predict Price"):
        try:
            prediction = model.predict(input_df)
            st.markdown(
                f"<h2 style='color:#88E788;'>Estimated Price: ₹{prediction[0]:,.2f}</h2>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Error making prediction: {e}")


if __name__ == "__main__":
    main()
