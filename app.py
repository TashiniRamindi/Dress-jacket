import streamlit as st
import joblib
import pandas as pd
import base64

# Function to load and encode the background image
def set_background_image(image_file):
    with open(image_file, "rb") as img:
        base64_str = base64.b64encode(img.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_str}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)

# Function to load and display a top image with increased size
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

def set_image_top(image_path):
    base64_str = get_base64_image(image_path)
    st.markdown(f'<img src="data:image/jpeg;base64,{base64_str}" style="display:block;margin-left:auto;margin-right:auto;width:80%;">', unsafe_allow_html=True)

# Set the background image
set_background_image("blue.jpg")  # Background image

# Set an image at the top with increased size
set_image_top("background.jpg")  # Top image file

# Load the saved models for both jacket and dress
model_dress = joblib.load("classification_model_dress.pkl")
model_jacket = joblib.load("classification_model_jacket.pkl")

columns_dress = joblib.load("dress_X_train.pkl")
columns_jacket = joblib.load("jacket_X_train.pkl")

# Function to preprocess inputs for dress data
def preprocess_input_dress(user_input):
    # One-Hot Encoding for categorical columns for dress
    dummy_cols = ['Collar', 'Neckline', 'Hemline', 'Style', 'Sleeve Style', 'Pattern', 'Product Colour', 'Material']
    input_df = pd.DataFrame([user_input], columns=user_input.keys())
    
    input_dummies = pd.get_dummies(input_df[dummy_cols], drop_first=True)
    input_df = pd.concat([input_df, input_dummies], axis=1)
    input_df = input_df.drop(columns=dummy_cols)
    
    # Ordinal Encoding for specific columns for dress
    fit_mapping = {'slim_fit': 0, 'regular_fit': 1, 'relaxed_fit': 3}
    length_mapping = {'mini': 0, 'knee': 1, 'midi': 2, 'maxi': 3}
    sleeve_length_mapping = {'sleeveless': 0, 'short_length': 1, 'elbow_length': 2, 'three_quarter_sleeve': 3, 'long_sleeve': 4}
    
    input_df['Fit'] = input_df['Fit'].map(fit_mapping)
    input_df['Length'] = input_df['Length'].map(length_mapping)
    input_df['Sleeve Length'] = input_df['Sleeve Length'].map(sleeve_length_mapping)
    
    # Add new features from radio buttons (Yes=1, No=0)
    input_df['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    input_df['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    input_df['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0
    
    # Reindex to match the columns the model was trained on
    input_df = input_df.reindex(columns=columns_dress, fill_value=0)
    
    return input_df

# Function to preprocess inputs for jacket data
def preprocess_input_jacket(user_input):
    # One-Hot Encoding for categorical columns for jacket
    dummy_cols = ['Collar', 'Neckline', 'Hemline', 'Style', 'Sleeve Style', 'Pattern', 'Product Colour', 'Material']
    input_df = pd.DataFrame([user_input], columns=user_input.keys())
    
    input_dummies = pd.get_dummies(input_df[dummy_cols], drop_first=True)
    input_df = pd.concat([input_df, input_dummies], axis=1)
    input_df = input_df.drop(columns=dummy_cols)
    
    # Ordinal Encoding for specific columns for jacket
    fit_mapping = {'regular_fit': 0, 'relaxed_fit': 1, 'slim_fit': 2, 'oversize_fit': 3}
    length_mapping = {'short': 0, 'medium': 1, 'long': 2}
    sleeve_length_mapping = {'sleeveless': 0, 'elbow_length': 1, 'long_sleeve': 2}
    
    input_df['Fit'] = input_df['Fit'].map(fit_mapping)
    input_df['Length'] = input_df['Length'].map(length_mapping)
    input_df['Sleeve Length'] = input_df['Sleeve Length'].map(sleeve_length_mapping)
    
    # Add new features from radio buttons (Yes=1, No=0)
    input_df['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    input_df['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    input_df['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0
    
    # Reindex to match the columns the model was trained on
    input_df = input_df.reindex(columns=columns_jacket, fill_value=0)
    
    return input_df

# Streamlit app interface
st.title("Cloth Season Prediction App")
st.write("Please specify whether the cloth is a Jacket or a Dress to predict the season.")

# Ask the user whether it is a jacket or dress
cloth_type = st.selectbox("Is the cloth a Jacket or a Dress?", ['Jacket', 'Dress'], index=None)

# User inputs for dress/jacket features
if cloth_type == 'Dress':
    # Dress-related inputs
    user_input = {
        'Fit': st.selectbox('Fit', ['slim_fit', 'regular_fit', 'relaxed_fit'], index=None),
        'Length': st.selectbox('Length', ['mini', 'knee', 'midi', 'maxi'], index=None),
        'Sleeve Length': st.selectbox('Sleeve Length', ['sleeveless', 'short_length', 'elbow_length', 'three_quarter_sleeve', 'long_sleeve'], index=None),
        'Collar': st.selectbox('Collar', ['shirt_collar', 'Basic', 'other_collar', 'no_collar', 'high_collar', 'polo_collar', 'Ruffled/Decorative'], index=None),
        'Neckline': st.selectbox('Neckline', ['other_neckline', 'collared_neck', 'off_shoulder', 'v_neck', 'high_neck', 'sweetheart_neck', 'crew_neck', 'square_neck'], index=None),
        'Hemline': st.selectbox('Hemline', ['curved_hem', 'straight_hem', 'other_hemline', 'asymmetrical_hem', 'flared_hem', 'ruffle_hem'], index=None),
        'Style': st.selectbox('Style', ['fit_and_flare', 'sundress', 'sweater & jersey', 'other_style', 'shirtdress & tshirt', 'babydoll', 'slip', 'a_line'], index=None),
        'Sleeve Style': st.selectbox('Sleeve Style', ['ruched', 'cuff', 'ruffle', 'bishop_sleeve', 'plain', 'other_sleeve_style', 'balloon', 'puff', 'kimono', 'no_sleeve', 'cap'], index=None),
        'Pattern': st.selectbox('Pattern', ['floral_prints', 'animal_prints', 'other', 'multicolor', 'cable_knit', 'printed', 'other_pattern', 'stripes_and_checks', 'solid_or_plain', 'polka_dot'], index=None),
        'Product Colour': st.selectbox('Product Colour', ['green', 'grey', 'pink', 'brown', 'metallics', 'blue', 'neutral', 'white', 'black', 'orange', 'purple', 'multi_color', 'red', 'yellow'], index=None),
        'Material': st.selectbox('Material', ['Other', 'Synthetic Fibers', 'Wool', 'Silk', 'Luxury Materials', 'Cotton', 'Metallic', 'Knitted and Jersey Materials', 'Leather', 'Polyester'], index=None),
        
        # Radio buttons for additional features
        'Breathable': st.radio('Is the dress breathable?', ('Yes', 'No'), index=None),
        'Lightweight': st.radio('Is the dress lightweight?', ('Yes', 'No'), index=None),
        'Water_Repellent': st.radio('Is the dress water repellent?', ('Yes', 'No'), index=None)
    }

    # Preprocess the inputs and predict
    input_data = preprocess_input_dress(user_input)
    prediction = model_dress.predict(input_data)
    st.write(f"The predicted season for the dress is: {


# Mapping for seasons
season_mapping = {0: 'spring', 1: 'summer', 2: 'winter', 3: 'autumn'}

# When the user presses the 'Predict' button
if st.button("Predict"):
    if cloth_type == 'Dress':
        preprocessed_input = preprocess_input_dress(user_input)
        prediction = model_dress.predict(preprocessed_input)
        # Map the numeric prediction to season name
        predicted_season = season_mapping[prediction[0]]
        st.write("The predicted season for this dress is:", predicted_season)

    elif cloth_type == 'Jacket':
        preprocessed_input = preprocess_input_jacket(user_input)
        prediction = model_jacket.predict(preprocessed_input)
        # Map the numeric prediction to season name
        predicted_season = season_mapping[prediction[0]]
        st.write("The predicted season for this jacket is:", predicted_season)

