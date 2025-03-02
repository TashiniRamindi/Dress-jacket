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
set_background_image("blue.jpg")  # This will be your background image

# Set an image at the top with increased size
set_image_top("background.jpg")  # The top image file, if you have one

# Load the saved models for both jacket and dress
model_dress = joblib.load("classification_model_dress.pkl")
model_jacket = joblib.load("classification_model_jacket.pkl")

columns_dress = joblib.load("dress_X_train.pkl")
columns_jacket = joblib.load("jacket_X_train.pkl")

# Function to preprocess inputs for dress data
def preprocess_input_dress(user_input):
    dummy_cols = ['Collar', 'Neckline', 'Hemline', 'Style', 'Sleeve Style', 'Pattern', 'Product Colour', 'Material']
    input_df = pd.DataFrame([user_input], columns=user_input.keys())
    
    input_dummies = pd.get_dummies(input_df[dummy_cols], drop_first=True)
    input_df = pd.concat([input_df, input_dummies], axis=1)
    input_df = input_df.drop(columns=dummy_cols)
    
    fit_mapping = {'slim_fit': 0, 'regular_fit': 1, 'relaxed_fit': 3}
    length_mapping = {'mini': 0, 'knee': 1, 'midi': 2, 'maxi': 3}
    sleeve_length_mapping = {'sleeveless': 0, 'short_length': 1, 'elbow_length': 2, 'three_quarter_sleeve': 3, 'long_sleeve': 4}
    
    input_df['Fit'] = input_df['Fit'].map(fit_mapping)
    input_df['Length'] = input_df['Length'].map(length_mapping)
    input_df['Sleeve Length'] = input_df['Sleeve Length'].map(sleeve_length_mapping)
    
    input_df['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    input_df['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    input_df['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0
    
    input_df = input_df.reindex(columns=columns_dress, fill_value=0)
    
    return input_df

# Function to preprocess inputs for jacket data
def preprocess_input_jacket(user_input):
    dummy_cols = ['Collar', 'Neckline', 'Hemline', 'Style', 'Sleeve Style', 'Pattern', 'Product Colour', 'Material']
    input_df = pd.DataFrame([user_input], columns=user_input.keys())
    
    input_dummies = pd.get_dummies(input_df[dummy_cols], drop_first=True)
    input_df = pd.concat([input_df, input_dummies], axis=1)
    input_df = input_df.drop(columns=dummy_cols)
    
    fit_mapping = {'regular_fit': 0, 'relaxed_fit': 1, 'slim_fit': 2, 'oversize_fit': 3}
    length_mapping = {'short': 0, 'medium': 1, 'long': 2}
    sleeve_length_mapping = {'sleeveless': 0, 'elbow_length': 1, 'long_sleeve': 2}
    
    input_df['Fit'] = input_df['Fit'].map(fit_mapping)
    input_df['Length'] = input_df['Length'].map(length_mapping)
    input_df['Sleeve Length'] = input_df['Sleeve Length'].map(sleeve_length_mapping)
    
    input_df['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    input_df['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    input_df['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0
    
    input_df = input_df.reindex(columns=columns_jacket, fill_value=0)
    
    return input_df

# Streamlit app interface
st.title("Clothing Season Prediction App", anchor="top")
st.markdown("<h4 style='text-align: center;'>Predict the Season for a Dress or Jacket</h4>", unsafe_allow_html=True)
st.write("This app predicts the most likely season (Spring, Summer, Autumn, Winter) based on the type of clothing you provide.")

# Ask the user whether it is a jacket or dress
cloth_type = st.selectbox("Is the clothing item a Jacket or a Dress?", ['Jacket', 'Dress'])

# Initialize the session state for user inputs if not already set
if 'user_input' not in st.session_state:
    st.session_state.user_input = {
        'Fit': None,
        'Length': None,
        'Sleeve Length': None,
        'Collar': None,
        'Neckline': None,
        'Hemline': None,
        'Style': None,
        'Sleeve Style': None,
        'Pattern': None,
        'Product Colour': None,
        'Material': None,
        'Breathable': None,
        'Lightweight': None,
        'Water_Repellent': None,
    }

# User inputs for dress/jacket features
if cloth_type == 'Dress':
    user_input = {
        'Collar': st.selectbox('What type of collar does the dress have?', ['shirt_collar', 'Basic', 'other_collar', 'no_collar', 'high_collar', 'polo_collar', 'Ruffled/Decorative']),
        'Neckline': st.selectbox('What type of neckline does the dress have?', ['other_neckline', 'collared_neck', 'off_shoulder', 'v_neck', 'high_neck', 'sweetheart_neck', 'crew_neck', 'square_neck']),
        'Hemline': st.selectbox('What type of hemline does the dress have?', ['curved_hem', 'straight_hem', 'other_hemline', 'asymmetrical_hem', 'flared_hem', 'ruffle_hem']),
        'Style': st.selectbox('What style is the dress?', ['fit_and_flare', 'sundress', 'sweater & jersey', 'other_style', 'shirtdress & tshirt', 'babydoll', 'slip', 'a_line']),
        'Fit': st.selectbox('What is the fit of the dress?', ['relaxed_fit', 'slim_fit', 'regular_fit']),
        'Length': st.selectbox('What is the length of the dress?', ['mini', 'midi', 'maxi', 'knee']),
        'Sleeve Length': st.selectbox('What sleeve length does the dress have?', ['long_sleeve', 'three_quarter_sleeve', 'short_length', 'elbow_length', 'sleeveless']),
        'Sleeve Style': st.selectbox('What sleeve style does the dress have?', ['ruched', 'cuff', 'ruffle', 'bishop_sleeve', 'plain', 'other_sleeve_style', 'balloon', 'puff', 'kimono', 'no_sleeve', 'cap']),
        'Pattern': st.selectbox('What pattern does the dress have?', ['floral_prints', 'animal_prints', 'other', 'multicolor', 'cable_knit', 'printed', 'other_pattern', 'stripes_and_checks', 'solid_or_plain', 'polka_dot']),
        'Material': st.selectbox('What material is the dress made from?', ['Other', 'Synthetic Fibers', 'Wool', 'Silk', 'Luxury Materials', 'Cotton', 'Metallic', 'Knitted and Jersey Materials', 'Leather', 'Polyester']),
        'Product Colour': st.selectbox('What color is the dress?', ['green', 'grey', 'pink', 'brown', 'metallics', 'blue', 'neutral', 'white', 'black', 'orange', 'purple', 'multi_color', 'red', 'yellow']),
        'Breathable': st.radio("Is the dress breathable?", ["Yes", "No"]),
        'Lightweight': st.radio("Is the dress lightweight?", ["Yes", "No"]),
        'Water_Repellent': st.radio("Is the dress water repellent?", ["Yes", "No"]),
    }
elif cloth_type == 'Jacket':
    user_input = {
        'Collar': st.selectbox('What type of collar does the jacket have?', ['shirt_collar', 'Basic', 'no_collar', 'high_collar', 'polo_collar', 'Ruffled/Decorative']),
        'Neckline': st.selectbox('What type of neckline does the jacket have?', ['other_neckline', 'collared_neck', 'off_shoulder', 'v_neck', 'high_neck', 'sweetheart_neck', 'crew_neck', 'square_neck']),
        'Hemline': st.selectbox('What type of hemline does the jacket have?', ['curved_hem', 'straight_hem', 'asymmetrical_hem', 'flared_hem']),
        'Style': st.selectbox('What style is the jacket?', ['fit_and_flare', 'sundress', 'oversized', 'sweater', 'other_style']),
        'Fit': st.selectbox('What is the fit of the jacket?', ['regular_fit', 'slim_fit', 'relaxed_fit', 'oversize_fit']),
        'Length': st.selectbox('What is the length of the jacket?', ['short', 'medium', 'long']),
        'Sleeve Length': st.selectbox('What sleeve length does the jacket have?', ['long_sleeve', 'sleeveless', 'elbow_length']),
        'Material': st.selectbox('What material is the jacket made from?', ['wool', 'polyester', 'denim', 'synthetic_fibers', 'leather', 'nylon']),
        'Breathable': st.radio("Is the jacket breathable?", ["Yes", "No"]),
        'Lightweight': st.radio("Is the jacket lightweight?", ["Yes", "No"]),
        'Water_Repellent': st.radio("Is the jacket water repellent?", ["Yes", "No"]),
    }

# Button to predict season
if st.button('Predict Season'):
    if cloth_type == 'Dress':
        preprocessed_input = preprocess_input_dress(user_input)
        prediction = model_dress.predict(preprocessed_input)
    elif cloth_type == 'Jacket':
        preprocessed_input = preprocess_input_jacket(user_input)
        prediction = model_jacket.predict(preprocessed_input)

    season_mapping = {0: 'Spring', 1: 'Summer', 2: 'Autumn', 3: 'Winter'}
    st.write(f"The predicted season for the {cloth_type} is: {season_mapping[prediction[0]]}")
