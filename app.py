a
    input_df = input_df.drop(columns=dummy_cols)
    
    # Ordinal Encoding for specific columns for dress
    fit_mapping = {'slim_fit': 0, 'regular_fit': 1, 'relaxed_fit': 3}
    length_mapping = {'mini': 0, 'knee': 1, 'midi': 2, 'maxi': 3}
    sleeve_length_mapping = {'sleeveless': 0, 'short_length': 1, 'elbow_length': 2, 'three_quarter_sleeve': 3, 'long_sleeve': 4}
    
    input_df['Fit'] = input_df['Fit'].map(fit_mapping)
    input_df['Length'] = input_df['Length'].map(length_mapping)
    input_df['Sleeve Length'] = input_df['Sleeve Length'].map(sleeve_length_mapping)
    
    # Add the new features from radio buttons (Yes=1, No=0)
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
    
    # Add the new features from radio buttons (Yes=1, No=0)
    input_df['Breathable'] = 1 if user_input['Breathable'] == 'Yes' else 0
    input_df['Lightweight'] = 1 if user_input['Lightweight'] == 'Yes' else 0
    input_df['Water_Repellent'] = 1 if user_input['Water_Repellent'] == 'Yes' else 0
    
    # Reindex to match the columns the model was trained on
    input_df = input_df.reindex(columns=columns_jacket, fill_value=0)
    
    return input_df

# Streamlit app interface
st.title("Season Prediction App", anchor="top")
st.markdown("<h4 style='text-align: center;'>Predict the Season for a Dress or Jacket</h4>", unsafe_allow_html=True)
st.write("This app predicts the most likely season (Spring, Summer, Autumn, Winter) based on the type of clothing you provide.")

# Ask the user whether it is a jacket or dress
cloth_type = st.selectbox("Is the cloth a Jacket or a Dress?", ['Jacket', 'Dress'],index=None)

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
        'Collar': st.selectbox('What type of collar does the dress have?', ['shirt_collar', 'Basic', 'no_collar', 'high_collar', 'polo_collar', 'Ruffled/Decorative','other_collar'],index=None),
        'Neckline': st.selectbox('What type of neckline does the dress have?', ['collared_neck', 'off_shoulder', 'v_neck', 'high_neck', 'sweetheart_neck', 'crew_neck', 'square_neck', 'other_neckline'],index=None),
        'Hemline': st.selectbox('What type of hemline does the dress have?', ['curved_hem', 'straight_hem', 'asymmetrical_hem', 'flared_hem', 'ruffle_hem','other_hemline'],index=None),
        'Style': st.selectbox('What style is the dress?', ['fit_and_flare', 'sundress', 'sweater & jersey', 'shirtdress & tshirt', 'babydoll', 'slip', 'a_line','other_style'],index=None),
        'Fit': st.selectbox('What is the fit of the dress?', ['relaxed_fit', 'slim_fit', 'regular_fit'],index=None),
        'Length': st.selectbox('What is the length of the dress?', ['mini', 'midi', 'maxi', 'knee'],index=None),
        'Sleeve Length': st.selectbox('What sleeve length does the dress have?', ['long_sleeve', 'three_quarter_sleeve', 'short_length', 'elbow_length', 'sleeveless'],index=None),
        'Sleeve Style': st.selectbox('What sleeve style does the dress have?', ['ruched', 'cuff', 'ruffle', 'bishop_sleeve', 'plain', 'balloon', 'puff', 'kimono', 'no_sleeve', 'cap', 'other_sleeve_style'],index=None),
        'Pattern': st.selectbox('What pattern does the dress have?', ['floral_prints', 'animal_prints','multicolor', 'cable_knit', 'printed','stripes_and_checks', 'solid_or_plain', 'polka_dot','other_pattern'],index=None),
        'Material': st.selectbox('What material is the dress made from?', ['Synthetic Fibers', 'Wool', 'Silk', 'Luxury Materials', 'Cotton', 'Metallic', 'Knitted and Jersey Materials', 'Leather', 'Polyester','Other'],index=None),
        'Product Colour': st.selectbox('What color is the dress?', ['green', 'grey', 'pink', 'brown', 'metallics', 'blue', 'neutral', 'white', 'black', 'orange', 'purple', 'multi_color', 'red', 'yellow'],index=None),
        'Breathable': st.radio("Is the dress breathable?", ["Yes", "No"],index=None),
        'Lightweight': st.radio("Is the dress lightweight?", ["Yes", "No"],index=None),
        'Water_Repellent': st.radio("Is the dress water repellent?", ["Yes", "No"],index=None),
    }


    
elif cloth_type == 'Jacket':
    user_input = {
        'Outerwear Type': st.selectbox('What type of outerwear is this?', ['jacket', 'vest', 'coat'], index=None),
        'Fit': st.selectbox('What is the fit of the jacket?', ['regular_fit', 'relaxed_fit', 'slim_fit', 'oversize_fit'], index=None),
        'Length': st.selectbox('What is the length of the jacket?', ['short', 'medium', 'long'], index=None),
        'Sleeve Length': st.selectbox('What sleeve length does the jacket have?', ['long_sleeve', 'sleeveless', 'elbow_length'], index=None),
        'Collar': st.selectbox('What type of collar does the jacket have?', ['point', 'no collar', 'band', 'notch', 'lapel','other_collar'], index=None),
        'Neckline': st.selectbox('What type of neckline does the jacket have?', ['collared_neck', 'hooded', 'funnel_neck', 'v_neck', 'other_neck'], index=None),
        'Hemline': st.selectbox('What type of hemline does the jacket have?', ['ribbed_hem', 'straight_hem', 'other_hem'], index=None),
        'Style': st.selectbox('What style is the jacket?', ['bomber', 'gilet', 'trucker', 'windbreaker', 'soft_shell', 'sweatshirt', 'puffer', 'other_style', 'harrington', 'rain_jacket', 'parka', 'cargo', 'shirt', 'trench', 'blazer', 'cocoon', 'anorak', 'overcoat', 'peacoat', 'hardshell', 'barn','other_style'], index=None),
        'Sleeve Style': st.selectbox('What sleeve style does the jacket have?', ['cuff_sleeve', 'no_sleeve', 'plain_sleeve', 'raglan_sleeve','other_sleeve_style'], index=None),
        'Pattern': st.selectbox('What pattern does the jacket have?', ['solid_or_plain', 'multicolor', 'printed','plaid', 'cable_knit', 'tie_dry', 'houndstooth', 'chevron','other'], index=None),
        'Product Colour': st.selectbox('What color is the jacket?', ['black', 'grey', 'blue', 'red', 'white', 'brown', 'yellow', 'pink', 'green', 'cream', 'beige', 'purple', 'orange', 'multi_color'], index=None),
        'Material': st.selectbox('What material is the jacket made from?', ['Polyamide', 'Cotton', 'Polyester', 'Nylon',  'fleece', 'Wool', 'denim', 'leather', 'faux_fur', 'corduroy', 'rib_knit', 'Other material'], index=None),
        'Breathable': st.radio("Is the jacket breathable?", ["Yes", "No"], index=None),
        'Lightweight': st.radio("Is the jacket lightweight?", ["Yes", "No"], index=None),
        'Water Repellent': st.radio("Is the jacket water repellent?", ["Yes", "No"], index=None),
    }



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
