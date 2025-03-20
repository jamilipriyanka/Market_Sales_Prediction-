import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import _pickle
import hashlib
import time
from xgboost import XGBRegressor
from datetime import datetime
import os

# Set page configuration
st.set_page_config(page_title="BigMart Sales Prediction",
                   page_icon="📊",
                   layout="wide",
                   initial_sidebar_state="collapsed")

# Print a message to verify Streamlit is running
print("Streamlit app is starting...")


# Function for password hashing
def make_hash(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


# Function to check hashed password
def check_password(password, hashed_password):
    return make_hash(password) == hashed_password


# Try to load the model
try:
    with open('model3.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    print("Model loaded successfully")
except (FileNotFoundError, _pickle.UnpicklingError) as e:
    print(f"Model file issue: {e}, creating new model...")
    # Create a new model for demonstration with exact same features as prediction function
    model = XGBRegressor()
    # Dummy training data with 9 features to match prediction function
    X = np.random.rand(100, 9) 
    # Generate dummy target values based on features
    y = 2000 + 10 * X[:, 0] + 200 * X[:, 1] + 5 * X[:, 2] - 100 * X[:, 3] + 50 * X[:, 4] + 25 * X[:, 5] + 75 * X[:, 6] + 15 * X[:, 7] + 40 * X[:, 8]
    model.fit(X, y)
    # Save the model
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    print("Created a new model with 9 features to match prediction function")

# Initialize session state for user authentication
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'users' not in st.session_state:
    # Initialize with a default admin account
    st.session_state.users = {
        'admin@bigmart.com': {
            'password': make_hash('admin123'),
            'name': 'Administrator',
            'role': 'admin',
            'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

# Initialize step to track form progress
if 'step' not in st.session_state:
    st.session_state.step = 1

# Initialize form data
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}

# Initialize current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Prediction"

# Initialize prediction result
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None

# Custom Styling with black and pink theme
st.markdown("""
    <style>
    /* Global Variables for Black and Pink Theme with Clean Black Glass */
    :root {
        --primary: #FF1493;       /* Hot Pink */
        --secondary: #FF69B4;     /* Pink */
        --background: #000000;    /* Black background */
        --text-light: #FFFFFF;    /* White text */
        --text-dark: #121212;     /* Dark text */
        --glass-bg: rgba(0, 0, 0, 0.6); /* Black glass background */
        --glass-border: rgba(40, 40, 40, 0.8); /* Dark gray border */
        --shadow-color: rgba(255, 20, 147, 0.5); /* Pink Shadow for highlights only */
        --input-glass-bg: rgba(10, 10, 10, 0.9); /* Clean black glass for inputs */
        --input-glass-border: rgba(30, 30, 30, 1); /* Solid dark border for inputs */
        --input-glass-hover: rgba(20, 20, 20, 1); /* Slightly lighter black for hover states */
    }

    /* Page Background - Black */
    .stApp {
        background-color: var(--background);
    }

    /* Main Title with Frosted Glass Effect */
    .main-title {
        font-size: 3em;
        font-weight: 800;
        text-align: center;
        color: var(--text-light);
        margin: 1.5rem auto;
        text-shadow: 0px 0px 8px var(--shadow-color);
        background: var(--glass-bg);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0px 5px 15px var(--shadow-color);
        border: 1px solid var(--glass-border);
    }

    /* Form Container - Enhanced Glass Morphism */
    

    .section-header {
        color: var(--text-light);
        font-size: 1.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 0px 0px 5px var(--shadow-color);
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: var(--background) !important;
        border-right: 2px solid var(--glass-border) !important;
        box-shadow: 4px 0px 10px var(--shadow-color) !important;
    }

    /* Sidebar Text Color */
    [data-testid="stSidebar"] * {
        color: var(--text-light) !important;
        font-weight: 500 !important;
    }

    /* Buttons - Pink Glass Effect */
    .stButton > button {
        background: var(--primary);
        color: var(--text-light);
        font-weight: bold;
        font-size: 1rem;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        border: 2px solid var(--glass-border);
        transition: all 0.3s ease;
        text-shadow: 0px 0px 6px rgba(0, 0, 0, 0.4);
        box-shadow: 0px 0px 10px var(--shadow-color);
        margin-top: 20px !important;
        margin-bottom: 20px !important;
    }

    .stButton > button:hover {
        background: var(--secondary);
        transform: scale(1.05);
        box-shadow: 0px 0px 15px var(--shadow-color);
    }

    .stButton > button:active {
        transform: scale(1);
    }

    /* Streamlit-specific overrides to target the right elements */
    /* Outer containers */
    div.stTextInput, div.stNumberInput, div.stSelectbox {
        margin-bottom: 15px !important;
    }
    
    /* Base input wrapper */
    div[data-baseweb] {
        background: transparent !important;
        border: none !important;
    }
    
    /* Direct input elements */
    input, select, textarea {
        background: #111111 !important;
        color: white !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        padding: 10px 15px !important;
        box-shadow: none !important;
        margin: 0 !important;
    }
    
    /* Select box overrides */
    div[data-baseweb="select"] div[data-testid="stMarkdown"] {
        display: none !important;  /* Hide duplicates */
    }
    
    div[data-baseweb="select"] div {
        background-color: #111111 !important;
        border-color: #333333 !important;
    }
    
    /* Input container */
    div[data-baseweb="base-input"] {
        background-color: #111111 !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="base-input"] > div {
        background: transparent !important;
        border: none !important;
    }
    
    div[data-baseweb="base-input"] input {
        border: none !important;
        background: transparent !important;
    }

    /* Hover and Focus for specific elements */
    div[data-baseweb="base-input"]:hover {
        border-color: #444444 !important;
    }
    
    div[data-baseweb="base-input"]:focus-within {
        border-color: var(--primary) !important;
    }

    /* Form field containers - Clean Black Style - Container Only */
    .stTextInput, .stNumberInput, .stSelectbox, .stSlider, .stRadio {
        margin-bottom: 15px !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
        box-shadow: none !important;
    }

    /* Placeholder Text Styling */
    ::placeholder {
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 500;
    }

    /* Adjustments for Textarea */
    textarea {
        min-height: 100px !important;
        resize: none !important;
    }

    /* Results Box - Enhanced Frosted Glass with Pink Glow */
    

    .result-header {
        color: var(--text-light);
        font-size: 1.5rem;
        font-weight: 700;
        text-shadow: 0px 0px 6px var(--shadow-color);
    }

    .result-value {
        font-size: 3rem;
        font-weight: 800;
        color: var(--primary);
        text-shadow: 0px 0px 12px var(--shadow-color);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--text-light);
        font-size: 0.9rem;
        padding: 1rem;
        border-radius: 10px;
        background: var(--glass-bg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        width: 100%;
        margin-top: 100px;
        border: 1px solid var(--glass-border);
    }

    /* Step Indicator */
    .step-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem auto;
    }

    .step {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: var(--glass-bg);
        display: flex;
        justify-content: center;
        align-items: center;
        font-weight: bold;
        color: var(--text-light);
        border: 2px solid var(--glass-border);
    }

    .step.active {
        background: var(--primary);
        box-shadow: 0px 0px 15px var(--shadow-color);
    }

    .step-line {
        height: 5px;
        width: 100px;
        background-color: var(--glass-border);
        margin: 0 10px;
    }

    /* User Profile Card */
    .user-profile {
        padding: 20px;
        background: var(--glass-bg);
        border-radius: 10px;
        text-align: center;
        border: 1px solid var(--glass-border);
        margin-bottom: 20px;
    }

    /* Nav Item */
    .nav-item {
        padding: 10px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .nav-item:hover, .nav-item.active {
        background: var(--glass-bg);
        box-shadow: 0px 0px 8px var(--shadow-color);
    }

    /* Header Navigation */
    .header-nav {
        display: flex;
        justify-content: space-around;
        padding: 10px;
        background: var(--glass-bg);
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid var(--glass-border);
    }

    .header-nav-item {
        padding: 8px 15px;
        border-radius: 8px;
        cursor: pointer;
        color: var(--text-light);
        font-weight: bold;
        transition: all 0.3s ease;
    }

    .header-nav-item:hover, .header-nav-item.active {
        background: var(--primary);
        box-shadow: 0px 0px 8px var(--shadow-color);
    }

    /* Text & Labels */
    label, p, h1, h2, h3, h4, .stMarkdown {
        color: var(--text-light) !important;
    }

    /* Form fields minimal styling */
    .stNumberInput > div > div > input, .stTextInput > div > div > input {
        /* Already handled by the direct input elements styling above */
    }

    /* Slider */
    .stSlider > div > div > div > div {
        background-color: var(--primary) !important;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
            padding: 1rem;
        }

        .form-container {
            padding: 1.5rem;
        }

        .result-value {
            font-size: 2rem;
        }
        
        .step-line {
            width: 50px;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Logo SVG in pink
logo_svg = """
<svg xmlns="http://www.w3.org/2000/svg" width="150" height="80" viewBox="0 0 200 100">
  <style>
    .text { font: bold 40px sans-serif; fill: #FF1493; }
    .subtitle { font: 18px sans-serif; fill: #FF69B4; }
    .icon { fill: #FF1493; }
  </style>
  <rect class="icon" x="10" y="30" width="40" height="40" rx="5" />
  <rect class="icon" x="55" y="30" width="20" height="40" rx="5" />
  <rect class="icon" x="80" y="50" width="20" height="20" rx="5" />
  <text x="110" y="50" class="text">Mart</text>
  <text x="110" y="70" class="subtitle">Sales Prediction</text>
</svg>
"""

# Function to make predictions
def predict_sales(item_weight, item_fat_content, item_visibility, item_type, 
                  item_mrp, outlet_establishment_year, outlet_size, 
                  outlet_location_type, outlet_type):
    
    # Prepare the input features
    # Encoding categorical variables
    fat_content_mapping = {
        'Low Fat': 0, 
        'Regular': 1
    }
    
    item_type_mapping = {
        'Dairy': 0, 
        'Soft Drinks': 1, 
        'Meat': 2, 
        'Fruits and Vegetables': 3,
        'Household': 4, 
        'Baking Goods': 5, 
        'Snack Foods': 6, 
        'Frozen Foods': 7,
        'Breakfast': 8, 
        'Health and Hygiene': 9, 
        'Hard Drinks': 10, 
        'Canned': 11,
        'Breads': 12, 
        'Starchy Foods': 13, 
        'Others': 14, 
        'Seafood': 15
    }
    
    outlet_size_mapping = {
        'Small': 0, 
        'Medium': 1, 
        'High': 2
    }
    
    outlet_location_mapping = {
        'Tier 1': 0, 
        'Tier 2': 1, 
        'Tier 3': 2
    }
    
    outlet_type_mapping = {
        'Grocery Store': 0, 
        'Supermarket Type1': 1, 
        'Supermarket Type2': 2, 
        'Supermarket Type3': 3
    }
    
    # Creating features array - all features needed for prediction
    features = np.array([
        item_weight,
        fat_content_mapping.get(item_fat_content, 0),
        item_visibility,
        item_type_mapping.get(item_type, 0),
        item_mrp,
        2023 - outlet_establishment_year,  # Convert year to outlet age
        outlet_size_mapping.get(outlet_size, 0),
        outlet_location_mapping.get(outlet_location_type, 0),
        outlet_type_mapping.get(outlet_type, 0)
    ]).reshape(1, -1)
    
    # Make prediction
    try:
        prediction = model.predict(features)[0]
        return round(prediction, 2)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Authentication/Registration screen when not logged in
if not st.session_state.user_authenticated:
    st.markdown('<h1 class="main-title">BigMart Sales Prediction</h1>',
                unsafe_allow_html=True)

    # Create tabs for login and register
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Sign In</h2>',
                    unsafe_allow_html=True)

        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password",
                                 type="password",
                                 key="login_password")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Login", key="login_btn"):
                if email in st.session_state.users and check_password(
                        password, st.session_state.users[email]['password']):
                    st.session_state.user_authenticated = True
                    st.session_state.current_user = email
                    st.session_state.users[email]['last_login'] = datetime.now(
                    ).strftime("%Y-%m-%d %H:%M:%S")
                    st.rerun()
                else:
                    st.error("Invalid email or password")

        st.markdown("</div>", unsafe_allow_html=True)

    with register_tab:
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Create an Account</h2>',
                    unsafe_allow_html=True)

        name = st.text_input("Full Name")
        new_email = st.text_input("Email", key="register_email")
        new_password = st.text_input("Password",
                                     type="password",
                                     key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Register", key="register_btn"):
                if not name or not new_email or not new_password:
                    st.error("Please fill in all fields")
                elif new_email in st.session_state.users:
                    st.error("Email already registered")
                elif new_password != confirm_password:
                    st.error("Passwords don't match")
                else:
                    st.session_state.users[new_email] = {
                        'password': make_hash(new_password),
                        'name': name,
                        'role': 'user',
                        'last_login': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.success("Account created successfully! Please login.")

        st.markdown("</div>", unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="footer">© 2023 BigMart Sales Prediction System. All rights reserved.</div>',
        unsafe_allow_html=True)

else:
    # Main Application after login
    st.markdown(f'<div style="text-align: center; margin-bottom: 20px;">{logo_svg}</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">BigMart Sales Prediction</h1>', unsafe_allow_html=True)

    # User profile section in sidebar
    with st.sidebar:
        st.markdown(
            f'<div class="user-profile"><h3>Welcome, {st.session_state.users[st.session_state.current_user]["name"]}</h3><p>Last Login: {st.session_state.users[st.session_state.current_user]["last_login"]}</p></div>',
            unsafe_allow_html=True)

        # Navigation
        st.markdown('<h3>Navigation</h3>', unsafe_allow_html=True)
        pages = ["Prediction", "Dashboard", "History", "Settings"]
        for page in pages:
            if st.button(page, key=f"nav_{page}"):
                st.session_state.current_page = page
                st.rerun()

        # Logout button
        if st.button("Logout"):
            st.session_state.user_authenticated = False
            st.session_state.current_user = None
            st.rerun()

    # Main content based on selected page
    if st.session_state.current_page == "Prediction":
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Item and Outlet Information</h2>', unsafe_allow_html=True)

        # Create a multi-step form
        # Step indicator
        st.markdown(
            '<div class="step-container"><div class="step active">1</div><div class="step-line"></div><div class="step">2</div></div>',
            unsafe_allow_html=True)

        # Step 1: Item Information
        if st.session_state.step == 1:
            # Create a 2-column layout for better organization
            col1, col2 = st.columns(2)
            
            with col1:
                item_weight = st.number_input("Item Weight (in grams)", 
                                              min_value=0.0, 
                                              max_value=50.0, 
                                              value=10.0, 
                                              step=0.1,
                                              help="Weight of the product in grams")
                
                item_fat_content = st.selectbox("Item Fat Content", 
                                               ["Low Fat", "Regular"],
                                               help="Fat content category of the product")
                
                item_visibility = st.slider("Item Visibility", 
                                           min_value=0.0, 
                                           max_value=0.3, 
                                           value=0.1, 
                                           step=0.01,
                                           help="Percentage of total display area allocated to this product")
                
                item_type = st.selectbox("Item Type", 
                                        ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables",
                                         "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                                         "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                                         "Breads", "Starchy Foods", "Others", "Seafood"],
                                        help="Category of the product")
            
            with col2:
                item_mrp = st.number_input("Item MRP (₹)", 
                                          min_value=10.0, 
                                          max_value=300.0, 
                                          value=100.0, 
                                          step=1.0,
                                          help="Maximum Retail Price of the product")
                
                # Store the input values in session state
                if st.button("Next", key="next_btn"):
                    st.session_state.form_data.update({
                        "item_weight": item_weight,
                        "item_fat_content": item_fat_content,
                        "item_visibility": item_visibility,
                        "item_type": item_type,
                        "item_mrp": item_mrp
                    })
                    st.session_state.step = 2
                    st.rerun()

        # Step 2: Outlet Information
        elif st.session_state.step == 2:
            # Create columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                outlet_establishment_year = st.number_input("Outlet Establishment Year", 
                                                          min_value=1900, 
                                                          max_value=2023, 
                                                          value=2000, 
                                                          step=1,
                                                          help="Year the outlet was established")
                
                outlet_size = st.selectbox("Outlet Size", 
                                          ["Small", "Medium", "High"],
                                          help="Size of the outlet")
            
            with col2:
                outlet_location_type = st.selectbox("Outlet Location Type", 
                                                  ["Tier 1", "Tier 2", "Tier 3"],
                                                  help="Type of city where the outlet is located")
                
                outlet_type = st.selectbox("Outlet Type", 
                                          ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"],
                                          help="Type of outlet")
            
            # Back and Predict buttons in one row
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Back", key="back_btn"):
                    st.session_state.step = 1
                    st.rerun()
            
            with col2:
                if st.button("Predict", key="predict_btn"):
                    # Get the stored values from step 1
                    item_data = st.session_state.form_data
                    
                    # Update with step 2 values
                    outlet_data = {
                        "outlet_establishment_year": outlet_establishment_year,
                        "outlet_size": outlet_size,
                        "outlet_location_type": outlet_location_type,
                        "outlet_type": outlet_type
                    }
                    
                    # Combine all data for prediction
                    prediction_input = {**item_data, **outlet_data}
                    
                    # Make prediction
                    with st.spinner("Calculating sales prediction..."):
                        result = predict_sales(
                            prediction_input["item_weight"],
                            prediction_input["item_fat_content"],
                            prediction_input["item_visibility"],
                            prediction_input["item_type"],
                            prediction_input["item_mrp"],
                            prediction_input["outlet_establishment_year"],
                            prediction_input["outlet_size"],
                            prediction_input["outlet_location_type"],
                            prediction_input["outlet_type"]
                        )
                    
                    if result is not None:
                        # Store the result in session state
                        st.session_state.prediction_result = result
                        
                        # Show the result in a nice glass morphism container
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.markdown('<h3 class="result-header">Predicted Item Sales</h3>', unsafe_allow_html=True)
                        st.markdown(f'<p class="result-value">₹ {result}</p>', unsafe_allow_html=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Option to make another prediction
                        if st.button("Make Another Prediction"):
                            st.session_state.step = 1
                            st.session_state.prediction_result = None
                            st.rerun()
                    else:
                        st.error("An error occurred during prediction. Please try again.")

        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.current_page == "Dashboard":
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Sales Dashboard</h2>', unsafe_allow_html=True)
        
        # Sample visualization
        st.write("Coming soon: Sales analytics and visualizations")
        
        # Create a sample chart
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household']
        values = [2500, 1800, 3200, 2100, 2700]
        
        ax.bar(categories, values, color='#FF1493')
        ax.set_title('Sample Sales by Category', color='white')
        ax.set_xlabel('Product Category', color='white')
        ax.set_ylabel('Sales Amount (₹)', color='white')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['right'].set_color('white')
        
        # Set the figure background to transparent
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        st.pyplot(fig)
        
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.current_page == "History":
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Prediction History</h2>', unsafe_allow_html=True)
        
        st.write("Your previous predictions will appear here.")
        
        # Sample history data
        if st.session_state.prediction_result is not None:
            st.markdown('<div class="result-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="result-header">Latest Prediction</h3>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-value">₹ {st.session_state.prediction_result}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No prediction history available. Make a prediction to see it here.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    elif st.session_state.current_page == "Settings":
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-header">Account Settings</h2>', unsafe_allow_html=True)
        
        # User account information
        user_data = st.session_state.users[st.session_state.current_user]
        
        st.write(f"Name: {user_data['name']}")
        st.write(f"Email: {st.session_state.current_user}")
        st.write(f"Role: {user_data['role']}")
        st.write(f"Last Login: {user_data['last_login']}")
        
        # Change password section
        st.markdown('<h3>Change Password</h3>', unsafe_allow_html=True)
        
        current_password = st.text_input("Current Password", type="password", key="current_pass")
        new_password = st.text_input("New Password", type="password", key="new_pass")
        confirm_new_password = st.text_input("Confirm New Password", type="password", key="confirm_new_pass")
        
        if st.button("Update Password"):
            if not check_password(current_password, user_data['password']):
                st.error("Current password is incorrect")
            elif new_password != confirm_new_password:
                st.error("New passwords don't match")
            elif not new_password:
                st.error("New password cannot be empty")
            else:
                # Update password
                st.session_state.users[st.session_state.current_user]['password'] = make_hash(new_password)
                st.success("Password updated successfully")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="footer">© 2023 BigMart Sales Prediction System. All rights reserved.</div>',
        unsafe_allow_html=True)
