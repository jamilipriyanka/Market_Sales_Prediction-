import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
import pickle
import os

def train_model():
    """
    Train an XGBoost regression model on the BigMart Sales data
    and save it to disk as model.pkl
    
    Note: This function assumes that the dataset is available.
    For real implementation, ensure the dataset exists or download it.
    """
    # Create dummy data if the real dataset is not available
    # In a real application, you would load data from a file or database
    np.random.seed(42)
    
    # Generate synthetic data for demonstration
    n_samples = 1000
    
    # Features
    item_weights = np.random.uniform(4.0, 21.0, n_samples)
    item_visibilities = np.random.uniform(0.0, 0.3, n_samples)
    item_mrp = np.random.uniform(30.0, 270.0, n_samples)
    outlet_establishment_years = np.random.randint(1985, 2010, n_samples)
    outlet_age = 2023 - outlet_establishment_years
    
    # Generate outlet types
    outlet_types = np.random.choice(
        ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],
        n_samples,
        p=[0.2, 0.5, 0.2, 0.1]
    )
    
    # Generate outlet locations
    outlet_locations = np.random.choice(
        ['Tier 1', 'Tier 2', 'Tier 3'],
        n_samples,
        p=[0.3, 0.4, 0.3]
    )
    
    # Generate outlet sizes
    outlet_sizes = np.random.choice(
        ['Small', 'Medium', 'High', 'Unknown'],
        n_samples,
        p=[0.3, 0.3, 0.3, 0.1]
    )
    
    # Generate item types
    item_types = np.random.choice(
        ['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 
         'Household', 'Baking Goods', 'Snack Foods', 'Frozen Foods',
         'Breakfast', 'Health and Hygiene', 'Hard Drinks', 'Canned',
         'Breads', 'Starchy Foods', 'Others'],
        n_samples
    )
    
    # Generate item fat content
    item_fat_contents = np.random.choice(
        ['Low Fat', 'Regular', 'Non Fat'],
        n_samples,
        p=[0.6, 0.3, 0.1]
    )
    
    # Create the target variable based on features (with noise)
    # This is a simplified model for demonstration
    base_sales = 1000 + item_mrp * 5 - item_visibilities * 1000 + item_weights * 20
    outlet_type_effect = np.where(outlet_types == 'Grocery Store', -200, 
                         np.where(outlet_types == 'Supermarket Type1', 0,
                         np.where(outlet_types == 'Supermarket Type2', 200, 400)))
    location_effect = np.where(outlet_locations == 'Tier 1', 200,
                      np.where(outlet_locations == 'Tier 2', 0, -200))
    age_effect = -outlet_age * 5
    
    # Combine all effects with some random noise
    sales = base_sales + outlet_type_effect + location_effect + age_effect + np.random.normal(0, 100, n_samples)
    sales = np.maximum(sales, 0)  # Ensure no negative sales
    
    # Create a DataFrame
    data = pd.DataFrame({
        'Item_Weight': item_weights,
        'Item_Visibility': item_visibilities,
        'Item_MRP': item_mrp,
        'Outlet_Establishment_Year': outlet_establishment_years,
        'Outlet_Age': outlet_age,
        'Outlet_Type': outlet_types,
        'Outlet_Location_Type': outlet_locations,
        'Outlet_Size': outlet_sizes,
        'Item_Type': item_types,
        'Item_Fat_Content': item_fat_contents,
        'Item_Outlet_Sales': sales
    })
    
    # Save the synthetic dataset (optional)
    data.to_csv('synthetic_bigmart_data.csv', index=False)
    print("Synthetic data created and saved to 'synthetic_bigmart_data.csv'")
    
    # Preprocessing
    # Encode categorical variables
    categorical_cols = ['Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Item_Type', 'Item_Fat_Content']
    
    for col in categorical_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
    
    # Split features and target
    X = data[['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Outlet_Age', 
              'Outlet_Type', 'Outlet_Location_Type', 'Outlet_Size']]
    y = data['Item_Outlet_Sales']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = XGBRegressor(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=3,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Model R² score on training data: {train_score:.4f}")
    print(f"Model R² score on testing data: {test_score:.4f}")
    
    # Save the model
    with open('model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print("Model trained and saved as 'model.pkl'")

if __name__ == "__main__":
    train_model()
