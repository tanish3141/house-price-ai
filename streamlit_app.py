import streamlit as st
import numpy as np
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
#custom class for catrggory pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6 
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs 
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    def fit(self, X, y=None):
        return self  # nothing else to do 
    def transform(self, X, y=None): 
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix] 
        if self.add_bedrooms_per_room: 
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room] 
        else:
            return np.c_[X, rooms_per_household, population_per_household]
model = joblib.load("house_price_model_compressed.pkl")

  
st.title("🏡 California House Price Prediction ")
st.markdown("Adjust the sliders and see how the estimated house price changes!")
income_dollars = st.slider(
    "Median Income (in dollars)",
    min_value=10000,
    max_value=150000,
    value=50000,
    step=5000
)

latitude = st.slider(
    "Latitude", 
    min_value=32.0, 
    max_value=42.0, 
    value=37.0, 
    step=0.1
)

longitude = st.slider(
    "Longitude", 
    min_value=-125.0, 
    max_value=-114.0, 
    value=-120.0, 
    step=0.1
)

housing_median_age = st.slider(
    "Median Age of Houses", 
    min_value=1, 
    max_value=50, 
    value=20
)

total_rooms = st.slider(
    "Total Rooms", 
    min_value=100, 
    max_value=10000, 
    value=2000, 
    step=100
)

total_bedrooms = st.slider(
    "Total Bedrooms", 
    min_value=50, 
    max_value=2000, 
    value=400, 
    step=50
)

population = st.slider(
    "Population", 
    min_value=100, 
    max_value=5000, 
    value=1000, 
    step=100
)

households = st.slider(
    "Households", 
    min_value=50, 
    max_value=2500, 
    value=400, 
    step=50
)

ocean_proximity = st.selectbox(
    "Ocean Proximity", 
    ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
)
median_income_scaled = income_dollars / 10000.0
input_data = np.array([[
    longitude, latitude, housing_median_age, total_rooms,
    total_bedrooms, population, households, median_income_scaled, ocean_proximity
]])
if st.button("Predict House Price"):
  prediction=model.predict(input_data)
  st.success(f"🏠 Estimated Median House Value: **${prediction[0]:,.2f}**")
  

