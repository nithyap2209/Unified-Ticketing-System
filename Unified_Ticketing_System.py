import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Mock data for the demonstration
np.random.seed(42)
data_size = 1000

# Generate transactional data
transactional_data = pd.DataFrame({
    'user_id': np.random.randint(1000, 2000, data_size),
    'route_id': np.random.randint(1, 50, data_size),
    'transport_mode': np.random.choice(['Metro', 'Bus', 'Train', 'Auto', 'Car/Bike'], data_size),
    'fare': np.random.uniform(10, 50, data_size),
    'timestamp': pd.date_range(start='1/1/2022', periods=data_size, freq='T')
})

# Data preprocessing
transactional_data['hour'] = transactional_data['timestamp'].dt.hour
scaler = StandardScaler()
transactional_data[['fare']] = scaler.fit_transform(transactional_data[['fare']])

# Train a simple model for fare prediction
X = transactional_data[['hour', 'route_id']]
y = transactional_data['fare']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Mock route optimization function
def optimize_route(start_point, end_point, routes):
    # Simple heuristic: shortest path based on the number of stops
    route = min(routes, key=lambda x: len(x))
    return route

# Mock routes
routes = {
    1: ['A', 'B', 'C', 'D', 'E'],
    2: ['A', 'F', 'G', 'H', 'E'],
    3: ['A', 'I', 'J', 'K', 'E']
}

# Streamlit app
st.title("Unified Transport Ticketing System")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    transport_mode = st.sidebar.selectbox('Transport Mode', ['Metro', 'Bus', 'Train', 'Auto', 'Car/Bike'])
    hour = st.sidebar.slider('Hour of the Day', 0, 23, 12)
    route_id = st.sidebar.selectbox('Route ID', list(range(1, 51)))
    data = {'transport_mode': transport_mode,
            'hour': hour,
            'route_id': route_id}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input
st.subheader('User Input parameters')
st.write(df)

# Predict fare
fare_prediction = model.predict(df[['hour', 'route_id']])
st.subheader('Predicted Fare')
st.write(f"${fare_prediction[0]:.2f}")

# Route optimization (dummy example)
optimized_route = optimize_route('A', 'E', routes.values())
st.subheader('Optimized Route')
st.write(' -> '.join(optimized_route))

# Displaying transaction history (sample)
st.subheader('Sample Transaction History')
st.write(transactional_data.head())