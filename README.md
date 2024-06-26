# Unified-Ticketing-System
##  Google Drive Link: https://drive.google.com/drive/folders/17TmQsCARgNbKBl-dSD0F0B1kQqFFq_L7?usp=sharing
Welcome to the Unified Transport Ticketing System repository! This project aims to create a seamless, integrated public transport network by implementing a unified ticketing system, route optimization algorithms, and a user-friendly application. This repository contains code for a Streamlit application that demonstrates key functionalities of the project using mock data.

Project Overview:
 Objective
  * Enhance commuter experience through convenience and real-time information.
  * Promote public transport usage by reducing travel time and improving efficiency.
  * Streamline fare collection processes with "One Ticketing" technology.
 Importance
  * Improves urban mobility by offering convenience, reduced travel time, and optimized transport routes.
    
 Project Scope
  * Integration: Connect electronic ticketing machines and backend systems across Metro Rail Limited (MRL), Metropolitan Transport Corporation (MTC), Southern Railway, Auto, and Car/Bike services.
  * One Ticketing: Develop infrastructure for seamless payment and fare management.
  * Route Optimization: Implement algorithms and an application for real-time transit information and route planning.
    
 Data Collection and Sources
  Real Data (if available)
  * Transactional data from MRL, MTC, Auto, and Car/Bike services.
  * Geo-spatial data for transport routes, stops, and congestion patterns.
 Toy/Mock Data (if real data is unavailable)
  * Design mock data representing transactional histories, route maps, commuter preferences, and fare structures.
  * Use synthetic data generation techniques for realistic data creation.
    
 Data Preprocessing and Analysis
  * Cleaning and Preprocessing: Prepare transactional data for fare calculations, commuter behavior analysis, and demand forecasting.
  * Exploratory Data Analysis (EDA): Identify usage patterns, peak travel times, and popular routes.
  * Machine Learning Models:
    * Predicting commuter demand
    * Optimizing fare structures
    * Route planning

Inspiration
   This project is inspired by the vision of the Chennai Unified Metropolitan Transport Authority (CUMTA) to create a seamless and integrated public transport network in Chennai. For more information, visit CUMTA's website.

Getting Started
  * Prerequisites
    * Python 3.x
    * Streamlit
    * Pandas
    * Scikit-learn
    * Numpy
   
## Code Explanation
* The main components of the code include:

 * Data Generation: Creates mock transactional data for demonstration purposes.
 * Data Preprocessing: Cleans and preprocesses the data for analysis.
 * Model Training: Trains a simple RandomForest model to predict fares.
 * Route Optimization: Provides a mock route optimization function.
 * Streamlit App: Implements the frontend of the application using Streamlit.



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
       
          
