import pandas as pd
import streamlit as st
import requests
import sqlite3
import re
import folium
from streamlit_folium import folium_static
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set up Streamlit layout
st.set_page_config(page_title="Bus Appointment and Arrival Prediction", layout="wide")

#importing the data
dat=pd.read_csv("Arrival Data.csv")

df=pd.read_csv("T_Data.csv")

# Split the data into features (X) and target variables (y)
X = df[['Time', 'Day', 'Traffic_level', 'Route_ID', 'Bus_ID']]
y_arrival = df['Arrival_time']
y_departure = df['Departure']

#Split the data into training and testing sets
X_train, X_test, y_arrival_train, y_arrival_test = train_test_split(X, y_arrival, test_size=0.2, random_state=42)
X_train, X_test, y_departure_train, y_departure_test = train_test_split(X, y_departure, test_size=0.2, random_state=42)

# Train the Random Forest model for arrival time prediction
rf_arrival = RandomForestRegressor(random_state=42)
rf_arrival.fit(X_train, y_arrival_train)

# Train the Random Forest model for departure time prediction
rf_departure = RandomForestRegressor(random_state=42)
rf_departure.fit(X_train, y_departure_train)

#Database connection
# Create a SQLite database connection
conn = sqlite3.connect('student_db.sqlite')
cursor = conn.cursor()

# Create a table to store student credentials if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    full_name TEXT,
                    reg_number TEXT
                )''')

def insert_student(full_name, reg_number):
    # Check if student already exists in the database
    cursor.execute("SELECT * FROM students WHERE full_name = ? AND reg_number = ?", (full_name, reg_number))
    existing_student = cursor.fetchone()

    if existing_student:
        st.warning("Student already exists in the database.")
    else:
        cursor.execute("INSERT INTO students (full_name, reg_number) VALUES (?, ?)", (full_name, reg_number))
        conn.commit()    

# Function to count the number of students logged in
def count_students():
    cursor.execute("SELECT COUNT(*) FROM students")
    result = cursor.fetchone()
    return result[0]

def validate_reg_number(reg_number):
    pattern = r'^[a-zA-Z]+\d+[a-zA-Z]+$'
    if not re.match(pattern, reg_number):
        st.warning("Invalid registration number. Please enter a mix of letters and numbers.")
        return False
    return True

# Function to clear the database
def clear_database():
    cursor.execute("DELETE FROM students")
    conn.commit()
    


# Create the Streamlit app
def main():
    if st.sidebar.checkbox("Home"):
        st.markdown("<h1 style='text-align: center;'>Welcome to Midlands Traffic Route Analysis.</h1>", unsafe_allow_html=True)
        
          # Senga area coordinates
        #traffic_data=[2,4,6]  
        senga_lat= -19.4616
        senga_lon= 29.8206
        
# Create a Folium map centered on the Senga area
        map_senga = folium.Map(location=[senga_lat, senga_lon], zoom_start=12)
        # Define the traffic route data
        traffic_data = [
        [-19.4700, 29.8300],  # Example coordinate pair 1
        [-19.4650, 29.8250],  # Example coordinate pair 2
    # Add more coordinate pairs as needed
        ]

        # Add traffic routes to the map as PolyLine or GeoJson layers
        # You would replace `traffic_data` with your actual traffic route data
        #traffic_data = ...  # Load or preprocess your traffic route data
        folium.PolyLine(locations=traffic_data, color='red', weight=2, opacity=0.8).add_to(map_senga)

       # Display the map in Streamlit
        st.title("Traffic Routes in Senga Area")
        folium_static(map_senga)
    
    
    dataset_expander = st.sidebar.expander("Sample Dataset")
    correlation_expander = st.sidebar.expander("Correlation Matrix")

    # Display the dataset inside the dataset expander
    with dataset_expander:
        st.write("Sample Dataset")
        st.write(dat.head())

    # Display the correlation matrix inside the correlation expander
    with correlation_expander:
        st.write("Correlation Matrix")
        st.write(df.corr())
        
    st.sidebar.title("Arrival and Departure Time Prediction")
    show_predictions = st.sidebar.checkbox("Arrival and Departure Time Predictions")

    # Display the selected predictions
    if show_predictions:
        st.subheader("Arrival and Departure Time Predictions")
        # User input for new prediction
        new_data = {
            'Time': [st.number_input("Time (700-1600)", min_value=700, max_value=1600, value=800)],
            'Day': [int(st.selectbox("Day (1-6)", options=[1, 2, 3, 4, 5, 6]))],
            'Traffic_level': [st.number_input("Traffic Level (1-50)", min_value=1, max_value=50, value=2)],
            'Route_ID': [int(st.selectbox("Route ID (1-4)", options=[1, 2, 3, 4]))],
            'Bus_ID': [int(st.selectbox("Bus ID (3, 6, 9)", options=[3, 6, 9]))]
        }

        new_df = pd.DataFrame(new_data)
        
        # Make predictions for arrival and departure times
        new_arrival_pred = rf_arrival.predict(new_df)
        new_departure_pred = rf_departure.predict(new_df)

        # Convert decimal predictions to time objects
        arrival_time = datetime.strptime("0800", "%H%M") + timedelta(minutes=new_arrival_pred[0])
        departure_time = datetime.strptime("0800", "%H%M") + timedelta(minutes=new_departure_pred[0])

        # Format the time objects as strings
        formatted_arrival_time = arrival_time.strftime("%H:%M")
        formatted_departure_time = departure_time.strftime("%H:%M")
        # Display the predicted arrival time
        st.write("Predicted Arrival Time:", formatted_arrival_time)
        # Display the predicted departure time
        st.write("Predicted Departure Time:", formatted_departure_time)
    
    
    # Set the sidebar heading
    st.sidebar.title("Analysis")

    # Create checkboxes for different graphs
    show_time_series = st.sidebar.checkbox("Bus Arrival and Departure Time Analysis")
    show_day_of_week = st.sidebar.checkbox("Distribution of Bus Records by Day")
    show_traffic_level = st.sidebar.checkbox("Traffic Level Analysis")
    show_bus_analysis = st.sidebar.checkbox("Bus Records Distribution by Bus ID")
    
    # Display the selected graphs
    #if st.sidebar.checkbox("Analysis"):
    if show_time_series:
        st.subheader("Time Series Analysis")
        fig_time_series = go.Figure()
        fig_time_series.add_trace(go.Scatter(x=df['Day'], y=df['Arrival_time'], mode='lines+markers', name='Arrival Time'))
        fig_time_series.add_trace(go.Scatter(x=df['Day'], y=df['Departure'], mode='lines+markers', name='Departure Time'))
        fig_time_series.update_layout(title='Bus Arrival and Departure Times Weekly Analysis',
                                      xaxis_title='Day',
                                      yaxis_title='Time')
        st.plotly_chart(fig_time_series)
    
    if show_day_of_week:
        st.subheader("Day of the Week Analysis")
        fig_day_of_week = go.Figure(data=[go.Pie(labels=df['Day'], values=df['Day'])])
        fig_day_of_week.update_layout(title='Distribution of Bus Records by Day of the Week')
        st.plotly_chart(fig_day_of_week)
       
    if show_traffic_level:
        st.subheader("Traffic Level Analysis")
        fig_traffic_level = go.Figure(data=[go.Bar(x=df['Day'], y=df['Traffic_level'])])
        fig_traffic_level.update_layout(title='Traffic Level Distribution',
                                        xaxis_title='Day',
                                        yaxis_title='Traffic Level')
        st.plotly_chart(fig_traffic_level)
    
    if show_bus_analysis:
        st.subheader("Bus Schedulling Analysis")
        fig_bus = go.Figure(data=[go.Bar(x=df['Day'], y=df['Bus_ID'])])
        fig_bus.update_layout(title='Distribution of Bus Records by Bus ID',
                              xaxis_title='Day',
                              yaxis_title='Bus ID')
        st.plotly_chart(fig_bus)
        
    approval_message = ""   
    st.sidebar.title("Student Login")    
    if st.sidebar.checkbox("Driver Message Box"):
    # Create an input field for the driver to send an approval message
       approval_message = st.sidebar.text_input("Driver's Message")
       #st.sidebar.info(f"Approval Message Sent: {approval_message}")
    # Student login section
    if st.sidebar.checkbox("Student Login"):
        
        full_name = st.text_input("Full Name")
        reg_number = st.text_input("Registration Number")
    
    # Check if student credentials are provided
        if full_name and reg_number:
           
          if not validate_reg_number(reg_number):
            return
          # Enable the "Send Request" button
          send_request_button = st.button("Send Request")

          if send_request_button:
        # Insert student credentials into the database
             insert_student(full_name, reg_number)
             

        # Count the number of students logged in
             total_students = count_students()

        # Display a popup message with the total number of students logged in
             st.info(f"Total Students Logged In: {total_students}")
            # Display the driver approval message
           
          if approval_message:
             st.info(f"Driver's Message: {approval_message}")
             #st.sidebar.info(f"Approval Message Sent: {approval_message}")

             
         
    st.sidebar.title("Driver Scheduling and Optimization")
    # Create a checkbox to display the total number of logged-in students
    show_total_students = st.sidebar.checkbox("Total Students Logged In")
    # Display the total number of students when the checkbox is selected
    if show_total_students:
        total_students = count_students()
        st.sidebar.info(f"Total Students Logged In: {total_students}")
        
          
        # Clear database option
        if st.checkbox("Clear Database"):
          clear_database()
          st.warning("Database cleared. All student records have been deleted.")
# Run the Streamlit app
if __name__ == "__main__":
    main()