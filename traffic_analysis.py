import pandas as pd

# Load the original dataset
file_path = 'C:/Users/user/Downloads/archive.zip'
df = pd.read_csv(file_path)

# Check for missing values
missing_values = df.isnull().sum()

# Check data types
data_types = df.dtypes

# Check columns
columns = df.columns

# Display the information
print("Missing values in each column:\n", missing_values)
print("\nData types of each column:\n", data_types)
print("\nColumn names:\n", columns)
# Check for unique junctions
unique_junctions = df['Junction'].unique()
num_unique_junctions = len(unique_junctions)

# Display the number of unique junctions and their names
print("Number of unique junctions:", num_unique_junctions)
print("Unique junctions:\n", unique_junctions)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Convert the DateTime column to datetime format
df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')

# Extract time-related features
df['hour'] = df['DateTime'].dt.hour
df['day_of_week'] = df['DateTime'].dt.dayofweek
df['month'] = df['DateTime'].dt.month
df['year'] = df['DateTime'].dt.year

# Get the unique junctions
unique_junctions = df['Junction'].unique()

# Perform EDA for each junction
for junction in unique_junctions:
    print(f"\nEDA for Junction {junction}")
    
    # Filter data for the junction
    df_junction = df[df['Junction'] == junction]
    
    # Traffic patterns by hour of the day
    plt.figure(figsize=(12, 6))
    df_junction.groupby('hour')['Vehicles'].mean().plot(kind='bar')
    plt.title(f'Average Traffic Volume by Hour of the Day - Junction {junction}')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Number of Vehicles')
    plt.show()
 # Traffic patterns by day of the week
    plt.figure(figsize=(12, 6))
    df_junction.groupby('day_of_week')['Vehicles'].mean().plot(kind='bar')
    plt.title(f'Average Traffic Volume by Day of the Week - Junction {junction}')
    plt.xlabel('Day of the Week')
    plt.ylabel('Average Number of Vehicles')
    plt.xticks(ticks=np.arange(7), labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()
# Traffic patterns by month
    plt.figure(figsize=(12, 6))
    df_junction.groupby('month')['Vehicles'].mean().plot(kind='bar')
    plt.title(f'Average Traffic Volume by Month - Junction {junction}')
    plt.xlabel('Month')
    plt.ylabel('Average Number of Vehicles')
    plt.xticks(ticks=np.arange(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.show()
# Time series plot for the junction
    plt.figure(figsize=(15, 6))
    plt.plot(df_junction['DateTime'], df_junction['Vehicles'], label=f'Junction {junction}')
    plt.title(f'Traffic Volume Over Time - Junction {junction}')
    plt.xlabel('DateTime')
    plt.ylabel('Number of Vehicles')
    plt.legend()
    plt.show()
# Heatmap of traffic volume by hour of the day and day of the week
    heatmap_data = df_junction.groupby(['hour', 'day_of_week'])['Vehicles'].mean().unstack()
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='viridis', annot=True, fmt=".1f")
    plt.title(f'Heatmap of Average Traffic Volume by Hour and Day - Junction {junction}')
    plt.xlabel('Day of the Week')
    plt.ylabel('Hour of the Day')
    plt.xticks(ticks=np.arange(7) + 0.5, labels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.show()
    