import pandas as pd
import numpy as np
import os

# Specify the folder path where your 60 CSV files are located
folder_path = 'D:\\Thesis\\CLAS\\Answers'

# Initialize an empty list to store average values
average_values = []
file = []
# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Create the full path to the CSV file
        file_path = os.path.join(folder_path, file_name)

        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Check if 'answer' column exists in the DataFrame
        # if 'answer' in df.columns:
            # Calculate the average value for each CSV file
        n = df[df[' answer'] == 'correct'].count()
        average_value = (n / 75) * 100  # Assuming a single column in each CSV file
        average_values.append(average_value.values[0])
        file.append(file_name)
        # print(average_value)

# Create a DataFrame with the average values
result_df = pd.DataFrame({'File Name': file,'Average Values': average_values})
# Save the DataFrame to a new CSV file
result_df.to_csv('average_values_output.csv', index=False)

# print(result_df)


# import pandas as pd 

# df = pd.read_csv('.//dataset.csv')
# print(df[df["Average_Values"]])