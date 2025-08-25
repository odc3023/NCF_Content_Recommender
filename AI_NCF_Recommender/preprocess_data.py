import pandas as pd
import os

# Path to the KT2 dataset folder
data_folder = "data/KT2"
output_file = "data/processed_kt2.csv"

# Initialize an empty DataFrame to store processed data
chunk_size = 10000  

# Process CSVs one by one
for file in os.listdir(data_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(data_folder, file)
        
        # Extract user_id from filename (e.g., 'u1.csv' -> user_id = 1)
        user_id = int(file.split('.')[0][1:])  # Removes 'u' prefix and converts to int

        # Process in chunks
        chunk_list = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            chunk = chunk[['timestamp', 'item_id', 'user_answer']]  # Keep only needed columns
            chunk.rename(columns={'item_id': 'resource_id', 'user_answer': 'interaction'}, inplace=True)
            chunk['user_id'] = user_id  # Add user_id column
            chunk['interaction'] = chunk['interaction'].apply(lambda x: 1 if x != '0' else 0)  # Binary interactions
            chunk_list.append(chunk)

        # Append data in chunks to the output file
        processed_df = pd.concat(chunk_list, ignore_index=True)
        processed_df.to_csv(output_file, mode='a', header=not os.path.exists(output_file), index=False)

        print(f"Processed {file} and appended to {output_file}")

print("Preprocessing complete! All user data combined into processed_kt2.csv.")