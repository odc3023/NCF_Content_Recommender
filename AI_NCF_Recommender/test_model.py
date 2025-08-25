import pickle
import torch
import pandas as pd
from train_ncf import NCF  # Import the trained NCF model

# Load the dataset
df = pd.read_csv("data/processed_kt2.csv", dtype={'resource_id': str})  # Ensure resource IDs are strings

# Load the saved user and resource mappings
with open("model/user_map.pkl", "rb") as f:
    user_map = pickle.load(f)

with open("model/resource_map.pkl", "rb") as f:
    resource_map = pickle.load(f)

# Select a random user from the dataset
user_id = df['user_id'].sample(1).iloc[0]

print(f"Generating recommendations for user: {user_id}")

# Ensure the user exists in the mappings
if user_id not in user_map:
    print(f"Error: Selected user {user_id} not found in mappings!")
else:
    # Convert user_id to tensor
    user_tensor = torch.tensor([user_map[user_id]], dtype=torch.long)

    # Load the trained model
    num_users = len(user_map)
    num_resources = len(resource_map)
    model = NCF(num_users, num_resources)
    model.load_state_dict(torch.load("model/ncf_model.pth"))
    model.eval()  # Set the model to evaluation mode

    # use all dataset 
    subset_resources = list(resource_map.keys())

    # Print the total number of resources selected for testing
    print(f"Total resources selected for testing: {len(subset_resources)}")

    # Predict scores for the selected subset of resources
    resource_scores = {}
    with torch.no_grad():
        for resource_id in subset_resources:  # Loop through the selected subset of resources
            resource_tensor = torch.tensor([resource_map[resource_id]], dtype=torch.long)
            prediction = model(user_tensor, resource_tensor).item()
            resource_scores[resource_id] = prediction

    # Sort resources by predicted interaction score (highest to lowest)
    top_n = 10
    top_resources = sorted(resource_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    # Print top N recommendations
    print(f"Top {top_n} recommended resources for user {user_id}:")
    for i, (resource, score) in enumerate(top_resources, 1):
        print(f"{i}. Resource ID: {resource} - Predicted Score: {score:.4f}")

    # Save to CSV
    output_df = pd.DataFrame(top_resources, columns=["resource_id", "predicted_score"])
    output_filename = f"recommendations_user_{user_id}.csv"
    output_df.to_csv(output_filename, index=False)
    print(f"Recommendations saved to {output_filename}")