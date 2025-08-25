from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
import pickle
from train_ncf import NCF  # Import your trained NCF model


# Load user and resource mappings
with open("model/user_map.pkl", "rb") as f:
    user_map = pickle.load(f)

with open("model/resource_map.pkl", "rb") as f:
    resource_map = pickle.load(f)

# Load trained model
num_users = len(user_map)
num_resources = len(resource_map)
model = NCF(num_users, num_resources)
model.load_state_dict(torch.load("model/ncf_model.pth"))
model.eval()  # Set the model to evaluation mode

# FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # *
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class UserRequest(BaseModel):
    user_id: int

# Define the recommendation endpoint
@app.post("/recommend/")
async def recommend(user_request: UserRequest):
    user_id = user_request.user_id
    
    if user_id not in user_map:
        return {"error": "User not found"}

    user_tensor = torch.tensor([user_map[user_id]], dtype=torch.long)

    # Predict scores for all resources
    resource_scores = {}
    with torch.no_grad():
        for resource_id in resource_map.keys():
            resource_tensor = torch.tensor([resource_map[resource_id]], dtype=torch.long)
            score = model(user_tensor, resource_tensor).item()
            resource_scores[resource_id] = score
    
    # Get top 5 recommendations
    top_5 = sorted(resource_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Return as JSON
    return {"user_id": user_id, "recommendations": [{"resource_id": r[0], "score": r[1]} for r in top_5]}