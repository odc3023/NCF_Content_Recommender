# Neural Collaborative Filtering (NCF) Recommender System

## Overview
This project implements a Neural Collaborative Filtering (NCF) recommender system using PyTorch. It predicts and recommends resources for users based on historical interaction data.

## Dataset
- The dataset is located in the `data/` folder.
- It contains processed data (`processed_kt2.csv`) and individual user interaction files in `data/KT2/`.
- The dataset includes user IDs, resource IDs, and interaction information.

## Model
- The trained NCF model is saved in `model/ncf_model.pth`.
- User and resource mappings are stored in `model/user_map.pkl` and `model/resource_map.pkl`.

## Setup
1. Clone the repository:
   ```sh
   git clone 
   cd 
   ```
2. Create and activate a virtual environment:
   ```sh
   python3 -m venv ncf_env
   source ncf_env/bin/activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
To test the model and generate recommendations:
```sh
cd AI_NCF_Recommender
python test_model.py
```
This will generate recommendations for a sample user and save them to a CSV file (e.g., `recommendations_user_723965.csv`).

## Real Results
```
Generating recommendations for user: 723965
Total resources selected for testing: 20031
Top 10 recommended resources for user 723965:
1. Resource ID: b3204 - Predicted Score: 1.0000
2. Resource ID: q4672 - Predicted Score: 1.0000
3. Resource ID: b3310 - Predicted Score: 1.0000
4. Resource ID: q4778 - Predicted Score: 1.0000
5. Resource ID: b5627 - Predicted Score: 1.0000
6. Resource ID: q8156 - Predicted Score: 1.0000
7. Resource ID: b201 - Predicted Score: 1.0000
8. Resource ID: q201 - Predicted Score: 1.0000
9. Resource ID: b489 - Predicted Score: 1.0000
10. Resource ID: q489 - Predicted Score: 1.0000
Recommendations saved to recommendations_user_723965.csv
```

## Files
- `app.py`: FastAPI app for serving recommendations
- `train_ncf.py`: Training script for the NCF model
- `test_model.py`: Script to test the model and generate recommendations
- `preprocess_data.py`: Data preprocessing script


## Contact
Name: Osheen Constable
Github Profile: https://github.com/odc3023 

