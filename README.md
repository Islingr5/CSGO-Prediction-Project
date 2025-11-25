# CS:GO AI Match Predictor (Time-Travel Logic)

> An advanced Machine Learning project that predicts the winner of professional CS:GO matches using historical data, ELO ratings, and momentum analysis. Built with XGBoost and engineered to prevent data leakage.

## Key Features

* **XGBoost Engine:** Optimized gradient boosting model with robust hyperparameters found via exhaustive search.
* **Time-Travel Simulation:** The system fetches team stats (ELO, Rank, Form) exactly as they were on the specific date of the last meeting, ensuring 100% realistic historical comparison.
* **Symmetric Training:** Implements data mirroring to eliminate positional bias (Team A vs B is treated same as Team B vs A).
* **Pistol Round Impact:** Analyzes historical pistol round win rates (Round 1 & 16) as a key momentum indicator.
* **Risk Analysis:** Provides a confidence score and risk warning (e.g., "High Risk / Coin Flip") for close matchups.
* **Leakage-Free:** Rigorous cleaning of "future" features to ensure model integrity.

## Performance

* **Accuracy:** ~78% (on 2019-2020 Test Data)
* **Methodology:** XGBoost Classification with TimeSeries Split and custom feature engineering (ELO, KAST, WR).

## Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Islingr5/CSGO-Prediction-Project.git](https://github.com/Islingr5/CSGO-Prediction-Project.git)
    cd CSGO-Prediction-Project
    ```
    *(Note: This repository tracks large binary files using Git LFS. Ensure you have Git LFS installed to fetch the model and datasets.)*

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Authentication Check:**
    Ensure you have correctly configured your Git Personal Access Token (PAT) for push operations.

## 💻 Usage & Execution

The project requires an initial training run to process all historical data and save the final model (`csgo_final_model.pkl`).

### 1. Initial Training & Model Generation (Run Once)

This script loads all raw data, generates features, trains the model on the full historical dataset (2015-2018), and validates its performance against the test set (2019-2020).

```bash
python run.py
```

## 💻 Usage & Execution

### 2. Interactive Prediction (Demo Mode)

After the model file (`csgo_v11_model.pkl`) is generated, you can use the built-in interactive shell to perform instant simulations.

* **Execution:** Loads the saved model instantly for predictions.
    ```bash
    python demo.py
    ```

**Interactive Commands:**

* **`list`**: Shows all available team names (paginated).
* **`maplist`**: Shows all available maps (e.g., Mirage, Dust2, Inferno).
* **Input Logic:** The system will prompt you for two teams and a map. It will then perform a **Time-Travel Simulation** by finding the last historical meeting between those teams and predicting the winner based on the stats of that specific day.
