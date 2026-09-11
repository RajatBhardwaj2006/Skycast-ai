import json
from pathlib import Path

def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python", "version": "3.12"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"}
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

def md_cell(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [s + "\n" for s in source.split("\n")]}

def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s + "\n" for s in source.split("\n")]}

def populate():
    # 01_eda.ipynb
    nb1_cells = [
        md_cell("# SkyCast — 01 Exploratory Data Analysis\n\nThis notebook analyzes the Indian airfare dataset distributions, pricing structures across airlines and cabin classes, and investigates why cabin class dominates feature importance."),
        code_cell("import pandas as pd\nimport numpy as np\n\n# Load datasets\nclean_df = pd.read_csv('Clean_Dataset.csv')\nnew_df = pd.read_csv('Flight-Price-Prediction-using-Machine-Learning-Random-Forest-XGBoost--main/IndianFlightdata - Sheet1.csv')\nprint(f'Clean dataset shape: {clean_df.shape}')\nprint(f'New Indian flight dataset shape: {new_df.shape}')"),
        md_cell("## 1. Cabin Class Pricing Disparity\nInvestigate why `class` accounts for ~89% of tree variance reduction."),
        code_cell("class_stats = clean_df.groupby('class')['price'].describe()\nprint(class_stats)\n# Observation: Business class mean is ~₹52,540 vs Economy mean ~₹6,572 (an 8x difference).\n# This is a genuine aviation pricing structure, not an artifact."),
        md_cell("## 2. Airline and Booking Window Trends"),
        code_cell("airline_stats = clean_df.groupby('airline')['price'].agg(['count', 'mean', 'median', 'std'])\nprint(airline_stats.sort_values('mean', ascending=False))\n\ndays_left_trend = clean_df.groupby('days_left')['price'].mean()\nprint('Price by days_left sample (1, 15, 30, 45):')\nprint(days_left_trend.loc[[1, 15, 30, 45]])"),
        md_cell("## 3. Route Coverage Analysis"),
        code_cell("routes = clean_df.groupby(['source_city', 'destination_city']).size()\nprint(f'Unique tier-1 routes: {len(routes)}')\nprint(routes.head(10))")
    ]
    Path("notebooks/01_eda.ipynb").write_text(json.dumps(make_nb(nb1_cells), indent=2), encoding="utf-8")

    # 02_preprocessing.ipynb
    nb2_cells = [
        md_cell("# SkyCast — 02 Data Cleaning & Feature Engineering\n\nStandardizing schemas, geospatial coordinates, Haversine distance, and scikit-learn preprocessing pipelines."),
        code_cell("import pandas as pd\nimport numpy as np\nfrom src.data.clean import clean_clean_dataset\nfrom src.features.geo_features import add_geographic_features\nfrom src.features.preprocess import build_preprocessor\n\nraw_data = pd.read_csv('data/processed/merged_data.csv', low_memory=False)\nprint(f'Loaded merged data: {raw_data.shape}')"),
        md_cell("## 1. Cleaning and Normalization"),
        code_cell("cleaned = clean_clean_dataset(raw_data)\nprint(f'Cleaned dataset: {cleaned.shape}')\nprint('Airlines:', cleaned['airline'].unique().tolist())\nprint('Classes:', cleaned['class'].unique().tolist())\nprint('Departure Bins:', cleaned['departure_time'].unique().tolist())"),
        md_cell("## 2. Geospatial Feature Engineering & Haversine Distance"),
        code_cell("featured = add_geographic_features(cleaned)\nprint(featured[['source_city', 'destination_city', 'distance_km', 'source_lat', 'destination_lat']].drop_duplicates().head(10))"),
        md_cell("## 3. Preprocessor Pipeline"),
        code_cell("preprocessor = build_preprocessor()\nprint(preprocessor)")
    ]
    Path("notebooks/02_preprocessing.ipynb").write_text(json.dumps(make_nb(nb2_cells), indent=2), encoding="utf-8")

    # 03_modeling.ipynb
    nb3_cells = [
        md_cell("# SkyCast — 03 Model Training, Comparison & Auditing\n\nTraining candidate regression algorithms (Linear, Ridge, Random Forest, HistGradientBoosting, XGBoost), evaluating in-distribution random splits vs realistic route-group and airport-group holdout splits."),
        code_cell("import json\nimport joblib\nimport pandas as pd\n\nwith open('models/metrics.json') as f:\n    metrics = json.load(f)\nprint('Best model:', metrics.get('best_model'))\nprint('Test MAE:', metrics.get('mae'))\nprint('Test R²:', metrics.get('r2'))"),
        md_cell("## 1. Candidate Comparison Table"),
        code_cell("with open('models/model_comparison.json') as f:\n    comp = json.load(f)\ncomp_df = pd.DataFrame(comp.get('rows', []))\nprint(comp_df[['model', 'mae', 'rmse', 'r2', 'training_seconds']])"),
        md_cell("## 2. Route-Group and Airport-Group Audits\nGeneralization to unseen routes and unseen origin airports."),
        code_cell("with open('models/validation.json') as f:\n    val = json.load(f)\nprint('Route-Group Audit:', val.get('route_group_audit', {}).get('metrics'))\nprint('Airport-Group Audit:', val.get('airport_group_audit', {}).get('metrics'))")
    ]
    Path("notebooks/03_modeling.ipynb").write_text(json.dumps(make_nb(nb3_cells), indent=2), encoding="utf-8")
    print("Notebooks successfully populated!")

if __name__ == "__main__":
    populate()
