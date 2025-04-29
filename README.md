# Movie Genre Classification Using NLP

## Project Overview
This project aims to develop a machine learning model that classifies movies into genres based on their plot descriptions. The dataset used is the IMDb Genre Classification Dataset, which contains movie plot summaries and their associated genres.

## Objectives
- Preprocess text-based movie plot descriptions.
- Convert raw text into numerical vectors suitable for machine learning.
- Explore various classifiers to identify the best model for multi-label genre classification.
- Analyze feature importance and misclassifications to gain insights.

## Dataset
- The dataset consists of movie plots and their genres.
- Due to size constraints, the dataset is **not included** in this repository.
- You can download the dataset from [Kaggle IMDb Genre Classification Dataset](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).

## Repository Structure
├── src/
│ └── main.py # Main script for training and evaluation
├── notebooks/
│ └── EDA.ipynb # Exploratory data analysis notebook
├── results/
│ └── genre_distribution.png # Visualization of genre distribution
├── README.md # This file
├── requirements.txt # Python dependencies
└── .gitignore # To ignore dataset files and other unnecessary files


## How to Run

1. **Clone the repository:**
https://github.com/khushia3/Movie-Genre-Classification.git

2. **Download the dataset:**
- Download `train_data.txt` from [Kaggle](https://www.kaggle.com/datasets/hijest/genre-classification-dataset-imdb).
- Place the file in the root directory or a `data/` folder.

3. **Install dependencies:**
pip install -r requirements.txt

4. **Run the main script:**
python src/main.py

## Key Features
- Text preprocessing with NLTK (stopword removal, cleaning).
- TF-IDF vectorization of movie plots.
- Multi-label classification using Logistic Regression, Random Forest, and Linear SVM.
- Model evaluation with F1 score and classification reports.
- Visualization of genre distribution.
- Feature importance extraction for Logistic Regression.
- Misclassification analysis via confusion matrices.

## Results
- The best performing model is Logistic Regression (or whichever your best model is).
- Achieved F1 score (micro-average): XX.XX%
- Insights into top predictive words per genre are provided.

## Notes on Dataset Size
- The dataset is large and **not included** in this repository.
- Please download it separately from Kaggle.
- The `.gitignore` file excludes dataset files to keep the repository lightweight.

## Future Work
- Experiment with deep learning models (e.g., BERT, LSTM).
- Hyperparameter tuning and cross-validation.
- Use pretrained embeddings for better semantic understanding.

## Contact
For questions or suggestions, please contact [your.email@example.com].

---

