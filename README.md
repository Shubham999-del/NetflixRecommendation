# NetflixRecommendation

# 🎬 Netflix Recommendation System

This repository showcases a movie recommendation system built using collaborative filtering techniques. It utilizes a large-scale Netflix dataset to recommend the most relevant movies to users based on their previous interactions. The project walks through comprehensive data preprocessing, missing value imputation, and the implementation of Singular Value Decomposition (SVD) for building personalized recommendations.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Data Preprocessing](#data-preprocessing)
- [Missing Value Imputation](#missing-value-imputation)
- [Modeling - SVD](#modeling---svd)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

---

## 🔍 Overview

The aim of this project is to build a robust movie recommendation engine using **collaborative filtering**. We employ the SVD (Singular Value Decomposition) technique to discover latent features that best represent the interactions between users and movies.

Key Highlights:
- Large-scale dataset handling
- Advanced missing value handling
- Dimensionality reduction using SVD
- Personalized recommendations

---

## 🎯 Dataset

- **Source:** [Netflix Prize dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data) or similar
- **Contents:** 
  - User IDs
  - Movie IDs
  - Ratings (1 to 5)
  - Timestamps (optional)

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** 
  - `pandas`, `numpy` – Data manipulation
  - `sklearn.impute` – KNN imputation
  - `matplotlib`, `seaborn` – Visualization
  - `scipy.sparse.linalg` – SVD
  - `surprise` (optional) – SVD recommender framework

---

## 🧹 Data Preprocessing

1. **Data Type Conversion:**  
   - Converted columns to appropriate data types (e.g., user IDs to strings, ratings to float).
   
2. **Null Value Detection:**  
   - Identified missing values using `isnull().sum()` and visual inspection.

3. **Handling Duplicates:**  
   - Removed duplicate user-movie interactions to avoid bias.

4. **Outlier Checks:**  
   - Checked for anomalous ratings outside the allowed range (1–5).

---

## 🧩 Missing Value Imputation

Various imputation techniques were explored based on the distribution and nature of missing data:

- **KNN Imputation:**  
  Imputed missing values using `KNNImputer` from `sklearn`, leveraging similarity among users/movies.

- **Forward Fill / Backward Fill:**  
  Used time-ordered fill methods where applicable, particularly for timestamped data.

- **Mean/Median Imputation:**  
  Conditional imputation based on user or movie-level statistics was tested.

---

## 🧠 Modeling - SVD

### 📌 Why SVD?

SVD is a matrix factorization technique used to reduce dimensionality and uncover latent user/item features. It is ideal for sparse datasets like ratings matrices.

### 🔢 Steps:

1. **Create User-Item Rating Matrix**
2. **Fill missing entries with average rating or zeros**
3. **Apply SVD:**
   ```python
   from scipy.sparse.linalg import svds
   U, sigma, Vt = svds(user_item_matrix, k=50)
