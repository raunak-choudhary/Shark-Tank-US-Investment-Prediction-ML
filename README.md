# Shark Tank US Investment Prediction

This project explores the investment behavior of prominent investors ("sharks") on the American TV show **Shark Tank US** by applying machine learning techniques to predict investment likelihood and amount based on historical pitch data from all 16 seasons.

## 🎯 Objective

- Understand what factors influence sharks’ investment decisions
- Predict which shark(s) are likely to invest in a startup
- Predict investment amounts using regression models
- Uncover industry preferences and funding trends using NLP and data visualizations

## 📊 Dataset Overview

- **Source**: Kaggle
- **Size**: 1365 startup pitches × 53 features
- **Key Features**:
  - Startup Name, Industry, Business Description
  - Pitchers' Gender, Location, Viewership
  - Ask Amount, Equity Offered, Valuation Requested
  - Deal Details and Shark-wise Investment Info
  - Shark Presence and Investment Amounts

## 🛠️ Key Methods and Tools

- **ML Models Used**:
  - Classification: Logistic Regression, SVM, Random Forest, XGBoost
  - Regression: Random Forest Regressor, XGBoost Regressor
- **Techniques Applied**:
  - SMOTE for class imbalance
  - Text Embeddings using OpenAI’s embedding models
  - Dimensionality Reduction: PCA (for modeling), t-SNE (for visualization)
  - Feature Engineering: Combined shark-wise investments into unified vectors
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, Confusion Matrix (for classification)
  - RMSE, R² Score (for regression)

## 📈 Insights

- XGBoost outperformed all other models in classification for **5 out of 6 sharks**
- Regression results showed XGBoost provided the best estimates for investment amounts
- High viewership and moderate ask amounts increased deal probability
- Sharks have clear industry biases:  
  - Barbara Corcoran → Food & Beverage  
  - Lori Greiner → Lifestyle/Home  
  - Daymond John → Fashion/Beauty

## 📦 Outputs

- Cleaned and preprocessed dataset with embedded features
- Visualizations of shark investments by industry
- 3D t-SNE plots for text embedding similarity
- Trained ML models for classification and regression tasks
- Shark-wise performance tables post hyperparameter tuning

## 👨‍💻 Authors

- **Raunak Choudhary**  
  Email: [rc5553@nyu.edu](mailto:rc5553@nyu.edu)  
  [LinkedIn](https://www.linkedin.com/in/raunak-choudhary)

- **Subhrajit Dey**  
  Email: [sd5963@nyu.edu](mailto:sd5963@nyu.edu)

- **Saniya Gapchup**  
  Email: [syg2021@nyu.edu](mailto:syg2021@nyu.edu)

## 🏫 Affiliation

Department of Computer Science and Engineering  
New York University – Tandon School of Engineering

## 🔮 Future Work

- Apply transformer-based NLP models (e.g., BERT) on Business Descriptions
- Incorporate external funding round data for validation
- Deploy as a web-based pitch evaluation tool or simulation game