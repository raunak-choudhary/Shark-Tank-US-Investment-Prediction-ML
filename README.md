# Shark Tank US Investment Prediction

Can you predict which investor will back a startup, and how much they will commit, from what is knowable before the pitch ends?

This project treats every episode of Shark Tank US as a labelled decision record and builds a per-investor model of investment behaviour across **1,365 pitches**, combining OpenAI text embeddings of unstructured pitch data with structured deal terms.

## The Problem

Most "did they get a deal?" analyses collapse the show into a single binary outcome. That throws away the interesting part. Six sharks with different capital, different risk appetites, and visibly different industry preferences are each making an independent decision on every pitch.

So the problem is modelled as it actually occurs:

**Task 1, multi-label classification.** For a given pitch, which of the six sharks invests? Six independent binary classifiers rather than one, because a pitch can attract any subset of the panel.

**Task 2, per-shark regression.** Conditional on the deal landscape, how much does each shark commit? Six regressors predicting dollar amounts.

The panel modelled: Barbara Corcoran, Mark Cuban, Lori Greiner, Robert Herjavec, Daymond John, and Kevin O'Leary.

## Dataset

A Kaggle-sourced record of Shark Tank US pitches: **1,365 rows across 53 columns**, spanning show metadata, pitch economics, pitcher demographics, and per-shark investment outcomes.

The raw data is heavily incomplete, carrying **34,467 missing values** on arrival. Column selection reduced 53 columns to 23 by removing three distinct categories of problem:

- **Outcome leakage.** `Total Deal Amount`, `Total Deal Equity`, `Deal Valuation`, and `Got Deal` are consequences of the decision being predicted, not inputs to it. Keeping them would let the model read the answer.
- **Non-predictive identifiers.** Episode number, pitch order, air dates, entrepreneur names, and company URLs.
- **Sparse or redundant fields.** Per-shark equity columns duplicating the amount columns, and guest-shark fields that are inconsistently populated.

After that reduction, missing values fell from 34,467 to 4,238. The remainder were handled by strategy rather than blanket deletion: median imputation for continuous economics (`Original Ask Amount`, `Original Offered Equity`, `Valuation Requested`, `US Viewership`), zero-fill for shark presence flags, and outright removal of `Pitchers City` and `Pitchers State`, which were missing in 834 and 566 rows respectively and too sparse to impute honestly.

## Feature Engineering

**Restructuring the target.** Six separate per-shark investment columns were folded into two vector-valued columns: a binary `Sharks Invested` indicator vector and a parallel `Sharks Investment Amounts` vector. This is what makes the multi-label framing tractable, since one row now carries the full panel decision.

**Embedding the unstructured fields.** `Startup Name`, `Industry`, `Business Description`, and `Pitchers Gender` are text. They were encoded with OpenAI's `text-embedding-3-small` model, producing dense vectors that carry semantic proximity a one-hot encoding cannot, so that a plant-based snack company and an organic beverage brand sit near each other in feature space rather than in unrelated columns.

**Dimensionality reduction.** Embeddings were compressed with PCA before entering the models, and separately projected to three dimensions with t-SNE for interactive visualisation of how startup categories cluster.

**Scaling.** MinMax normalisation across the feature frame, then standardisation before model fitting.

## Handling Severe Class Imbalance

Every shark invests in a small minority of pitches. Training-set class balance:

| Shark | Did not invest | Invested | Minority share |
| --- | --- | --- | --- |
| Barbara Corcoran | 861 | 94 | 9.8% |
| Mark Cuban | 779 | 176 | 18.4% |
| Lori Greiner | 813 | 142 | 14.9% |
| Robert Herjavec | 868 | 87 | 9.1% |
| Daymond John | 867 | 88 | 9.2% |
| Kevin O'Leary | 869 | 86 | 9.0% |

A model that predicts "no investment" every time scores over 90% accuracy on some of these and is useless. **SMOTE** was applied to the training split only, synthesising minority examples by interpolating between neighbouring positive cases rather than duplicating them, balancing each shark's training set before fitting.

## Models

**Classification:** Logistic Regression, SVM (RBF), Random Forest, XGBoost. Chosen to span the range deliberately: a linear interpretable baseline, a kernel method for non-linear boundaries, a bagged ensemble, and a boosted ensemble.

**Regression:** Random Forest Regressor and XGBoost Regressor.

**Protocol:** 70/15/15 train, validation, test split with a fixed random state. `GridSearchCV` with 5-fold cross-validation tuned each classifier per shark. Models and scalers are serialised to disk for reuse.

## Results

### Classification

Random Forest was the best-performing model for **all six sharks** on the held-out test set.

| Shark | Best model | Test accuracy | Minority-class F1 |
| --- | --- | --- | --- |
| Barbara Corcoran | Random Forest | 0.790 | 0.27 |
| Mark Cuban | Random Forest | 0.668 | 0.21 |
| Lori Greiner | Random Forest | 0.717 | 0.24 |
| Robert Herjavec | Random Forest | 0.742 | 0.13 |
| Daymond John | Random Forest | 0.795 | 0.25 |
| Kevin O'Leary | Random Forest | 0.790 | 0.04 |

Full model comparison for a representative shark (Daymond John): Random Forest 0.795, XGBoost 0.722, SVM 0.688, Logistic Regression 0.639. The ordering held broadly across the panel, with Logistic Regression consistently weakest, in the 0.48 to 0.64 range.

**Reading these numbers honestly.** Accuracy in the high 0.70s looks strong but is carried by the majority class. The minority-class F1 column is the one that matters, and it is low throughout. The models are far better at recognising that a shark will pass than at identifying the pitches they will actually back. Daymond John's classifier reaches 0.54 recall on actual investments, the best on the panel; Kevin O'Leary's collapses to 0.06. Predicting individual investor behaviour from pre-pitch metadata is genuinely hard, and these results say so.

### Regression

| Shark | Best model | Test RMSE | Test R² |
| --- | --- | --- | --- |
| Daymond John | XGBoost | $8,524 | 0.958 |
| Kevin O'Leary | Random Forest | $27,258 | 0.891 |
| Barbara Corcoran | XGBoost | $22,571 | 0.835 |
| Lori Greiner | Random Forest | $85,037 | 0.645 |
| Mark Cuban | Random Forest | $109,627 | 0.590 |
| Robert Herjavec | Random Forest | $216,531 | -5.012 |

Random Forest took four of six, XGBoost two.

The spread here is the finding. Investment amounts are extremely long-tailed: Robert Herjavec's training mean is $27,137 against a maximum of $5,000,000. A negative R² means the model performs worse than simply predicting the mean, and that is exactly what a handful of outsized deals do to a squared-error objective on a small sample. Mean absolute error tells a very different story from RMSE on the same shark ($22,524 MAE against $216,531 RMSE), which is the signature of a few extreme errors dominating.

## Exploratory Analysis

- Per-shark industry allocation charts across all six investors, revealing that each concentrates capital in a distinctly different set of industries rather than investing uniformly
- A feature correlation matrix across the reduced feature set
- Interactive 3D t-SNE projection of the text embeddings, showing how startup categories cluster semantically, exported as standalone HTML

## Tech Stack

**Language and environment:** Python 3, Jupyter Notebook, Google Colab

**Data:** pandas, NumPy

**Machine learning:** scikit-learn (RandomForestClassifier, RandomForestRegressor, LogisticRegression, SVC, SVR, LinearRegression, GridSearchCV, train_test_split, StandardScaler, MinMaxScaler, PCA, t-SNE, metrics), XGBoost (XGBClassifier, XGBRegressor), imbalanced-learn (SMOTE)

**NLP:** OpenAI API, `text-embedding-3-small`

**Visualisation:** Matplotlib, seaborn, Plotly

**Persistence:** pickle, NumPy `.npy` arrays

## Repository Contents

```
Shark_Tank_US_Prediction_Code.ipynb    Full analysis notebook
Shark_Tank_US_Prediction_Code.html     HTML export
Shark_Tank_US_Prediction_Code.pdf      PDF export
Shark Tank US ... Report.pdf            Written project report
Shark Tank US ... Presentation.pdf      Slide deck
Project_proposal_ML.pdf                 Original project proposal
Shark Tank US ... Recording.mp4         Full project walkthrough recording
```

## Running It

```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn openai matplotlib seaborn plotly jupyter
jupyter notebook Shark_Tank_US_Prediction_Code.ipynb
```

Two things to set up before the notebook will run end to end:

**The dataset is not committed.** The notebook expects `Shark Tank US dataset.csv`, sourced from Kaggle. Place it alongside the notebook, or adjust the path in the loading cell.

**An OpenAI API key is required** for the embedding stage. The key has been stripped from the committed notebook, so supply your own via the client constructor or an environment variable. The embedding cells write `.npy` files that later cells load, so they need to run once before the modelling section.

## Limitations

Stated plainly, because they bound how far these results should be read:

- **Sample size.** 1,365 pitches split across six independent per-shark models leaves under 100 positive examples for most of the panel. That is thin for the flexible ensembles used.
- **Minority-class performance.** As the F1 column shows, the classifiers identify non-investment far more reliably than investment.
- **Outlier sensitivity in regression.** Squared-error objectives on a long-tailed target produce unstable R² across sharks, most visibly for Robert Herjavec.
- **Pre-pitch features only.** The most predictive signal in reality is the pitch itself, the negotiation, and the on-camera dynamic. None of that is in this dataset.

## Future Work

- Transformer-based encoders such as BERT applied directly to business descriptions, rather than pooled embeddings
- Log-transforming the investment amount target to stabilise regression against the long tail
- Threshold tuning and precision-recall optimisation instead of accuracy-driven model selection
- External validation against post-show funding round data
- Deployment as an interactive pitch evaluation tool

## Team

- **Raunak Choudhary** ([rc5553@nyu.edu](mailto:rc5553@nyu.edu)) · [LinkedIn](https://www.linkedin.com/in/raunak-choudhary)
- **Subhrajit Dey** ([sd5963@nyu.edu](mailto:sd5963@nyu.edu))
- **Saniya Gapchup** ([syg2021@nyu.edu](mailto:syg2021@nyu.edu))

Department of Computer Science and Engineering, New York University Tandon School of Engineering.
