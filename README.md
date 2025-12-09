# 📊 Digital Marketing Conversion Prediction

This project analyzes a synthetic digital marketing dataset and builds a predictive model to estimate the probability that a customer will convert (`Conversion = 1`). The data includes demographics, campaign characteristics, and engagement metrics for 8,000 customers, with the goal of understanding which factors drive conversions and how well simple machine learning models can capture these patterns.

## 📜 Project Overview

The dataset contains 20 columns describing customers and their interactions with digital marketing campaigns: age, gender, income, campaign channel (Email, Social Media, SEO, PPC, Referral), campaign type (Awareness, Consideration, Conversion, Retention), ad spend, click and conversion rates, website engagement, loyalty information, and a binary `Conversion` outcome. The data is synthetic, clean (no missing values), and suitable for educational analysis and modeling.

This repository walks through exploratory data analysis (EDA), statistical tests, and a baseline classification pipeline to predict whether a customer will convert based on these features.

## 🎯 Objectives

- Explore how **categorical features** (`Gender`, `CampaignChannel`, `CampaignType`) relate to conversion using cross-tabulations, barplots, and a chi-square test of independence.
- Analyze **numeric features** (e.g., `AdSpend`, `ClickThroughRate`, `ConversionRate`, `WebsiteVisits`, `TimeOnSite`, `EmailOpens`, `LoyaltyPoints`) with KDE plots, boxplots, and summary statistics (mean, median, standard deviation).
- Use **t-tests** to check whether numeric features have significantly different means between converters and non‑converters.
- Build a simple supervised learning baseline to predict `Conversion`, and interpret which variables are most important for the model.

## 🔍 Data Description

- **Rows:** 8,000 customers  
- **Columns (examples):**
  - `CustomerID` – unique identifier (dropped for modeling)  
  - `Age`, `Income` – demographic information  
  - `Gender` – `Male`, `Female`  
  - `CampaignChannel` – `Email`, `Social Media`, `SEO`, `PPC`, `Referral`  
  - `CampaignType` – `Awareness`, `Consideration`, `Conversion`, `Retention`  
  - `AdSpend` – campaign spend in USD  
  - `ClickThroughRate`, `ConversionRate` – historical engagement metrics  
  - `WebsiteVisits`, `PagesPerVisit`, `TimeOnSite`, `SocialShares`, `EmailOpens`, `EmailClicks`  
  - `PreviousPurchases`, `LoyaltyPoints`  
  - `Conversion` – binary target (0 = no conversion, 1 = conversion)

Columns like `AdvertisingPlatform` and `AdvertisingTool` are constant and are dropped as non-informative.

## 🧪 Analysis Approach

### Exploratory Data Analysis

- **Categorical vs Conversion**
  - Grouped barplots for `Gender`, `CampaignChannel`, `CampaignType` by conversion status.
  - Chi-square tests of independence to assess whether conversion rates differ significantly across categories (e.g., `CampaignType = Conversion` shows higher conversion rate than other types).

- **Numeric Features**
  - For each numeric feature: KDE + boxplot to inspect distribution, spread, and outliers.
  - Summary tables with mean, median, and standard deviation for each feature.
  - Pearson correlation matrix for all numeric features and `Conversion` to check linear relationships and multicollinearity.

- **t-tests for Numeric vs Conversion**
  - Welch two-sample t-tests comparing the mean of each numeric feature between non-converters and converters.
  - p-value barplot to visualize which numeric features exhibit statistically significant mean differences.

### Modeling (Baseline)

- **Preprocessing**
  - Drop ID and constant columns.
  - One-hot encode categorical variables (`Gender`, `CampaignChannel`, `CampaignType`).
  - Optionally standardize numeric features for linear models.

- **Models**
  - Logistic Regression as an interpretable baseline classifier.
  - Optionally tree-based models (e.g., Random Forest, Gradient Boosting) to capture non-linear effects.

- **Evaluation**
  - Train/validation split stratified on `Conversion`.
  - Metrics such as accuracy, confusion matrix, and ROC–AUC.

## 🔧 Tech Stack

- Python (pandas, NumPy) for data manipulation and summary statistics  
- Matplotlib & Seaborn for visualizations (KDEs, boxplots, barplots, heatmaps)  
- SciPy and scikit-learn for statistical tests and baseline models  
- Jupyter Notebook for interactive analysis and documentation  

## 📂 Project Structure

- `digital_marketing_campaign_dataset.csv` – main dataset used for analysis  
- `Digital_Marketing_Conversion.ipynb` – notebook containing EDA, statistical tests, and modeling pipeline  
- `README.md` – high-level project description 

This project is primarily educational: it demonstrates how to go from raw tabular marketing data to an interpretable predictive model using standard data science tools and statistical reasoning.
