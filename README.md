
# Live dashboard: https://224104c-rainfall-dashboard.streamlit.app

# Note on AI Use

I selected the project topic and dataset, designed and ran the notebook, and reviewed the results myself. I used Claude (an AI assistant) to help improve code readability, clean up the code, identify and fix coding errors, and generate some code sections when I found them difficult to implement independently. I tested the generated and modified code, checked the outputs, and wrote the analysis and interpretations in my own words.

# Goal

Forecast next month's rainfall in Ratnapura and compare ensemble methods: bagging (Random Forest), boosting (XGBoost, LightGBM, CatBoost), stacking and voting, and two hybrids (RF + SARIMA, RF + ANN).

# Data

Open-Meteo historical weather (reanalysis, i.e. modelled data, not station readings), 2010 to 2025, aggregated to monthly values.

# Interpretation

## Overall

All 11 models beat the seasonal naive baseline (RMSE 191.4, R² -0.07), so each learned something beyond "same month last year". The best model was the ANN (RMSE 127.7, R² 0.52), followed by CatBoost (131.8). Monthly rainfall is noisy, so even the best model explains only about half the variation. The test set has only 48 months, so gaps between neighbouring models are small and I treat them as close, not decisive.

## Bagging vs boosting

CatBoost (boosting, RMSE 131.8) was the best single tree model. Random Forest (bagging, 149.2), XGBoost (147.3) and LightGBM (158.2) were worse. With about 130 training months, CatBoost's built-in regularisation probably helped it avoid overfitting. So boosting did not beat bagging as a rule; it depended on the implementation.

## Hyperparameter tuning

Tuning the Random Forest with a time-ordered cross-validation (best: max_depth None, min_samples_leaf 5, n_estimators 500) gave RMSE 149.1, essentially the same as the default Random Forest (149.2). With only about 130 rows there is little for tuning to improve, so the default settings were already near the best this model can do here.

## Voting and stacking

Voting (RMSE 143.4) beat Random Forest, XGBoost and LightGBM but not CatBoost (131.8). Stacking (139.4) also did not beat CatBoost. The stacking weights show why: the meta-learner gave CatBoost a weight of 0.568 and the other three models 0, so the stack was really a scaled and shifted CatBoost. The four base models are all tree-based and make similar errors, so there is little for a meta-learner to combine, and it learned its weights from only about 100 out-of-fold predictions.

## RF + SARIMA hybrid

The hybrid (143.8) beat RF alone (149.2) but not SARIMA alone (138.8). SARIMA already captured the seasonal pattern, so the leftover residuals were mostly random noise, and training RF on them added variance. I used SARIMA instead of plain ARIMA because rainfall repeats every year.

## RF + ANN hybrid

The hybrid (133.2) beat RF alone but not the ANN alone (127.7). Averaging a stronger and a weaker model gives a result in between.

## Why the ANN did well

It is an average of 5 small networks, which reduces variance like bagging does. Rainfall also follows a smooth yearly cycle, which suits a smooth model, while trees approximate it with steps. These reasons are likely, not proven, and the result may not hold on a larger dataset.

## Feature importance

In the Random Forest, rain_mm_lag12 (rainfall 12 months earlier) was by far the most important feature, followed by rain_mm_lag1 and rain_mm_lag3. In XGBoost, rain_mm_lag12 was also first, followed by month_cos, rain_mm_lag1 and month_sin. Both models rely on rainfall from a year earlier, which shows the yearly seasonal cycle drives the forecast, and XGBoost also uses the month encoding directly.

## Tabular vs image/text data

Tree ensembles work very well on small tabular data like this, need little preprocessing and are fast to train. On images and text, deep neural networks usually do better because they learn structure (pixels, word order) that trees cannot see.

## Limitations

Open-Meteo is reanalysis (modelled) data, not station readings. It is one location, 192 months, and one train/test split; rolling cross-validation would be more robust. Extreme events such as the 2016 floods cannot be predicted from lagged values. The ANN hit its iteration limit (ConvergenceWarning), so its result comes from the final iteration.
