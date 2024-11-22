# Project Overview

This project implements and evaluates different model selection techniques to compare the performance of regression models. The following methods are covered:

- **Cross-Validation (K-Fold):** Evaluates model performance by splitting the data into multiple folds.
- **Bootstrapping:** Estimates variability in performance using multiple resamples of the dataset.
- **Akaike Information Criterion (AIC):** Balances model fit and complexity by penalizing overly complex models.

These methods are tested on synthetic data using Linear and Ridge Regression models implemented from scratch.

## How to Run the Code

### Prerequisites

Install Python and required libraries: `numpy`, `matplotlib`.

### Steps to Execute

1. Open the terminal at the project folder:
2. Command: `jupyter notebook Project2.ipynb`
3. Run all cells sequentially
4. View model evaluation outputs (e.g., MSE scores, AIC values)
5. View the plots for insights into model performance and residuals

## Outputs

1. **Cross-Validation:**
    - MSE Scores: [0.2473, 0.2475, 0.2448, 0.2401, 0.2326]
    - Average MSE: 0.2425
2. **Bootstrapping:**
    - MSE Scores (first 5): [0.2408, 0.2316, 0.2475, 0.2507, 0.2412]
    - Average MSE: 0.2397
3. **AIC:**
    - Value: -6247.93
4. **Visualizations:**
    - Residual distribution plot
    - Error comparison across methods
    - Bootstrapping confidence intervals

## Key Questions and Answers

1. **Do Cross-Validation and Bootstrapping agree with AIC?**
    - Yes, all three methods largely agree in this implementation.
    - Example:
      - Cross-validation and bootstrapping indicate a lower MSE for the selected model.
      - AIC, with a low value (-6247.93), aligns with the performance metrics, confirming the model selection.

2. **In what cases might these methods fail?**
    - **Cross-Validation:** Fails for small datasets where splits lead to insufficient training data.
    - **Bootstrapping:** Assumes data is i.i.d.; fails for dependent datasets like time series.
    - **AIC:** Assumes normally distributed residuals; results may mislead for non-linear models.

3. **What could mitigate these issues?**
    - Implement stratified k-fold for imbalanced datasets.
    - Use block bootstrapping for dependent data.
    - Extend AIC to include models with corrected versions like AICc.

4. **What parameters are exposed?**
    - **Cross-Validation:**
      - `k` (number of folds), default: 5
    - **Bootstrapping:**
      - `num_iterations` (default: 100), seed for reproducibility
    - **Regression Models:**
      - Regularization parameter `alpha` (for Ridge), default: 1.0

## Additional Features

- Synthetic data generator for flexible testing
- Residual analysis and visualization tools for insights
- Extendable framework for adding new model selectors

## Compatibility

- Tested on Windows, macOS, and Linux
- Requires `numpy`, `matplotlib`, `Jupyter`
