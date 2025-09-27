### **Final Results Mid May**

| Model                                                        | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Selected Features (Non-zero Coefficients)                                  | Notes                                                                                                              |
|:-------------------------------------------------------------|:---------|:--------------------|:-----------------|:------------------|:---------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| **Lasso Logistic Regression** (No Weighting/ No Subsampling) | 0.8538  | 0.78               | 0.05            | 0.10              | 15 selected                                                                | High accuracy for the majority class, but poor recall for heart disease (Class 1).                                 |
| **Lasso Logistic Regression** (Class Weighting)              | 0.6769  | 0.27               | 0.67            | 0.39              | 15 selected                                                               | Class weighting improved recall for heart disease, but at the cost of precision.                                   |
| **Lasso Logistic Regression** (Subsampling)                  | 0.5790  | 0.23               | 0.73            | 0.34              | 2 features selected: `glucose`, `currentSmoker`                            | Sub-sampling improved recall for heart disease, but resulted in lower overall accuracy & precision. Only 2 features selected.  |
| **Neural Network**                                           | 0.8467  | 0.43               | 0.02            | 0.04              | 128 hidden neurons, dropout rate 0.3                                       | High accuracy for the majority class, but very low recall for heart disease predictions.                           |
| **Random Forest**                                            | 0.8491  | 0.53               | 0.06            | 0.11              | 15 selected                                                                | High accuracy for the majority class, but low recall for heart disease.                                            |
| **LassoNet** (No Weighting/ No Subsampling)                  | 0.8467  | 0.45               | 0.04            | 0.07              | All 15 features selected                                                 | Integrates Lasso feature selection with neural network. Similar results to Lasso Logistic Regression.              |
| **LassoNet** (Class Weighting)                                | 0.6733  | 0.23               | 0.47            | 0.31              | All 15 features selected                                                 | Class weighting for imbalanced data, leading to improved recall but lower accuracy & precision.                                |
| **LassoNet** (Subsampling)                                    | 0.6616  | 0.25               | 0.61            | 0.36              | All 15 features selected                                                 | Subsampling approach for class imbalance, leads to better recall for heart disease but lower accuracy & precision. |


---
26.May next meeting 14.30
---
### Next Steps

1. **Deepen Understanding of LassoNet:**
   - why **LassoNet** might not be giving the desired performance. (**subsampling** necessary, or are there other techniques?)
   - alternative **settings** and **hyperparameters** to optimize
   - Study the **implementation of LassoNet**

2. **Improve Neural Network Model:**
   - Experiment with **different architectures** and fine-tune the number of **hidden layers** and **neurons**
