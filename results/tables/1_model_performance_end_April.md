## Initial Results

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Notes                                                                                                                   |
|:------|:---------|:--------------------|:-----------------|:------------------|:------------------------------------------------------------------------------------------------------------------------|
| Lasso Logistic Regression | 0.8526 | 0.64 | 0.07 | 0.13 | High accuracy, poor recall for positives. Predicted some sick patients, but missed most                                 |
| Simple Neural Network | 0.8479 | 0.00 | 0.00 | 0.00 | Only predicts majority class. Did not predict any sick patients at all                                                  |
| Random Forest | 0.8432 | 0.25 | 0.02 | 0.03 | Slightly better for positives, still poor. Predicted very few sick patients                                             |



---

## Next Steps

### 1. Class Imbalance (and Gridsearch)
- `class_weight='balanced'` for Logistic Regression and Random Forest models.
- Adjust the loss function for Neural Network

### 2. Retrain Models
- Retrain and Compare results again: Accuracy, Precision, Recall, F1-Score.
- Focus on improving Recall and F1-Score for heart disease class (class 1).

### 3. Old (imbalanced) vs New (balanced) model performances

### 4. Try LassoNet Implementation (and compare)
- https://github.com/lasso-net/lassonet

---
