## **Final Results End May**

### Subsampling
| Model                                 | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Selected Features (Non-zero Coefficients)                                                                          | Notes                                                                                                              |
|:--------------------------------------------|:---------|:--------------------|:-----------------|:-------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| **Lasso Logistic Regression**  | 0.5790   | 0.29                | 0.68             | 0.41               | 8 features selected: `glucose`, `currentSmoker`, `age`, `male`, `BMI`, `diabetes`, `education`, `cigsPerDay`,  Lambda=0.05 | Sub-sampling improved recall for heart disease, but resulted in lower overall accuracy & precision. Only 2 features selected.  |
| **Neural Network**             | 0.6592   | 0.29                | 0.65             | 0.40               | 128 hidden neurons, dropout rate 0.2                                                                               | |
| **Random Forest**              | 0.6309   | 0.27                | 0.67             | 0.39               | 15 selected                                                                                                        | |
| **LassoNet**                   | 0.6616   | 0.28                | 0.73             | 0.40               | All 15 features selected                                                                                           | |

---

### Class Weighting

| Model                     | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | Selected Features (Non-zero Coefficients)                                                                          | Notes                                                                                                              |
|:-----------------------------------------------|:---------|:--------------------|:-----------------|:-------------------|:-------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------|
| **Lasso Logistic Regression**                  | 0.6769   | 0.27                | 0.67             | 0.38               | 15 selected, Lambda=15                                                                                             | Class weighting improved recall for heart disease, but at the cost of precision.                                   |
| **Neural Network**                             | 0.6639   | 0.26                | 0.66             | 0.37               | 128 hidden neurons, dropout rate 0.2                                                                               | |
| **Random Forest**                              | 0.8491   | 0.50                | 0.03             | 0.06               | 15 selected                                                                                                        | |
| **LassoNet**                                   | 0.6733   | 0.23                | 0.47             | 0.31               | All 15 features selected                                                                                           | |


---

---
26.May next meeting 14.30
---


Notes:

df_majority.size= 46'032
df_minority.size= 8'240

X_sub.size=15450

