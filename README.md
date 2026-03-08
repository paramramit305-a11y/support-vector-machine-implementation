# Support Vector Machines — SVC & SVR Implementation

This repository contains two Jupyter notebooks demonstrating the implementation of **Support Vector Classification (SVC)** and **Support Vector Regression (SVR)** using scikit-learn.

---

## 📂 Notebooks

| Notebook | Task | Dataset |
|---|---|---|
| `svc_implementation.ipynb` | Classification | Iris |
| `svr_implementation.ipynb` | Regression | Diabetes |

---

## 🔷 SVC — Support Vector Classification

**Dataset:** Iris (150 samples, 4 features, 3 classes)

### Workflow
1. Load the Iris dataset and inspect for null values
2. Split features and target, then perform a 70/30 train-test split (stratified)
3. Scale features using `StandardScaler`
4. Train and evaluate `SVC` with multiple kernels
5. Tune the regularization parameter `C`

### Kernels Compared
- RBF *(default)*
- Linear
- Polynomial
- Sigmoid

### Hyperparameter Tuning
Tested `C` values: `0.5, 1, 2, 3, 4, 5` with the RBF kernel.

### Evaluation Metric
- Accuracy Score
- Classification Report (precision, recall, F1-score)

---

## 🔶 SVR — Support Vector Regression

**Dataset:** Diabetes (442 samples, 10 features)

### Workflow
1. Load the Diabetes dataset
2. Perform a 70/30 train-test split
3. Scale the target variable `y` using `StandardScaler`
4. Train and evaluate `SVR` with multiple kernels
5. Tune hyperparameters using `GridSearchCV`
6. Compare with `LinearSVR`

## Technologies Used

- Python
- Scikit-learn
- NumPy
- Pandas
- Jupyter Notebook

### Kernels Compared
- RBF *(default)*
- Linear
- Polynomial

### Hyperparameter Tuning
Grid search over:
- `C`: `[1, 2, 5, 10, 50, 100]`
- `kernel`: `['rbf', 'linear']`
- `epsilon`: `[0.01, 0.1, 0.2, 0.3, 0.5]`

Best params applied to final model, then benchmarked against `LinearSVR`.

### Evaluation Metric
- R² Score (train & test)

---

## 🛠 Requirements

```bash
pip install scikit-learn pandas numpy
```

---

## 🚀 Usage

```bash
jupyter notebook svc_implementation.ipynb
jupyter notebook svr_implementation.ipynb
```
