# Student Performance Prediction

Exploratory analysis and machine learning on the UCI **Student Performance** dataset (Math + Portuguese courses) to predict students' final grade (`G3`) and whether they pass or fail.

## Dataset

- **Source:** [UCI Machine Learning Repository – Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Size:** 1,044 records (395 Math + 649 Portuguese), 33 attributes
- **Attributes:** demographics, family background, study habits, alcohol use, health, absences, and period grades `G1`, `G2`, `G3`
- **Target:** `G3` (final grade, 0–20)
- **Note:** 382 students appear in both courses, once per course with that subject's grades.

The combined dataset is provided as `student-combined.arff` (Weka-compatible). A `course` attribute marks whether each row is Math or Portuguese.

## What the project does

1. Loads the ARFF file and checks data quality
2. Exploratory data analysis with saved plots (distributions, correlations, group comparisons)
3. Preprocessing: one-hot encoding, scaling, train/test split
4. **Regression:** predicts `G3` as a number (Linear, Ridge, Random Forest, Gradient Boosting)
5. **Classification:** predicts Pass (`G3 >= 10`) or Fail (Logistic Regression, Decision Tree, Random Forest, k-NN)
6. Runs both tasks **with and without** the earlier grades `G1` and `G2` to show how much they drive the predictions

## Getting started

```bash
git clone <your-repo-url>
cd <your-repo-name>
pip install pandas numpy matplotlib scikit-learn
python student_performance.py
```

Plots are saved to `plots/`, and metrics are saved to `regression_results.csv` and `classification_results.csv`.

## Results

| Task | With G1 & G2 | Without G1 & G2 |
|---|---|---|
| Regression: best test R² (Gradient Boosting) | 0.84 | 0.27 |
| Classification: best test accuracy | ~0.90 | ~0.79 |

**Key findings**

- `G1` and `G2` are by far the strongest predictors of `G3` (correlations of about 0.81 and 0.91).
- Without them, prediction drops sharply. The most useful remaining features are `failures` and `absences`.
- Students with `G3 = 0` (53 records, likely dropouts) are outliers that hurt regression accuracy.
- Several categorical attributes are heavily imbalanced (e.g. `higher` is 91% "yes"), so they carry little information.

## Repository structure

```
├── student_performance.py     # main program
├── student-combined.arff      # combined dataset (Math + Portuguese)
├── plots/                     # generated figures
├── regression_results.csv     # generated metrics
├── classification_results.csv # generated metrics
└── README.md
```

## Using the data in Weka

Open `student-combined.arff` in Weka. `G3` is the last attribute and is used as the class by default. For classifiers such as J48, apply the `Discretize` filter to `G3` first.

## Tech stack

Python, pandas, NumPy, matplotlib, scikit-learn

## Citation

P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.* In Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008), pp. 5–12, Porto, Portugal, April 2008.
