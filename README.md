# FINAL REPORT

## Early Identification of At-Risk Students Using Machine Learning

**Course:** IT3091 – Machine Learning  
**Track:** Education Guided Data Track  
**Group ID:** 2026-AI-53  
**Dataset:** UCI Student Performance Dataset  
**GitHub Repository:** https://github.com/harini281/ml_assignment

---

## 1. Introduction

Academic underperformance can affect students' educational progress and future opportunities. Identifying students who may be at risk of poor academic performance can help educational institutions provide timely academic support.

This project, titled **Early Identification of At-Risk Students Using Machine Learning**, investigates the use of supervised machine learning classification techniques to identify students at risk of academic underperformance.

The project uses student performance data to develop and evaluate machine learning models. The models are compared using accuracy, precision, recall, and F1-score to understand how effectively they identify at-risk students.

The main objective is to explore whether machine learning can support the early identification of students who may benefit from additional academic assistance.

## 2. Problem Statement

The dataset contains students with different academic outcomes. Identifying at-risk students is challenging because students who require support represent a smaller proportion of the dataset.

A model that predicts most students as not at risk may achieve high accuracy while failing to identify students who actually need help.

Therefore, this project focuses not only on overall prediction accuracy but also on the model's ability to identify at-risk students.

## 3. Project Objectives

The objectives of this project are:

1. To explore the UCI Student Performance Dataset.
2. To prepare the dataset for machine learning classification.
3. To create a binary target representing academic risk.
4. To develop Support Vector Machine (SVM) and Gradient Boosting classification models.
5. To tune model hyperparameters using cross-validation.
6. To evaluate and compare model performance using multiple metrics.
7. To identify the strengths and limitations of the evaluated models.

## 4. Dataset Description

The project uses the UCI Student Performance Dataset, which contains information about students and their academic and personal characteristics.

The dataset contains **649 student records**. The target variable is derived from the final grade, G3, which ranges from 0 to 20.

### 4.1 Target Variable

The final grade was converted into a binary classification target:

- **At-Risk:** G3 less than 10.
- **Not-At-Risk:** G3 greater than or equal to 10.

This creates a classification problem in which the models predict whether a student belongs to the at-risk or not-at-risk category.

### 4.2 Class Distribution

The target distribution is:

| Category | Number of students | Percentage |
|---|---:|---:|
| At-Risk | 100 | 15.41% |
| Not-At-Risk | 549 | 84.59% |
| **Total** | **649** | **100%** |

The dataset is imbalanced because the not-at-risk category contains considerably more students than the at-risk category.

This imbalance is important when evaluating model performance, as accuracy alone may not indicate how effectively the model identifies at-risk students.

## 5. Data Preparation and Preprocessing

The following preparation steps were implemented in the notebook.

### 5.1 Target Creation and Feature Selection

The final grade, G3, was used to create the target variable. G3 was excluded from the input features because using the same grade to predict the risk label would result in target leakage.

The earlier grades G1 and G2 were also excluded from the input features to investigate risk identification without directly relying on those previous grades.

The resulting feature set contained 30 input features: 17 categorical features and 13 numerical features.

### 5.2 Training and Testing Split

The dataset was divided into training and testing sets using an 80:20 split.

- Training set: 519 students.
- Testing set: 130 students.
- Random state: 42.
- Stratification: Used to preserve the target class proportions across the two sets.

The training set was used to develop and tune the models, while the test set was used to evaluate their predictive performance.

### 5.3 Categorical Feature Encoding

Categorical features were transformed using `OneHotEncoder` with `handle_unknown="ignore"`.

This converts categorical values into numerical representations that can be processed by machine learning algorithms. It also allows the transformer to handle previously unseen categories.

### 5.4 Numerical Feature Scaling

Numerical features were standardized using `StandardScaler`.

Scaling is particularly relevant to SVM because the scale of input features can influence the model's decision boundary.

### 5.5 Preprocessing Pipeline

A `ColumnTransformer` was used to apply the appropriate transformations to categorical and numerical features. The preprocessing steps were integrated with the classifiers through scikit-learn pipelines.

This approach ensures that the transformations are fitted on the training data rather than on the complete dataset before splitting, reducing the risk of data leakage.

## 6. Machine Learning Models

Two supervised classification algorithms were investigated: Support Vector Machine and Gradient Boosting.

### 6.1 Support Vector Machine (SVM)

SVM is a supervised learning algorithm that attempts to find a decision boundary separating different classes.

A default SVM model was first trained and evaluated. A second SVM model was then developed using hyperparameter tuning and class weighting.

The tuned model used an RBF kernel, C = 0.1, and balanced class weights.

### 6.2 Gradient Boosting

Gradient Boosting is an ensemble learning method that combines multiple decision trees in a sequential process. Each new tree attempts to improve the predictions of the existing ensemble.

A default Gradient Boosting model was trained and evaluated. A tuned version was subsequently developed by searching different values for the number of estimators, learning rate, and maximum tree depth.

The best configuration identified during tuning used 50 estimators, a learning rate of 0.2, and a maximum depth of 3.

## 7. Hyperparameter Tuning

GridSearchCV was used to explore alternative model configurations using five-fold cross-validation.

The tuning process used macro recall as its scoring metric. Macro recall calculates recall separately for each class and averages the results, giving both classes equal importance despite the class imbalance.

### 7.1 SVM Search Space

The SVM search considered:

- C values: 0.1, 1, and 10.
- Kernels: linear and RBF.
- Class weights: default and balanced.
- Cross-validation: five folds.

The best configuration was:

`C = 0.1, kernel = RBF, class_weight = balanced`

The best cross-validation macro recall was approximately 0.788.

### 7.2 Gradient Boosting Search Space

The Gradient Boosting search considered:

- Number of estimators: 50, 100, and 200.
- Learning rate: 0.05, 0.1, and 0.2.
- Maximum depth: 1, 2, and 3.
- Cross-validation: five folds.

The best configuration was:

`n_estimators = 50, learning_rate = 0.2, max_depth = 3`

The best cross-validation macro recall was approximately 0.714.

The cross-validation scores were used to select model configurations. Final test metrics were then used to compare the selected models on the held-out test set.

## 8. Model Evaluation Metrics

Four evaluation metrics were used.

**Accuracy:** The proportion of all predictions that were correct.

**Precision:** The proportion of students predicted as at risk who were actually at risk.

**Recall:** The proportion of actual at-risk students correctly identified by the model.

**F1-score:** The harmonic mean of precision and recall, balancing the two measures.

Recall is particularly important for this project because failing to identify an at-risk student may mean that the student misses an opportunity for additional support. However, precision also matters because a large number of false alarms may place an unnecessary burden on support staff.

## 9. Model and Method Comparison

The following results were obtained from the model evaluations recorded during development.

| Model | Accuracy | At-Risk Precision | At-Risk Recall | F1-score |
|---|---:|---:|---:|---:|
| Default SVM | 83.08% | 37.50% | 15.00% | 21.43% |
| Tuned SVM | 65.38% | 23.40% | 55.00% | 32.84% |
| Default Gradient Boosting | 78.46% | 25.00% | 20.00% | 22.22% |
| Tuned Gradient Boosting | 78.46% | 27.78% | 25.00% | 26.32% |

### 9.1 Default SVM

The default SVM achieved the highest accuracy at 83.08%. However, its recall for the at-risk class was only 15%. This indicates that it identified only a small proportion of the students who were actually at risk.

Therefore, its high accuracy should not be interpreted as evidence that it is the best model for the project's main objective.

### 9.2 Tuned SVM

The tuned SVM achieved 55% recall for the at-risk class, compared with 15% for the default SVM. Its F1-score also increased from 21.43% to 32.84%.

However, its accuracy decreased to 65.38%, and its precision was 23.40%. This indicates that the model identified more at-risk students but also generated more false-positive predictions.

### 9.3 Default Gradient Boosting

The default Gradient Boosting model achieved 78.46% accuracy and 20% recall for the at-risk class.

Although its overall accuracy was reasonably high, its ability to identify at-risk students remained limited.

### 9.4 Tuned Gradient Boosting

The tuned Gradient Boosting model maintained 78.46% accuracy while improving recall from 20% to 25%. Its precision increased from 25% to 27.78%, and its F1-score increased from 22.22% to 26.32%.

These results show an improvement over the default Gradient Boosting model on the reported precision, recall, and F1 metrics.

### 9.5 Overall Comparison

Among the four evaluated models, the tuned SVM achieved the highest at-risk recall and F1-score. It may therefore be the most promising candidate among these experiments when identifying at-risk students is the primary objective.

Nevertheless, its lower precision and accuracy must be considered. The results also show that no single metric is sufficient to judge a model's usefulness for this problem.

The tuned Gradient Boosting model provided a more moderate improvement over its default version, while the default SVM prioritized overall accuracy but identified relatively few at-risk students.

## 10. Important Decisions and Rationale

The main modelling decisions were:

| Decision | Reason |
|---|---|
| Convert G3 into a binary target | To frame the task as at-risk classification. |
| Exclude G3 from input features | To prevent target leakage. |
| Exclude G1 and G2 | To investigate classification without directly using earlier grade values. |
| Use a stratified train-test split | To preserve class proportions in both sets. |
| Apply one-hot encoding | To represent categorical features numerically. |
| Standardize numerical features | To put numerical features on comparable scales. |
| Use preprocessing pipelines | To reduce preprocessing leakage. |
| Compare accuracy, precision, recall, and F1-score | To assess both overall correctness and at-risk detection. |
| Tune with macro recall | To give both classes equal weight during model selection. |
| Test class-balanced SVM | To address the underrepresented at-risk class. |

## 11. Limitations

The project has several limitations.

First, the dataset is imbalanced, with only 100 of the 649 students categorized as at risk.

Second, the classification target is derived from the final grade. Although G3 is excluded from the input features, this definition identifies risk using the final outcome rather than a separately validated early-warning label.

Third, the reported test results show a trade-off between recall and precision. In particular, the tuned SVM identifies more at-risk students but also generates more false positives.

Finally, the results are based on this dataset and should not automatically be generalized to every school or student population.

## 12. Conclusion

This project investigated the use of Support Vector Machine and Gradient Boosting classifiers to identify students at risk of academic underperformance using the UCI Student Performance Dataset.

The workflow included target creation, feature selection, stratified data splitting, preprocessing, model training, hyperparameter tuning, and performance comparison.

The results demonstrated that model selection depends on the project's objective. The default SVM achieved the highest accuracy, while the tuned SVM achieved the highest recall and F1-score for the at-risk class among the four evaluated models.

The findings highlight the importance of evaluating class-specific performance when developing models for imbalanced educational data. Further validation would be necessary before using such a model to guide real educational interventions.

## 13. Project Repository

The project notebook and code are maintained in the following GitHub repository:

https://github.com/harini281/ml_assignment

The notebook contains the data preparation, model development, tuning, evaluation, and comparison work completed for this project.
