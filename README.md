# Statistical-Data-Analysis-of-Gene-Expression-Data

## Project Description
This project aims to classify cancer types (invasive vs. non-invasive) using gene expression data and statistical analysis. Various dimensionality reduction methods and machine learning models are applied to improve the classification's predictive accuracy and to identify the most influential genes in determining cancer invasiveness.

## Overview
Early detection of invasive cancers is crucial in improving patient outcomes and developing effective treatment strategies. Gene expression data provides a valuable avenue for identifying patterns that differentiate between invasive and non-invasive cancers. However, gene expression datasets are often high-dimensional, making it challenging to train accurate and efficient models.

The objective of this project is to tackle the challenge of high dimensionality through dimensionality reduction techniques and to assess the performance of various machine learning models in classifying cancer types. The key focus is to:

(1.) Reduce the feature space to improve model performance and interoperability.
(2.) Evaluate the impact of dimensionality reduction on different supervised learning algorithms.
(3.) Use resampling techniques such as K-fold cross-validation and bootstrapping to ensure the robustness of the results and avoid overfitting.
(4.) By leveraging these approaches, this project demonstrates how machine learning models can achieve high accuracy in cancer classification tasks while maintaining computational efficiency and stability.

## Project Structure
1. **Data Preprocessing**: Includes handling missing values, dimensionality reduction using t-tests, variance-based selection, and Lasso regression. 
2. **Exploratory Data Analysis**: Visualizes data distributions and gene correlations.
3. **Clustering and Dimensionality Reduction**: Applies K-means clustering, hierarchical clustering, Principal Component Analysis (PCA), and correlation-based clustering.
4. **Model Training and Evaluation**: Logistic regression, Poisson regression, K-Nearest Neighbors (KNN), Random Forest, Support Vector Machine (SVM), and XGBoost are trained with different data preprocessing techniques.

## Project Files
- **data/gene-expression-invasive-vs-noninvasive-cancer.csv**: The dataset containing gene expression values for each patient.
- **notebooks/Gene_Expression_Statistical_Data_Analysis.rmd**: Jupyter notebooks for each phase, including preprocessing, model training, and analysis.
- **report/Statistical_Data_Analysis_Report.pdf**: Detailed project report outlining methodologies, results, and conclusions.
- **README.md**: Project documentation (this file).

## Data
The dataset consists of gene expression levels from 78 patients, with 4949 gene features representing expression levels. The dataset is labeled with two classes: *Class 1: Invasive cancer*, *Class 2: Non-invasive cancer* To make the analysis more computationally feasible, a random subset of 2000 genes was selected. Missing data issues were handled and the resulting dataset was then used for dimensionality reduction and machine learning classification.

## Data Processing Steps
- **Missing Values**: Missing values imputed using KNN, with row deletion applied for rows with high missing values.
- **Dimensionality Reduction**: 
   - **Two-Sample t-Test**: Selected genes with statistically significant differences across cancer types.
   - **Variance-Based Selection**: Top genes based on high variance were retained for analysis.
   - **Lasso Regression**: Applied as a feature selection method to reduce multicollinearity.
- **Scaling and Balancing**: Scaling is not required as gene expression values are within a standard range, and the dataset has minimal class imbalance.

## Exploratory Data Analysis
- **Gene Correlations**: Explored gene correlations to identify relationships among genes and groups.
- **PCA Analysis**: Analyzed principal components to visualize class separation.
- **Clustering**: Hierarchical clustering and K-means clustering were used to observe gene grouping.

(1.) **Preprocessing**

**kNN Imputation**: Missing values were imputed using the kNN algorithm, which estimates missing values based on the closest data points in the feature space.
**Outlier Removal**: Rows with an excessive number of missing values were removed to improve the quality of the data.
**Standardization**: Gene expression levels were standardized to ensure that the models could process the data effectively.

(2.) **Dimensionality Reduction**
To tackle the high dimensionality of the gene expression data, several techniques were applied:

**Two-Sample t-Test**: A statistical method was employed to identify genes that exhibit significant differences between the invasive and non-invasive cancer groups. This reduced the number of features from 2000 to a more manageable subset.
**LASSO Regression**: L1 regularization was applied to further reduce the feature space by penalizing less important genes. This approach helps eliminate irrelevant features and improves the model's generalization.
**Variance-Based Feature Selection**: Genes with high variance were selected based on the assumption that they carry the most discriminative information. This method retained genes that contribute most to class separation.

(3.) Supervised Machine Learning
Various supervised models were applied to the reduced dataset to classify the cancer types:

**Logistic Regression**: A baseline model used for binary classification.
**K-Nearest Neighbors (KNN)**: A distance-based classifier that showed strong performance after dimensionality reduction.
**Support Vector Machines (SVM)**: A model that constructs hyperplanes to separate classes, which proved effective with high-dimensional data.
**Random Forest**: An ensemble method that uses multiple decision trees to improve accuracy and robustness.
**XGBoost**: A gradient-boosted decision tree algorithm optimized for high performance.

(4.) Validation and Resampling
To ensure that the models were not overfitting and could generalize to unseen data, the following resampling methods were used:

**K-fold Cross-Validation**: The dataset was split into 7 folds to ensure that each model was evaluated on different subsets of the data, providing a more robust estimate of model performance.
**Bootstrapping**: Multiple resamples were drawn from the original dataset to validate the stability of the model's predictions across different data distributions.

(5.) Unsupervised Learning
In addition to supervised classification, unsupervised learning techniques were applied to explore the structure of the data:

**Principal Component Analysis (PCA)**: PCA was used to reduce the dimensionality further by extracting the most important principal components. This helped visualize the separation between the cancer types.
**Clustering**: Hierarchical clustering and K-means clustering were performed to explore the natural groupings in the dataset and compare them with the known cancer labels.

## Machine Learning Models and Results
The following table shows the misclassification errors for each model under different preprocessing conditions.

| Model                    | 2001 Columns | t-Test Reduced | Top 100 t-Test Reduced | t-Test + Lasso Reduced |
|--------------------------|--------------|----------------|-------------------------|-------------------------|
| **Logistic Regression**  | 0.388        | 0.363         | 0.55                   | 0.155                   |
| **Poisson Regression**   | 0.363        | 0.428         | 0.559                  | 0.233                   |
| **LDA**                  | 0.388        | 0.28          | 0.452                  | 0.09                    |
| **K-Nearest Neighbors**  | 0.313        | 0.155         | 0.299                  | 0                       |
| **Random Forest**        | 0.336        | 0.26          | 0.298                  | 0.319                   |
| **Support Vector Machine (SVM)** | 0.33 | 0.233         | 0.415                  | 0.07                    |
| **GLM Boost**            | 0.472        | 0.44          | 0.44                   | 0.116                   |
| **XGBoost**              | 0.493        | 0.33          | 0.376                  | 0.311                   |

## Clustering Results
- **K-means Clustering**: Identified 2 optimal clusters based on silhouette analysis.
- **Hierarchical Clustering**: Generated distinct clusters using Ward linkage.

## Dependencies
- `dplyr` - Data wrangling
- `VIM` - Data imputation
- `purrr` - Functional programming
- `ggplot2` - Data visualization
- `ggcorrplot` - Correlation visualization
- `caret` - Machine learning procedures
- `DMwR` - KNN imputation
- `caTools` - Numeric data analysis
- `class` - K-Nearest Neighbors (KNN)
- `xgboost` - XGBoost model
- `mboost` - GLM Boost model
- `e1071` - Support Vector Machine (SVM)
- `randomForest` - Random Forest model
- `glmnet` - LASSO logistic regression

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Results

- **KNN** performed exceptionally well, achieving a 0 misclassification error when combined with the two-sample t-test and LASSO-reduced datasets during cross-validation. This high accuracy is attributed to the dimensionality reduction techniques that filtered out irrelevant genes. **SVM** also demonstrated strong performance, with a low misclassification rate, especially on datasets reduced by LASSO. The ability of SVM to handle high-dimensional data made it a reliable model for this task.

- **Dimensionality Reduction Impact**: The combination of the two-sample t-test and LASSO regression yielded the best results in terms of feature reduction without sacrificing accuracy. The feature set was reduced to just 29 key genes, allowing for efficient model training without overfitting.
- **Clustering and PCA**: Clustering methods, especially hierarchical clustering with complete linkage, showed a clear separation between the invasive and non-invasive classes. PCA, while useful for visualization, did not significantly improve model performance but provided insights into the underlying structure of the data.
- **Resampling Stability**: K-fold cross-validation and bootstrapping confirmed that the models trained on reduced data were robust and generalizable, with consistent performance across different data splits.

## Conclusion and Summary
This project demonstrates the effectiveness of combining dimensionality reduction techniques with machine learning models for gene expression analysis and cancer classification. It highlights the importance of feature selection in high-dimensional datasets and provides insights into model stability through robust validation techniques. The results show that K-Nearest Neighbors (KNN) with dimensionality reduction using Lasso regression achieved the best performance, with a misclassification error of 0. The t-test combined with Lasso regression proved highly effective for this gene expression dataset.

## Contributing
Contributions are welcome! If you find any issues or want to improve the code, feel free to open a pull request or create an issue in the repository.

## License
This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for more details.

For more details, please refer to the [project report](./Statistical_Data_Analysis_Report.pdf).

