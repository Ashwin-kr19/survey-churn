# Project Report: Churn Prediction System using FastAPI and Machine Learning

---

## 1. Project Overview:
The objective of the project is to develop a churn prediction system that works across multiple datasets. It aims to create a predictive model capable of identifying customers likely to leave a service. This project also includes the implementation of a FastAPI backend for handling model training and predictions, and an added focus on the explainability of the model’s predictions using techniques such as SHAP and LIME.

---

## 2. Project Workflow:

1. **Data Preprocessing:**
   - **Data Input:** The system accepts datasets through an upload feature, which allows for multiple datasets to be used. It supports CSV files that are uploaded via an API or web interface.
   - **Data Cleaning and Preprocessing:** It includes handling missing values, encoding categorical variables using OneHotEncoder, and scaling numerical features using StandardScaler.

2. **Model Training:**
   - The user provides the target column (e.g., `churn`) for the dataset.
   - The RandomForestClassifier model is chosen due to its balance between performance and interpretability.
   - Categorical columns with a high number of unique values are dropped to avoid overfitting.
   - The pipeline splits the data into training and validation sets (80/20 split) and trains the model on the training data.
   - Validation accuracy is displayed, and the trained model is stored for future predictions.

3. **Prediction and Evaluation:**
   - After training, the model can predict on a test dataset uploaded via the API.
   - The system validates that the test dataset has the same structure as the training dataset.
   - Predictions are returned to the user along with a pie chart showing the distribution of predicted classes (churn vs non-churn).
   - Model evaluation includes standard metrics like accuracy, precision, recall, and AUC-ROC.

4. **Backend API:**
   - The project incorporates a FastAPI backend to handle data upload, model training, and prediction.
   - Two key endpoints are `/train` for training the model and `/predict` for making predictions on new datasets.
   - The backend provides a flexible, scalable solution for churn prediction across different datasets with varying features.

---

## 3. Technologies Used:

- **Backend:** FastAPI for building a robust and scalable API.
- **Frontend:** Jinja2 templates for rendering pages and matplotlib for plotting charts.
- **Machine Learning:** 
  - Preprocessing: OneHotEncoder for categorical features, StandardScaler for numerical features.
  - Model: RandomForestClassifier from scikit-learn.
  - SHAP and LIME for explainability (future implementation).

- **Other Libraries:** Pandas for data manipulation, Matplotlib for visualization, scikit-learn for model building and evaluation.

---

## 4. Key Features:

1. **Model Flexibility:**
   - Handles different datasets by processing both numerical and categorical features.
   - Supports multiple datasets with varying feature sets, making it adaptable for different businesses.

2. **Explainability:**
   - Although not fully implemented in the current version, the project aims to incorporate SHAP and LIME to explain individual predictions and global model behavior, providing transparency into how the model arrives at decisions.

3. **Web Interface:**
   - Users can interact with the system via a web interface to upload data, train models, and get predictions.

4. **CORS Enabled:**
   - The API has CORS middleware to ensure cross-origin requests can be made from external domains, making the API usable by multiple frontends.

---

## 5. Challenges and Solutions:

- **Handling of Different Datasets:** The project had to handle datasets with varying features while ensuring that the target column remains consistent. This was managed by dynamically selecting and transforming relevant features during preprocessing.
- **Categorical Columns with High Cardinality:** To avoid overfitting due to large categorical columns, the system drops columns that have unique values exceeding a specified threshold.
- **Model Explainability:** Integrating SHAP and LIME explanations for model predictions is a planned enhancement. These techniques will help in understanding feature importance and individual prediction rationales.

---

## 6. Future Enhancements:

- **Model Interpretability:** Implement global and local explainability techniques using SHAP, LIME, or surrogate models. This will make the model more transparent and allow users to see why specific customers are predicted to churn.
- **Extending for Multiple Models:** Add support for additional machine learning algorithms (e.g., XGBoost, Logistic Regression) to see how they perform on different datasets.
- **Advanced Visualizations:** Improve result reporting by adding more advanced visualizations, such as feature importance charts, confusion matrices, and more interactive dashboards.

---

## 7. Conclusion:
This project successfully implements a scalable churn prediction system using FastAPI and machine learning techniques. It provides a flexible pipeline that can handle various datasets, offers a web interface for interaction, and delivers valuable insights into customer churn. Future enhancements will focus on explainability and model interpretability to give users deeper insights into predictions.

---
