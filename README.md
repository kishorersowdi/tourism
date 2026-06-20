# tourism
Tourism package purchase prediction project using XG Boost , demonstration of CI/CD integration and MLOps pipeline on GitHub to automate the end-to-end workflow.

High-Level Architecture

The solution follows an end-to-end machine learning architecture consisting of Data Ingestion, Data Cleaning, Feature Engineering, Data Transformation, Model Training, Hyperparameter Optimization, Model Evaluation, Experiment Tracking, and Deployment layers. The dataset is preprocessed using StandardScaler and OneHotEncoder through a Scikit-Learn pipeline. An XGBoost classifier is trained with class imbalance handling using scale_pos_weight and optimized using GridSearchCV with 5-fold cross-validation. MLflow is used for experiment tracking and model management. The final model is serialized using Joblib and deployed through a Streamlit web application that predicts the likelihood of a customer purchasing a tourism package.

                +------------------+
                | Tourism Dataset  |
                | (tourism.csv)    |
                +--------+---------+
                         |
                         v
                +------------------+
                | Data Cleaning    |
                |------------------|
                | Gender Fix       |
                | Marital Fix      |
                | Drop Columns     |
                +--------+---------+
                         |
                         v
                +------------------+
                | Feature Selection|
                +--------+---------+
                         |
                         v
                +------------------+
                | Train/Test Split |
                | Stratified Split |
                +--------+---------+
                         |
                         v
                +------------------+
                | Preprocessing    |
                |------------------|
                | StandardScaler   |
                | OneHotEncoder    |
                +--------+---------+
                         |
                         v
                +------------------+
                | XGBoost Model    |
                |------------------|
                | scale_pos_weight |
                | GridSearchCV     |
                +--------+---------+
                         |
                         v
                +------------------+
                | Model Evaluation |
                |------------------|
                | Accuracy         |
                | Precision        |
                | Recall           |
                | F1 Score         |
                +--------+---------+
                         |
                         v
                +------------------+
                | MLflow Tracking  |
                |------------------|
                | Parameters       |
                | Metrics          |
                | Model Artifact   |
                +--------+---------+
                         |
                         v
                +------------------+
                | tourism_model.pkl|
                +--------+---------+
                         |
                         v
                +------------------+
                | Streamlit App    |
                |------------------|
                | User Inputs      |
                | Predict Proba    |
                | Purchase Chance  |
                +------------------+
