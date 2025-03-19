# Breast-Cancer-Prediction-Using-Machine-Learning

## Introduction
This project leverages Machine Learning (ML) techniques to enhance the accuracy of breast cancer risk assessment, diagnosis, and treatment personalization. With over 2.3 million new cases and 688,000 deaths reported in 2020, improved predictive models are crucial for better clinical outcomes.

## Key Features
- Utilizes multiple ML models for prediction.
- Compares model performance using metrics like Accuracy, AUC, Precision, F1 Score, and Time Taken.
- Provides insights into the strengths and limitations of each model.

## Machine Learning Models Implemented
- Logistic Regression
- K-Nearest Neighbours (KNN)
- Support Vector Classifier (SVC)
- Gaussian Naive Bayes
- Random Forest
- Multi-layer Perceptron (MLP)
- AdaBoost
- XGBoost

## Model Performance Overview
| Model               | 1-Fold Accuracy | 1-Fold AUC | 1-Fold Precision | 1-Fold F1 | 5-Fold Accuracy | 5-Fold AUC | 5-Fold Precision | 5-Fold F1 | Time Taken (5-Fold) |
|---------------------|-----------------|-------------|-------------------|------------|------------------|-------------|-------------------|------------|----------------------|
| Logistic Regression  | 0.9737          | 0.9980      | 0.9545            | 0.9655      | 0.9451           | 0.9980      | 0.9545            | 0.9655      | 0.74567 sec          |
| K-Nearest Neighbours | 0.9737          | 0.9980      | 0.9545            | 0.9655      | 0.9319           | 0.9902      | 0.9545            | 0.9655      | 1.04597 sec          |
| Support Vector Classifier | 0.9649     | 0.9977      | 0.9756            | 0.9524      | 0.9407           | 0.9977      | 0.9756            | 0.9524      | 1.65291 sec          |
| Gaussian Naive Bayes | 0.9737          | 0.9971      | 1.0000            | 0.9639      | 0.9341           | 0.9971      | 1.0000            | 0.9639      | 0.53282 sec          |
| Random Forest        | 0.9561          | 0.9933      | 0.9524            | 0.9412      | 0.9451           | 0.9961      | 0.9524            | 0.9412      | 1.36128 sec          |
| Multi-layer Perceptron| 0.9737         | 0.9980      | 0.9545            | 0.9655      | 0.9582           | 0.9980      | 0.9545            | 0.9655      | 6.58423 sec          |
| AdaBoost             | 0.9561          | 0.9895      | 0.9524            | 0.9412      | 0.9319           | 0.9895      | 0.9524            | 0.9412      | 0.89895 sec          |
| XGBoost              | 0.9386          | 0.9912      | 0.9500            | 0.9157      | 0.9451           | 0.9912      | 0.9500            | 0.9157      | 2.74414 sec          |

## Evaluation Metrics
- **Accuracy:** Measures overall success in predicting correct classifications.
- **F1 Score:** Balances precision and recall, useful for imbalanced datasets.
- **AUC (Area Under the ROC Curve):** Evaluates the model's ability to distinguish between classes.
- **Precision:** Measures how many positive predictions were correct.
- **Time Taken:** Measures computational efficiency of the models.

## Challenges and Ethical Considerations
- **Data Quality and Availability:** High-quality datasets are crucial but often limited in the medical domain.
- **Bias in Training Data:** Imbalanced or biased datasets can affect model fairness.
- **Privacy and Security:** Protection of sensitive medical data is essential.
- **Interpretability:** Some complex models lack transparency, impacting clinical adoption.

## Future Directions
- Enhancing ML model accuracy using advanced deep learning techniques.
- Improving feature selection to reduce computational costs.
- Integrating ML models into clinical workflows for real-time decision support.

## Conclusion
Machine learning models have demonstrated strong potential in breast cancer prediction, improving early detection and enabling personalized treatment strategies. As the field advances, addressing data bias, enhancing interpretability, and integrating ML tools into healthcare systems will be crucial for achieving optimal clinical outcomes.

