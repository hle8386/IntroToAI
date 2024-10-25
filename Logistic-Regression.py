import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
#DataFrame
df = pd.read_csv('Breast_cancer_data.csv')

X = df[['mean_radius', 'mean_perimeter', 'mean_texture', 'mean_area', 'mean_smoothness']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Make predictions
y_pred_log_reg = log_reg.predict(X_test)

# Evaluate the model
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))

# Print the coefficients (weights)
weights = log_reg.coef_[0]
features = X.columns  # Get the feature names

# Display the feature names with their corresponding weights
for feature, weight in zip(features, weights):
    print(f"Weight for {feature}: {weight:.4f}")

predictions = log_reg.predict(X.iloc[[0]])
print(predictions)

intercept = log_reg.intercept_[0]
print(intercept)