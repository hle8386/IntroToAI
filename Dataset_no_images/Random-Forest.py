import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Breast_cancer_data.csv')

X = df[['mean_radius', 'mean_perimeter', 'mean_texture', 'mean_area', 'mean_smoothness']]
y = df['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf.predict(X_test)

# Evaluate the model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

# Feature Importances
importances = rf.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f'Feature: {feature}, Importance: {importance}')

predictions = rf.predict(X.iloc[[0]])
print(predictions)