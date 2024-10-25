import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('Breast_cancer_data.csv')

# Encode the target labels (diagnosis: M -> 1, B -> 0)
le = LabelEncoder()
df['diagnosis'] = le.fit_transform(df['diagnosis'])

# Selecting features for classification
X = df[['mean_radius', 'mean_perimeter', 'mean_texture', 'mean_area', 'mean_smoothness']]
y = df['diagnosis']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions using the Decision Tree
y_pred_dt = dt_model.predict(X_test)

# Evaluate the Decision Tree model
dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_classification_report = classification_report(y_test, y_pred_dt)

print(f"Decision Tree Accuracy: {dt_accuracy}")
print(f"Decision Tree Classification Report:\n{dt_classification_report}")
