import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
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

# Initialize and train the K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions using KNN
y_pred_knn = knn.predict(X_test)

# Evaluate the KNN model
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_classification_report = classification_report(y_test, y_pred_knn)

print(f"KNN Accuracy: {knn_accuracy}")
print(f"KNN Classification Report:\n{knn_classification_report}")
