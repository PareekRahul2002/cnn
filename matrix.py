import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Assuming you have your test data and trained model
X_test, y_test = load_and_preprocess_test_data()
trained_model = load_model('final_super_resolution_model.h5')

# Make predictions on the test data
y_pred = trained_model.predict(X_test)

# Flatten and discretize the arrays for classification
y_test_flat = np.round(y_test).astype(int).reshape(-1)
y_pred_flat = np.round(y_pred).astype(int).reshape(-1)

# Use label encoding for unique classes
label_encoder = LabelEncoder()
label_encoder.fit(np.unique(np.concatenate((y_test_flat, y_pred_flat))))
y_test_encoded = label_encoder.transform(y_test_flat)
y_pred_encoded = label_encoder.transform(y_pred_flat)

# Calculate accuracy
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f'Accuracy: {accuracy}')

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
print(f'Confusion Matrix:\n{conf_matrix}')

# Generate classification report with precision, recall, and F1-score
class_report = classification_report(y_test_encoded, y_pred_encoded)
print(f'Classification Report:\n{class_report}')
