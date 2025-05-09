# Preprocessing and Decision Tree

# import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import SCIN dataset 
dataSet = pd.read_csv('CSVs/metadata.csv')

# Dropping the following features
dataSet = dataSet.drop(['isic_id','patient_id','lesion_id','diagnosis_confirm_type','diagnosis',
                        'attribution','copyright_license','image_type','acquisition_day','concomitant_biopsy',
                        'anatom_site_special','mel_class','mel_thick_mm','mel_ulcer','nevus_type',
                        'diagnosis_1','diagnosis_2','diagnosis_3','diagnosis_4','diagnosis_5', 'diagnosis_confirm_type'], axis=1)

# Drop rows where fitzpatrick_skin_type is missing
dataSet = dataSet.dropna(subset=['fitzpatrick_skin_type'])

# Replacing missing values in 'dermoscopic_type' with 'contact non-polarized'
dataSet['dermoscopic_type'] = dataSet['dermoscopic_type'].replace(np.NaN, 'contact non-polarized')

# Make age categorical 
bins = [0, 30, 50, 70, 100]  # Age groups
labels = ['Young', 'Middle-aged', 'Senior', 'Elderly']
dataSet['age_category'] = pd.cut(dataSet['age_approx'], bins=bins, labels=labels)
dataSet = dataSet.drop(columns=['age_approx'])  # Drop original age column

# Make size categorical 
bins = [0, 3, 6, 10, float('inf')]  # Size groups
labels = ['Small', 'Medium', 'Large', 'Very Large']
dataSet['size_category'] = pd.cut(dataSet['clin_size_long_diam_mm'], bins=bins, labels=labels)
dataSet = dataSet.drop(columns=['clin_size_long_diam_mm'])  # Drop original size column

# Encode categorical variables
encoder = LabelEncoder()
categorical_columns = ['anatom_site_general', 'dermoscopic_type',
                       'fitzpatrick_skin_type', 'sex', 'age_category', 'size_category']
for col in categorical_columns:
    dataSet[col] = encoder.fit_transform(dataSet[col])

# Encode target variable
target_encoder = LabelEncoder()
dataSet['benign_malignant'] = target_encoder.fit_transform(dataSet['benign_malignant'])

# Building the Decision Tree
x = dataSet.drop('benign_malignant', axis=1)
y = dataSet['benign_malignant']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = tree.DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy, precision, recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f'Decision Tree Accuracy: {accuracy:.2f}')
print(f'Decision Tree Precision: {precision:.2f}')
print(f'Decision Tree Recall: {recall:.2f}')

class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
class_names = list(class_mapping.keys())

# Visualize the decision tree
plt.figure(figsize=(30,12))
tree.plot_tree(clf, feature_names=x.columns, class_names=class_names, filled=True)
plt.show()
