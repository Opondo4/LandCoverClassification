#..........importing libraries to perform operations on data
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from rasterio.enums import Resampling
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler, LabelEncoder

#.........loading Training data, Test data and Sample submission format
training_df = pd.read_csv("train_land_cover_assignment.csv")
test_df = pd.read_csv("test_land_cover_assignment.csv")

#.........Visualizing nature of datasets
print(training_df.columns)
print(test_df.columns)
training_df.info()
print(training_df)

#..........#Data preprocessing
print(training_df.isnull().sum())#checking for null values
training_elements = training_df.drop(columns=['subid', 'building', 'cropland', 'wcover'], errors='ignore')  # These are target labels.
training_labels = training_df[['building', 'cropland', 'wcover']] #Extracting target labels from the dataframe
print (training_labels)

#..........Fill missing values with median
training_elements.fillna(training_elements.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

#..........Normalization to ensure all numerical input features have uniform scale
scaler = StandardScaler()  # Initialization of standard scaler
training_elements_scaled = scaler.fit_transform(training_elements)
test_features_scaled = scaler.transform(test_df.drop(columns=['subid'], errors='ignore'))

#..........Encoding to convert categorical labels to numerical format to be read by model
label_encoders = {col: LabelEncoder().fit(training_labels[col]) for col in training_labels.columns}#Creates a dictionary of encoders for each label column
train_labels_encoded = np.column_stack([label_encoders[col].transform(training_labels[col]) for col in training_labels.columns])#Converts each categorical label into numeric values

#..........Train-validation split to divide the dataset into training and validation subsets.
X_train, X_val, y_train, y_val = train_test_split(training_elements_scaled, train_labels_encoded, test_size=0.2, random_state=42)

#Initialization and training of separate Random Forest Classifier for each land cover category
# Each classifier is trained separatelyto predict whether a given instance belongs to that class.
rf_models = {col: RandomForestClassifier(n_estimators=100, random_state=42) for col in training_labels.columns}
for i, col in enumerate(training_labels.columns):
    rf_models[col].fit(X_train, y_train[:, i])

#Model validation and evaluating its perfomance
y_val_pred = np.column_stack([rf_models[col].predict(X_val) for col in training_labels.columns])  # Predict on validation data
for i, col in enumerate(training_labels.columns):
    print(f"Classification Report for {col}:")  # Print evaluation metric
    print(classification_report(y_val[:, i], y_val_pred[:, i]))

# Create predictions for test set:
#predicts the probability that a test sample belongs to a given land cover category (Buildings, Cropland, Woody Vegetation Cover).
test_predictions = {col: rf_models[col].predict_proba(test_features_scaled)[:, 1] for col in training_labels.columns}
# Create submission file
submission = pd.DataFrame({'subid': test_df['subid']})  # create submission dataframe
for col in training_labels.columns:
    submission[col] = test_predictions[col]  # Add predicted probabilities
submission.to_csv("submission.csv", index=False)  # Save submission file

print("Submission file generated: submission.csv")  # Confirm completion


















