#Land Cover Classification Project
#Project Overview
This project aims to generate a model that can be used to classify land cover into three key categories: Buildings, Cropland, and Woody Vegetation Cover (>60%). The goal is to develop a machine learning model that predicts land cover types based on geospatial data and provides probability scores for each category.

Dataset
The dataset includes geospatial features and land cover labels for a region undergoing rapid land-use changes. The data is divided into: Training Data (train_land_cover_assignment.csv), Test Data (test_land_cover_assignment.csv): Contains features without labels, used for predictions

Methodology

1. Data Preprocessing

Load training and test datasets.

Handle missing values by filling them with the median.

Normalize numerical features using StandardScaler to ensure uniform scaling.

Encode categorical labels using LabelEncoder for machine learning compatibility.

2. Model Training

Split training data into 80% training and 20% validation sets.

Train RandomForestClassifier for each land cover category (Buildings, Cropland, Woody Vegetation Cover).

Evaluate models using classification reports on the validation set.

3. Prediction

Apply trained models to the test dataset.

Generate probability scores for each category.

Format predictions according to the required submission format.

4. Submission

Create a submission file (submission.csv) containing the id and predicted probabilities for each land cover category.

Save results for submission.

#Installation & Requirements

Prerequisites

Ensure you have Python 3.7+ installed. Install dependencies using:

pip install -r requirements.txt

Running the Project

Navigate to the project directory:

cd Land_Cover_Classification

Run the classification script:

python src/Land_classification.py

The generated submission file will be saved as submission.csv in the project folder.

Results & Findings

The Random Forest model provided accurate predictions for each land cover class.

Feature normalization and proper encoding improved model performance.

Model evaluation showed strong classification results, with potential improvements using advanced deep learning techniques.

Recommendations

Include additional geospatial features such as NDVI (Normalized Difference Vegetation Index) to enhance predictions.

Explore feature selection methods to reduce redundant data and improve efficiency.

Contributor

Valentine Owino :Geomatics Engineer, GIS Specialist, Remote Sensing Expert
