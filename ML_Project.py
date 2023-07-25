# Step 1: Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder


# Step 2: Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Step 3: Exploratory Data Analysis (EDA)

# Import libraries for EDA
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training data
train_data = pd.read_csv('train.csv')

# View the first few rows of the training data
train_data.head()

# Check the shape of the training data
print("Shape of training data:", train_data.shape)

# Check the data types of the columns
train_data.info()

# Check the distribution of the target variable
sns.countplot(x='Class', data=train_data)
plt.title('Distribution of Target Variable')
plt.show()

# Check the correlation between features
corr_matrix = train_data.iloc[:, 1:-1].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Drop any unnecessary columns
train_data.drop('Id', axis=1, inplace=True)

# Fill missing values, handle outliers, perform feature scaling, etc. (based on EDA observations)

# Handling Missing Values
# Identify columns with missing values
missing_columns = train_data.columns[train_data.isnull().any()]
print("Columns with missing values:", missing_columns)

# Impute missing values with the mean
train_data.fillna(train_data.mean(), inplace=True)

# Handling Outliers (Example: Z-score method)
from scipy import stats

# Identify columns with outliers (assuming numerical columns)
numerical_columns = train_data.select_dtypes(include='number').columns

# Loop through each numerical column
for column in numerical_columns:
    z_scores = stats.zscore(train_data[column])
    threshold = 3  # Set the threshold for determining outliers
    outliers = train_data[abs(z_scores) > threshold]
    print("Outliers in column", column, ":", outliers)

    # Remove outliers 
    # train_data = train_data[abs(z_scores) <= threshold]

# Feature Scaling (Example: Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train_data[numerical_columns])
train_data[numerical_columns] = scaled_data

# Confirm the changes
print(train_data.head())


# Load the test data
test_data = pd.read_csv('test.csv')

# Drop the 'Id' column from the test data
test_data.drop('Id', axis=1, inplace=True)

# Load the sample submission data
sample_submission = pd.read_csv('sample_submission.csv')

# Load the greeks data
greeks = pd.read_csv('greeks.csv')

# Confirm the columns in each dataset
print("Columns in test data:", test_data.columns)
print("Columns in train data:", train_data.columns)
print("Columns in sample submission:", sample_submission.columns)
print("Columns in greeks:", greeks.columns)

# Split the data into features (X) and target (y)
X = train_data.iloc[:, :-1]
y = train_data['Class']

# Split the data into training set and test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Confirm the shape of the training and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Step 4: Data Preprocessing


# Load the training data from train.csv
train_df = pd.read_csv('train.csv')

# Handle Missing Values
train_df.fillna(train_df.mean(), inplace=True)  # Replace missing values with mean

# Handle Outliers (example using Z-score method)
from scipy import stats
z_scores = stats.zscore(train_df.select_dtypes(include='number').iloc[:, 1:-1])  # Compute Z-scores for numerical columns
train_df = train_df[(z_scores < 3).all(axis=1)]  # Keep rows with Z-scores less than 3

# Feature Scaling (example using Min-Max scaling)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
train_df[numerical_columns] = scaler.fit_transform(train_df[numerical_columns])

  

# Feature Selection (example using correlation analysis)
correlation_matrix = train_df.iloc[:, 1:-1].corr()
# Perform correlation analysis and select relevant features based on a threshold
selected_features = correlation_matrix[correlation_matrix.abs() > 0.5].dropna(how='all', axis=1).dropna(how='all', axis=0).columns
train_df = train_df[['Id'] + list(selected_features) + ['Class']]

# Perform any other necessary data preprocessing steps based on your specific dataset and requirements

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X = train_df.iloc[:, 1:-1].values  # Features
y = train_df['Class'].values  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of train and test sets
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)


# Step 5: Creating and Evaluating Classification Models

# Initialize the classifiers
classifier1 = LogisticRegression()
classifier2 = SVC(probability=True)
classifier3 = RandomForestClassifier()

# Fit the classifiers on the training data
classifier1.fit(X_train, y_train)
classifier2.fit(X_train, y_train)
classifier3.fit(X_train, y_train)

# Make predictions on the test data
y_pred1 = classifier1.predict(X_test)
y_pred2 = classifier2.predict(X_test)
y_pred3 = classifier3.predict(X_test)

# Calculate the evaluation metrics for each classifier
accuracy1 = accuracy_score(y_test, y_pred1)
precision1 = precision_score(y_test, y_pred1)
recall1 = recall_score(y_test, y_pred1)
f1_score1 = f1_score(y_test, y_pred1)
roc_auc1 = roc_auc_score(y_test, y_pred1)

accuracy2 = accuracy_score(y_test, y_pred2)
precision2 = precision_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)
f1_score2 = f1_score(y_test, y_pred2)
roc_auc2 = roc_auc_score(y_test, y_pred2)

accuracy3 = accuracy_score(y_test, y_pred3)
precision3 = precision_score(y_test, y_pred3)
recall3 = recall_score(y_test, y_pred3)
f1_score3 = f1_score(y_test, y_pred3)
roc_auc3 = roc_auc_score(y_test, y_pred3)

# Print the evaluation metrics for each classifier
print("Classifier 1: Logistic Regression")
print("Accuracy:", accuracy1)
print("Precision:", precision1)
print("Recall:", recall1)
print("F1 Score:", f1_score1)
print("ROC AUC Score:", roc_auc1)
print()

print("Classifier 2: Support Vector Machine")
print("Accuracy:", accuracy2)
print("Precision:", precision2)
print("Recall:", recall2)
print("F1 Score:", f1_score2)
print("ROC AUC Score:", roc_auc2)
print()

print("Classifier 3: Random Forest")
print("Accuracy:", accuracy3)
print("Precision:", precision3)
print("Recall:", recall3)
print("F1 Score:", f1_score3)
print("ROC AUC Score:", roc_auc3)

# Step 6: Split the data into train and test sets
X_train, X_val, y_train, y_val = train_test_split(train_data.drop('Class', axis=1), train_data['Class'], test_size=0.2, random_state=42)

# Step 7: Create and train classification models

models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(probability=True),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

for model_name, model in models.items():
    # Preprocess training data
    X_train.fillna(X_train.mean(), inplace=True)  # Handle missing values
    X_train = pd.get_dummies(X_train)  # Convert categorical variables to numerical
    X_train = X_train.astype(float)  # Convert data types if needed

    # Convert target variable to numeric labels if needed
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)

    # Train the model
    model.fit(X_train, y_train)

    # Preprocess validation data
    X_val.fillna(X_val.mean(), inplace=True)  # Handle missing values
    X_val = pd.get_dummies(X_val)  # Convert categorical variables to numerical
    X_val = X_val.astype(float)  # Convert data types if needed

    # Convert target variable to numeric labels if needed
    y_val = label_encoder.transform(y_val)

    # Make predictions on the validation data
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_pred_proba)

    # Print the evaluation metrics
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC-ROC: {roc_auc:.4f}")
    print()

# Step 8: Making Predictions and Evaluating the Model

# Preprocess the test data
test_data_filled = test_data.fillna(test_data.mean())  # Handle missing values using mean
test_data_encoded = pd.get_dummies(test_data_filled)  # Perform one-hot encoding

# Reorder the columns of the test data to match the training data
missing_columns = set(X_train.columns) - set(test_data_encoded.columns)
for column in missing_columns:
    test_data_encoded[column] = 0  # Add missing columns with default values

test_data_encoded = test_data_encoded[X_train.columns]  # Reorder columns

# Make predictions on the test data
y_pred = classifier.predict(test_data_encoded)

# Print the predictions
print(y_pred)


# Preprocess the test data
test_data.fillna(test_data.mean(), inplace=True)  # Handle missing values using mean
test_data_encoded = pd.get_dummies(test_data)  # Convert categorical variables to numerical

# Reorder the columns of the test data to match the training data
test_data_encoded = test_data_encoded.reindex(columns=X_train.columns, fill_value=0)

# Make predictions on the test data
test_predictions = classifier.predict(test_data_encoded)

# Create a submission DataFrame
submission_df = pd.DataFrame({'Id': range(1, len(test_data) + 1), 'Class': test_predictions})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Print the first few rows of the submission DataFrame
print(submission_df.head())