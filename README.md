import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Features and target
X = train_df.drop("Survived", axis=1)
y = train_df["Survived"]

# Save PassengerId for submission
passenger_ids = test_df["PassengerId"]

# Drop unnecessary columns
drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]

X = X.drop(columns=drop_cols)
test_features = test_df.drop(columns=drop_cols)

# Define column types
numeric_features = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
categorical_features = ["Sex", "Embarked"]

# Numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

# Split data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Validate model
y_pred = model.predict(X_valid)

accuracy = accuracy_score(y_valid, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Predict test set
test_predictions = model.predict(test_features)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": passenger_ids,
    "Survived": test_predictions
})

submission.to_csv("submission.csv", index=False)

print("submission.csv created successfully!")

# Save trained model
joblib.dump(model, "titanic_model.pkl")

print("titanic_model.pkl saved successfully!")

