import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ==============================================================================
# Main Configuration
# ==============================================================================
DATA_FILE = 'data_core_updated.csv'
MODEL_OUTPUT_FILE = 'fertilizer_pipeline.joblib'


# ==============================================================================
# Part 1: Evaluate the Model's Performance
# ==============================================================================
print("=" * 50)
print("Part 1: Evaluating Model Performance")
print("=" * 50)

# Load the dataset
df = pd.read_csv(DATA_FILE)

# Separate features (X) and target (y)
X = df.drop('Fertilizer', axis=1)
y = df['Fertilizer']

# Split the data into training and testing sets for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data successfully split into {len(X_train)} training samples and {len(X_test)} testing samples.")

# Define the preprocessing steps for numerical and categorical features
categorical_features = ['Soil Type', 'Crop Type']
numerical_features = ['Temperature', 'Humidity', 'Soil Moisture', 'Nitrogen', 'Potassium', 'Phosphorus']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model with the best parameters we found previously
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Create the full machine learning pipeline
evaluation_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('classifier', model)])

# Train the pipeline on the training data
print("\nTraining the model for evaluation...")
evaluation_pipeline.fit(X_train, y_train)
print("Training complete.")

# Make predictions on the test data
print("\nMaking predictions on the test set...")
y_pred = evaluation_pipeline.predict(X_test)

# Calculate and display the model's accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy on Test Data: {accuracy:.4f}")

# Display the detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# ==============================================================================
# Part 2: Train the Final Model for Production
# ==============================================================================
print("\n" + "=" * 50)
print("Part 2: Training Final Model for Production")
print("=" * 50)

# For the final model, we train on the ENTIRE dataset to give it the most
# information possible. We use the same pipeline definition as before.
final_production_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                            ('classifier', model)])

print("Training the final model on all available data...")
# We use the full X and y datasets here
final_production_pipeline.fit(X, y)
print("Final model training complete.")

# Save the final, production-ready pipeline to a file
joblib.dump(final_production_pipeline, MODEL_OUTPUT_FILE)
print(f"\nProduction model successfully saved to '{MODEL_OUTPUT_FILE}'")
print("This file is now ready to be used in your web application.")