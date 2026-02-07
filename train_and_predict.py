
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load Dataset
dataset_path = "final_soil_fertility_dataset_final.csv"
df = pd.read_csv(dataset_path)

# Relabel Fertilizer
custom_fertilizer_order = ['Organic', 'Chemical', 'Hybrid']
le_fertilizer = LabelEncoder()
le_fertilizer.classes_ = np.array(custom_fertilizer_order)
df['Fertilizer'] = le_fertilizer.transform(df['Fertilizer'])

# Encode Soil Type
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

# Feature and Target Separation
X = df.drop(['Fertilizer', 'Soil Degradation Risk', 'Soil Health Score', 'Yield'], axis=1)
Y = df[['Fertilizer', 'Soil Degradation Risk', 'Soil Health Score', 'Yield']]

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# Model Training
fertilizer_model = RandomForestClassifier(
    n_estimators=200, max_depth=20, class_weight='balanced', random_state=42
)
soil_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
)

fertilizer_model.fit(X_train, Y_train['Fertilizer'])
soil_model.fit(X_train, Y_train[['Soil Degradation Risk', 'Soil Health Score', 'Yield']])

# Evaluation
y_pred_fertilizer = fertilizer_model.predict(X_test)
y_pred_soil = soil_model.predict(X_test)

print("Fertilizer Prediction Accuracy:", accuracy_score(Y_test['Fertilizer'], y_pred_fertilizer))
print("Soil Degradation R2 Score:", r2_score(Y_test['Soil Degradation Risk'], y_pred_soil[:, 0]))
print("Soil Health Score R2 Score:", r2_score(Y_test['Soil Health Score'], y_pred_soil[:, 1]))
print("Yield MAE:", mean_absolute_error(Y_test['Yield'], y_pred_soil[:, 2]))

# Feature Importance
plt.barh(X.columns, fertilizer_model.feature_importances_)
plt.xlabel("Feature Importance")
plt.title("Fertilizer Model Feature Importance")
plt.show()

# Sample Inputs
samples = {
    "Organic": pd.DataFrame({
        'Nitrogen': [5], 'Potassium': [3], 'Phosphorus': [4],
        'Moisture': [5], 'Temperature': [40],
        'Soil Type': [le_soil.transform(['Sandy'])[0]]
    }),
    "Chemical": pd.DataFrame({
        'Nitrogen': [45], 'Potassium': [40], 'Phosphorus': [38],
        'Moisture': [45], 'Temperature': [25],
        'Soil Type': [le_soil.transform(['Sandy'])[0]]
    }),
    "Hybrid": pd.DataFrame({
        'Nitrogen': [20], 'Potassium': [15], 'Phosphorus': [20],
        'Moisture': [30], 'Temperature': [25],
        'Soil Type': [le_soil.transform(['Loamy'])[0]]
    })
}

def predict_and_display(sample_input, label):
    scaled_input = scaler.transform(sample_input)
    fertilizer_pred = fertilizer_model.predict(scaled_input)[0]
    soil_pred = soil_model.predict(scaled_input)[0]

    print(f"\n[Sample: {label}]")
    print("Predicted Fertilizer (0=Organic,1=Chemical,2=Hybrid):", fertilizer_pred)
    print("Soil Degradation Risk:", round(soil_pred[0], 2))
    print("Soil Health Score:", round(soil_pred[1], 2))
    print("Yield:", round(soil_pred[2], 2), "quintals/hectare")

for label, sample in samples.items():
    predict_and_display(sample, label)
