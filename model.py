import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df1 = pd.read_csv("data/train_part1.csv")
df2 = pd.read_csv("data/train_part2.csv")

df = pd.concat([df1, df2], ignore_index=True)

print("\nOriginal Dataset Shape:")
print(df.shape)




df_ml = df.copy()

numeric_columns = [
    "Bathrooms",
    "Balconies",
    "YearBuilt",
    "Latitude",
    "Longitude"
]

for column in numeric_columns:
    df_ml[column] = df_ml[column].fillna(
        df_ml[column].median()
    )




categorical_missing_columns = [
    "City",
    "Locality",
    "Furnishing",
    "Parking",
    "Facing"
]

for column in categorical_missing_columns:
    df_ml[column] = df_ml[column].fillna("Unknown")




df_ml["PropertyAge"] = 2026 - df_ml["YearBuilt"]



numerical_features = [
    "BHK",
    "Bathrooms",
    "Balconies",
    "SuperBuiltUpArea_sqft",
    "BuiltUpArea_sqft",
    "CarpetArea_sqft",
    "Floor",
    "TotalFloors",
    "YearBuilt",
    "PropertyAge",
    "AmenitiesCount",
    "Latitude",
    "Longitude"
]

categorical_features = [
    "City",
    "Locality",
    "PropertyType",
    "Furnishing",
    "Parking",
    "BuildingType",
    "Facing"
]

all_features = numerical_features + categorical_features

X = df_ml[all_features]

y = df_ml["Price_INR"]




X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Records:")
print(len(X_train))

print("\nTesting Records:")
print(len(X_test))


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)

random_forest = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("random_forest", random_forest)
    ]
)


model.fit(
    X_train,
    y_train
)

print("\nModel Training Completed.")
predictions = model.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)



print("\nMAE:")
print(f"₹{mae:,.2f}")

print("\nRMSE:")
print(f"₹{rmse:,.2f}")

print("\nR2 Score:")
print(f"{r2:.4f}")

model_filename = "real_estate_price_model.pkl"

joblib.dump(
    model,
    model_filename
)

print(f"\nModel File:")
print(model_filename)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print("\nFirst 10 Predictions:")
print(comparison.head(10))


