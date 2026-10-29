import pandas as pd

df1 = pd.read_csv("data/train_part1.csv")
df2 = pd.read_csv("data/train_part2.csv")

df = pd.concat([df1, df2], ignore_index=True)
df_model = df.copy()

df_model["Bathrooms"] = df_model["Bathrooms"].fillna(
    df_model["Bathrooms"].median()
)

df_model["Balconies"] = df_model["Balconies"].fillna(
    df_model["Balconies"].median()
)

df_model["YearBuilt"] = df_model["YearBuilt"].fillna(
    df_model["YearBuilt"].median()
)

df_model["Latitude"] = df_model["Latitude"].fillna(
    df_model["Latitude"].median()
)

df_model["Longitude"] = df_model["Longitude"].fillna(
    df_model["Longitude"].median()
)

df_model["City"] = df_model["City"].fillna("Unknown")

df_model["Locality"] = df_model["Locality"].fillna("Unknown")

df_model["Furnishing"] = df_model["Furnishing"].fillna("Unknown")

df_model["Parking"] = df_model["Parking"].fillna("Unknown")

df_model["Facing"] = df_model["Facing"].fillna("Unknown")



df_model["Price_per_sqft"] = (
    df_model["Price_INR"] / df_model["CarpetArea_sqft"]
)


print("\nNew Feature: Price_per_sqft")

print("\nFirst 10 values:")
print(
    df_model[
        ["Price_INR", "CarpetArea_sqft", "Price_per_sqft"]
    ].head(10)
)

print("\nDataset Shape:")
print(df_model.shape)

print("\nNew Columns:")
print(df_model.columns.tolist())

