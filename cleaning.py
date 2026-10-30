
import pandas as pd

df1 = pd.read_csv("data/train_part1.csv")
df2 = pd.read_csv("data/train_part2.csv")
df = pd.concat([df1, df2], ignore_index=True)

print("=" * 60)
print("REAL ESTATE DATA CLEANING")
print("=" * 60)

print("\nOriginal Shape:")
print(df.shape)

df_clean = df.copy()
df_clean["Bathrooms"] = df_clean["Bathrooms"].fillna(
    df_clean["Bathrooms"].median()
)

df_clean["Balconies"] = df_clean["Balconies"].fillna(
    df_clean["Balconies"].median()
)

df_clean["YearBuilt"] = df_clean["YearBuilt"].fillna(
    df_clean["YearBuilt"].median()
)

df_clean["Latitude"] = df_clean["Latitude"].fillna(
    df_clean["Latitude"].median()
)

df_clean["Longitude"] = df_clean["Longitude"].fillna(
    df_clean["Longitude"].median()
)

df_clean["City"] = df_clean["City"].fillna("Unknown")

df_clean["Locality"] = df_clean["Locality"].fillna("Unknown")

df_clean["Furnishing"] = df_clean["Furnishing"].fillna("Unknown")

df_clean["Parking"] = df_clean["Parking"].fillna("Unknown")

df_clean["Facing"] = df_clean["Facing"].fillna("Unknown")




print("\nMissing Values After Cleaning:")
print(df_clean.isnull().sum())

print("\nShape After Cleaning:")
print(df_clean.shape)

print("\nDuplicate Rows:")
print(df_clean.duplicated().sum())

print("\nFirst 5 Rows:")
print(df_clean.head())


print("\n" + "=" * 60)
print("DATA CLEANING COMPLETED")
print("=" * 60)