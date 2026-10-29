<<<<<<< HEAD
import pandas as pd
df1 = pd.read_csv("data/train_part1.csv")
df2 = pd.read_csv("data/train_part2.csv")
df = pd.concat([df1, df2], ignore_index=True)
print("=" * 60)
print("Real Estate Dataset Market Anlysis")
print("=" * 60)
print("\n1. Dataset Shape:")
print(df.shape)
print("\n2. Column Names:")
print(df.columns.tolist())
print("\n3. First 5 rows:")
print(df.head())
print("\n4. Data Types:")
print(df.dtypes)
print("\n5. Missing Values:")
print(df.isnull().sum())
print("\n6. Missing Value Percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\n7. Duplicate Rows:")
print(df.duplicated().sum())
print("\n8. Statistical Summary:")
print(df.describe())
print("\n" + "=" * 60)
print("Investigation Completed")
print("=" * 60)

print("\n" + "=" * 60)
print("MISSING VALUE RELATIONSHIPS")
print("=" * 60)

print("\nRERA Registration vs Missing RERAID:")
rera_check = df.groupby("IsRERARegistered")["RERAID"].apply(
    lambda x: x.isnull().sum()
)

print(rera_check)
print("\nRows with missing City:")
print(df[df["City"].isnull()][["ListingID", "City", "Locality", "Price_INR"]].head(10))
print("\nRows with missing Parking:")
print(df[df["Parking"].isnull()][
    ["ListingID", "City", "PropertyType", "Parking", "Price_INR"]
].head(10))
print("\nRows with missing Furnishing:")
print(df[df["Furnishing"].isnull()][
    ["ListingID", "City", "PropertyType", "Furnishing", "Price_INR"]
=======
import pandas as pd
df1 = pd.read_csv("data/train_part1.csv")
df2 = pd.read_csv("data/train_part2.csv")
df = pd.concat([df1, df2], ignore_index=True)
print("=" * 60)
print("Real Estate Dataset Market Anlysis")
print("=" * 60)
print("\n1. Dataset Shape:")
print(df.shape)
print("\n2. Column Names:")
print(df.columns.tolist())
print("\n3. First 5 rows:")
print(df.head())
print("\n4. Data Types:")
print(df.dtypes)
print("\n5. Missing Values:")
print(df.isnull().sum())
print("\n6. Missing Value Percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\n7. Duplicate Rows:")
print(df.duplicated().sum())
print("\n8. Statistical Summary:")
print(df.describe())
print("\n" + "=" * 60)
print("Investigation Completed")
print("=" * 60)

print("\n" + "=" * 60)
print("MISSING VALUE RELATIONSHIPS")
print("=" * 60)

print("\nRERA Registration vs Missing RERAID:")
rera_check = df.groupby("IsRERARegistered")["RERAID"].apply(
    lambda x: x.isnull().sum()
)

print(rera_check)
print("\nRows with missing City:")
print(df[df["City"].isnull()][["ListingID", "City", "Locality", "Price_INR"]].head(10))
print("\nRows with missing Parking:")
print(df[df["Parking"].isnull()][
    ["ListingID", "City", "PropertyType", "Parking", "Price_INR"]
].head(10))
print("\nRows with missing Furnishing:")
print(df[df["Furnishing"].isnull()][
    ["ListingID", "City", "PropertyType", "Furnishing", "Price_INR"]
>>>>>>> 6035529d0fea56a3565b45923dd4478d577b1184
].head(10))