import pandas as pd
import matplotlib.pyplot as plt


df1 = pd.read_csv("data/train_part1.csv")
df2 = pd.read_csv("data/train_part2.csv")

df = pd.concat([df1, df2], ignore_index=True)

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





print("\nDataset Shape:")
print(df_clean.shape)

print("\nNumber of Duplicate Rows:")
print(df_clean.duplicated().sum())


print("\n" + "=" * 60)
print("CITY-WISE REAL ESTATE ANALYSIS")
print("=" * 60)

city_count = df_clean["City"].value_counts()

print("\nNumber of Properties by City:")
print(city_count)


city_avg_price = (
    df_clean.groupby("City")["Price_INR"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Property Price by City:")
print(city_avg_price)


city_median_price = (
    df_clean.groupby("City")["Price_INR"]
    .median()
    .sort_values(ascending=False)
)

print("\nMedian Property Price by City:")
print(city_median_price)



city_median_plot = city_median_price.drop(
    "Unknown",
    errors="ignore"
)

plt.figure(figsize=(10, 6))

city_median_plot.plot(kind="bar")

plt.title("Median Property Price by City")
plt.xlabel("City")
plt.ylabel("Median Price (INR)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


property_count = df_clean["PropertyType"].value_counts()

print("\nNumber of Properties by Property Type:")
print(property_count)


property_avg_price = (
    df_clean.groupby("PropertyType")["Price_INR"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Property Type:")
print(property_avg_price)


property_median_price = (
    df_clean.groupby("PropertyType")["Price_INR"]
    .median()
    .sort_values(ascending=False)
)

print("\nMedian Price by Property Type:")
print(property_median_price)



bhk_count = df_clean["BHK"].value_counts().sort_index()

print("\nNumber of Properties by BHK:")
print(bhk_count)


bhk_avg_price = (
    df_clean.groupby("BHK")["Price_INR"]
    .mean()
    .sort_index()
)

print("\nAverage Price by BHK:")
print(bhk_avg_price)


plt.figure(figsize=(10, 6))

bhk_avg_price.plot(kind="bar")

plt.title("Average Property Price by BHK")
plt.xlabel("BHK")
plt.ylabel("Average Price (INR)")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()

print("\n" + "=" * 60)
print("FURNISHING ANALYSIS")
print("=" * 60)

furnishing_count = df_clean["Furnishing"].value_counts()

print("\nNumber of Properties by Furnishing:")
print(furnishing_count)


furnishing_avg_price = (
    df_clean.groupby("Furnishing")["Price_INR"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage Price by Furnishing:")
print(furnishing_avg_price)



rera_count = df_clean["IsRERARegistered"].value_counts()

print("\nRERA Registration Count:")
print(rera_count)


rera_avg_price = (
    df_clean.groupby("IsRERARegistered")["Price_INR"]
    .mean()
)

print("\nAverage Price by RERA Registration:")
print(rera_avg_price)
print("\n" + "=" * 60)
print("PRICE ANALYSIS")
print("=" * 60)

print("\nMinimum Property Price:")
print(df_clean["Price_INR"].min())

print("\nMaximum Property Price:")
print(df_clean["Price_INR"].max())

print("\nAverage Property Price:")
print(df_clean["Price_INR"].mean())

print("\nMedian Property Price:")
print(df_clean["Price_INR"].median())



print("\nAverage Super Built-Up Area:")
print(df_clean["SuperBuiltUpArea_sqft"].mean())

print("\nAverage Built-Up Area:")
print(df_clean["BuiltUpArea_sqft"].mean())

print("\nAverage Carpet Area:")
print(df_clean["CarpetArea_sqft"].mean())


