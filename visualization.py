import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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



df_clean["Price_per_sqft"] = (
    df_clean["Price_INR"] / df_clean["CarpetArea_sqft"]
)

df_clean["PropertyAge"] = 2026 - df_clean["YearBuilt"]

df_clean["Bathrooms_per_BHK"] = (
    df_clean["Bathrooms"] /
    df_clean["BHK"].replace(0, 1)
)

df_clean["Amenities_per_BHK"] = (
    df_clean["AmenitiesCount"] /
    df_clean["BHK"].replace(0, 1)
)



city_price = (
    df_clean[df_clean["City"] != "Unknown"]
    .groupby("City")["Price_INR"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

city_price.plot(kind="bar")

plt.title("Median Property Price by City")
plt.xlabel("City")
plt.ylabel("Median Price (INR)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



property_type_price = (
    df_clean.groupby("PropertyType")["Price_INR"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

property_type_price.plot(kind="bar")

plt.title("Median Property Price by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Median Price (INR)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



bhk_price = (
    df_clean.groupby("BHK")["Price_INR"]
    .mean()
)

plt.figure(figsize=(10, 6))

bhk_price.plot(kind="line", marker="o")

plt.title("Average Property Price by BHK")
plt.xlabel("BHK")
plt.ylabel("Average Price (INR)")
plt.grid(True)
plt.tight_layout()

plt.show()



price_sqft_city = (
    df_clean[df_clean["City"] != "Unknown"]
    .groupby("City")["Price_per_sqft"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 6))

price_sqft_city.plot(kind="bar")

plt.title("Median Price per Sqft by City")
plt.xlabel("City")
plt.ylabel("Price per Sqft (INR)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()



rera_count = df_clean["IsRERARegistered"].value_counts()

plt.figure(figsize=(7, 5))

rera_count.plot(
    kind="pie",
    autopct="%1.1f%%",
    labels=["RERA Registered", "Not RERA Registered"]
)

plt.title("RERA Registration Distribution")
plt.ylabel("")

plt.show()



furnishing_price = (
    df_clean.groupby("Furnishing")["Price_INR"]
    .median()
    .sort_values(ascending=False)
)

plt.figure(figsize=(9, 6))

furnishing_price.plot(kind="bar")

plt.title("Median Property Price by Furnishing Type")
plt.xlabel("Furnishing Type")
plt.ylabel("Median Price (INR)")
plt.xticks(rotation=30)
plt.tight_layout()

plt.show()



plt.figure(figsize=(10, 6))

plt.scatter(
    df_clean["PropertyAge"],
    df_clean["Price_INR"],
    alpha=0.5
)

plt.title("Property Age vs Property Price")
plt.xlabel("Property Age (Years)")
plt.ylabel("Property Price (INR)")

plt.tight_layout()

plt.show()


print("\n" + "=" * 60)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nTotal Records:", len(df_clean))
print("Total Columns:", len(df_clean.columns))

print("\nCharts Created:")
print("1. Median Property Price by City")
print("2. Median Property Price by Property Type")
print("3. Average Property Price by BHK")
print("4. Median Price per Sqft by City")
print("5. RERA Registration Distribution")
print("6. Median Property Price by Furnishing")
print("7. Property Age vs Property Price")