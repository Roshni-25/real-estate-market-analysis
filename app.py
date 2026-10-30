import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Real Estate Market Analysis",
    page_icon="🏠",
    layout="wide"
)

@st.cache_data
def load_data():
    
    df1 = pd.read_csv("data/train_part1.csv")
    df2 = pd.read_csv("data/train_part2.csv")

    df = pd.concat([df1, df2], ignore_index=True)
    return df


df = load_data()

df_clean = df.copy()


df_clean["Bathrooms"] = df_clean["Bathrooms"].fillna(df_clean["Bathrooms"].median())
df_clean["Balconies"] = df_clean["Balconies"].fillna(df_clean["Balconies"].median())
df_clean["YearBuilt"] = df_clean["YearBuilt"].fillna(df_clean["YearBuilt"].median())
df_clean["Latitude"] = df_clean["Latitude"].fillna(df_clean["Latitude"].median())
df_clean["Longitude"] = df_clean["Longitude"].fillna(df_clean["Longitude"].median())


df_clean["City"] = df_clean["City"].fillna("Unknown")
df_clean["Locality"] = df_clean["Locality"].fillna("Unknown")
df_clean["Furnishing"] = df_clean["Furnishing"].fillna("Unknown")
df_clean["Parking"] = df_clean["Parking"].fillna("Unknown")
df_clean["Facing"] = df_clean["Facing"].fillna("Unknown")


df_clean["Price_per_sqft"] = df_clean["Price_INR"] / df_clean["CarpetArea_sqft"]


df_clean["PropertyAge"] = 2026 - df_clean["YearBuilt"]

@st.cache_resource
def load_model():
    model = joblib.load("real_estate_price_model.pkl")
    return model


model = load_model()


st.title("🏠 Real Estate Market Analysis Dashboard")

st.write(
    "Explore real estate market trends and predict property prices "
    "using a machine learning model."
)


st.sidebar.header("Dashboard Filters")

city_options = sorted(df_clean["City"].dropna().unique())
selected_city = st.sidebar.selectbox("Select City", ["All"] + city_options)

property_type_options = sorted(df_clean["PropertyType"].dropna().unique())
selected_property_type = st.sidebar.selectbox(
    "Select Property Type", ["All"] + property_type_options
)


bhk_options = sorted(df_clean["BHK"].unique())
selected_bhk = st.sidebar.selectbox("Select BHK", ["All"] + bhk_options)


filtered_df = df_clean.copy()

if selected_city != "All":
    filtered_df = filtered_df[filtered_df["City"] == selected_city]

if selected_property_type != "All":
    filtered_df = filtered_df[filtered_df["PropertyType"] == selected_property_type]

if selected_bhk != "All":
    filtered_df = filtered_df[filtered_df["BHK"] == selected_bhk]


st.subheader("📊 Market Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Properties", len(filtered_df))

with col2:
    st.metric(
        "Average Price",
        f"₹{filtered_df['Price_INR'].mean():,.0f}" if len(filtered_df) > 0 else "N/A"
    )

with col3:
    st.metric(
        "Median Price",
        f"₹{filtered_df['Price_INR'].median():,.0f}" if len(filtered_df) > 0 else "N/A"
    )

with col4:
    st.metric(
        "Average Carpet Area",
        f"{filtered_df['CarpetArea_sqft'].mean():,.0f} sqft"
        if len(filtered_df) > 0 else "N/A"
    )

st.subheader("💰 Median Property Price by City")

if len(filtered_df) > 0:
    city_price = (
        filtered_df.groupby("City")["Price_INR"]
        .median()
        .sort_values(ascending=False)
    )
    st.bar_chart(city_price)

st.subheader("🏢 Median Price by Property Type")

if len(filtered_df) > 0:
    property_price = (
        filtered_df.groupby("PropertyType")["Price_INR"]
        .median()
        .sort_values(ascending=False)
    )
    st.bar_chart(property_price)


st.subheader("🛏️ Average Price by BHK")

if len(filtered_df) > 0:
    bhk_price = (
        filtered_df.groupby("BHK")["Price_INR"]
        .mean()
        .sort_index()
    )
    st.line_chart(bhk_price)

st.subheader("📐 Median Price per Sqft by City")

if len(filtered_df) > 0:
    price_sqft = (
        filtered_df.groupby("City")["Price_per_sqft"]
        .median()
        .sort_values(ascending=False)
    )
    st.bar_chart(price_sqft)



st.subheader("📋 RERA Registration")

if len(filtered_df) > 0:
    rera_counts = filtered_df["IsRERARegistered"].value_counts()
    st.bar_chart(rera_counts)



st.subheader("🛋️ Median Price by Furnishing")

if len(filtered_df) > 0:
    furnishing_price = (
        filtered_df.groupby("Furnishing")["Price_INR"]
        .median()
        .sort_values(ascending=False)
    )
    st.bar_chart(furnishing_price)



st.subheader("🏗️ Property Age vs Price")

if len(filtered_df) > 0:
    age_price = (
        filtered_df.groupby("PropertyAge")["Price_INR"]
        .mean()
        .sort_index()
    )
    st.line_chart(age_price)




st.subheader("📋 Property Data")

st.dataframe(filtered_df, width="stretch")



st.divider()

st.header("🤖 Property Price Prediction")

st.write("Enter the property details below to estimate its market price.")

# We use 3 columns so the input boxes look neat
col1, col2, col3 = st.columns(3)


with col1:

    prediction_city = st.selectbox(
        "City",
        sorted(df_clean["City"].dropna().unique()),
        key="prediction_city"
    )

    prediction_locality = st.selectbox(
        "Locality",
        sorted(df_clean["Locality"].dropna().unique()),
        key="prediction_locality"
    )

    prediction_property_type = st.selectbox(
        "Property Type",
        sorted(df_clean["PropertyType"].dropna().unique()),
        key="prediction_property_type"
    )

    prediction_bhk = st.number_input(
        "BHK", min_value=0, max_value=10, value=3, step=1
    )

    prediction_bathrooms = st.number_input(
        "Bathrooms", min_value=0.0, max_value=10.0, value=2.0, step=1.0
    )


with col2:

    prediction_balconies = st.number_input(
        "Balconies", min_value=0.0, max_value=10.0, value=1.0, step=1.0
    )

    prediction_super_area = st.number_input(
        "Super Built-Up Area (sqft)",
        min_value=100, max_value=20000, value=1500, step=50
    )

    prediction_built_area = st.number_input(
        "Built-Up Area (sqft)",
        min_value=100, max_value=20000, value=1200, step=50
    )

    prediction_carpet_area = st.number_input(
        "Carpet Area (sqft)",
        min_value=100, max_value=20000, value=1000, step=50
    )

    prediction_floor = st.number_input(
        "Floor", min_value=0, max_value=100, value=3, step=1
    )

# ----- Column 3 -----
with col3:

    prediction_total_floors = st.number_input(
        "Total Floors", min_value=1, max_value=100, value=10, step=1
    )

    prediction_year = st.number_input(
        "Year Built", min_value=1900, max_value=2026, value=2018, step=1
    )

    prediction_amenities = st.number_input(
        "Amenities Count", min_value=0, max_value=100, value=5, step=1
    )

    prediction_latitude = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0, value=28.61, format="%.6f"
    )

    prediction_longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, value=77.23, format="%.6f"
    )


# Second row of inputs (dropdowns)
col4, col5, col6 = st.columns(3)

with col4:
    prediction_furnishing = st.selectbox(
        "Furnishing",
        sorted(df_clean["Furnishing"].dropna().unique()),
        key="prediction_furnishing"
    )

with col5:
    prediction_parking = st.selectbox(
        "Parking",
        sorted(df_clean["Parking"].dropna().unique()),
        key="prediction_parking"
    )

with col6:
    prediction_building = st.selectbox(
        "Building Type",
        sorted(df_clean["BuildingType"].dropna().unique()),
        key="prediction_building"
    )

prediction_facing = st.selectbox(
    "Facing",
    sorted(df_clean["Facing"].dropna().unique()),
    key="prediction_facing"
)




if st.button("🔮 Predict Property Price", width="stretch"):

    
    property_age = 2026 - prediction_year


    prediction_data = pd.DataFrame({
        "BHK": [prediction_bhk],
        "Bathrooms": [prediction_bathrooms],
        "Balconies": [prediction_balconies],
        "SuperBuiltUpArea_sqft": [prediction_super_area],
        "BuiltUpArea_sqft": [prediction_built_area],
        "CarpetArea_sqft": [prediction_carpet_area],
        "Floor": [prediction_floor],
        "TotalFloors": [prediction_total_floors],
        "YearBuilt": [prediction_year],
        "PropertyAge": [property_age],
        "AmenitiesCount": [prediction_amenities],
        "Latitude": [prediction_latitude],
        "Longitude": [prediction_longitude],
        "City": [prediction_city],
        "Locality": [prediction_locality],
        "PropertyType": [prediction_property_type],
        "Furnishing": [prediction_furnishing],
        "Parking": [prediction_parking],
        "BuildingType": [prediction_building],
        "Facing": [prediction_facing]
    })

   
    predicted_price = model.predict(prediction_data)[0]

   
    st.success("Price Prediction Completed!")

    st.metric("Estimated Property Price", f"₹{predicted_price:,.0f}")

    st.info(
        "This is an ML-based estimated price and should be "
        "used as an analytical estimate rather than an exact "
        "market valuation."

    )