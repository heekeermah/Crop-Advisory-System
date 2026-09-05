
import streamlit as st
import pandas as pd
import joblib


# ---------------------------------------------------------
# LOAD MODEL AND DATA
# ---------------------------------------------------------

# Load the trained XGBoost pipeline
# The pipeline already contains preprocessing and the model
model = joblib.load("crop_prediction_model.pkl")


# Load dataset used during training
# This is used only for generating human-readable explanations
# by understanding the normal conditions where each crop grows.
crop_data = pd.read_csv("original_dataset.csv")


# Ensure column name matches the model/application naming convention
crop_data = crop_data.rename(
    columns={
        "soil": "soil_texture"
    }
)


# ---------------------------------------------------------
# CREATE CROP PROFILES
# ---------------------------------------------------------

# For each crop, calculate observed ranges from the dataset.
# Example:
# Rice normally appears between rainfall values X-Y.
# These profiles allow us to explain WHY a crop was recommended.
crop_profiles = {}

for crop in crop_data["crop_label"].unique():

    crop_sample = crop_data[
        crop_data["crop_label"] == crop
    ]

    crop_profiles[crop] = {

        "N_min": crop_sample["N"].min(),
        "N_max": crop_sample["N"].max(),

        "P_min": crop_sample["P"].min(),
        "P_max": crop_sample["P"].max(),

        "K_min": crop_sample["K"].min(),
        "K_max": crop_sample["K"].max(),

        "rainfall_min": crop_sample["rainfall"].min(),
        "rainfall_max": crop_sample["rainfall"].max(),

        "humidity_min": crop_sample["humidity"].min(),
        "humidity_max": crop_sample["humidity"].max(),

        "soil_types": crop_sample["soil_texture"].unique()
    }


# ---------------------------------------------------------
# EXPLANATION FUNCTION
# ---------------------------------------------------------

def generate_crop_explanation(crop, farm_conditions):

    profile = crop_profiles[crop]

    reasons = []


    # Rainfall explanation
    if profile["rainfall_min"] <= farm_conditions["rainfall"] <= profile["rainfall_max"]:

        reasons.append(
            f"✓ Your rainfall ({farm_conditions['rainfall']}mm) "
            f"matches {crop}'s preferred rainfall conditions"
        )

    else:

        reasons.append(
            f"⚠ Your rainfall ({farm_conditions['rainfall']}mm) "
            f"is outside the common range observed for {crop}"
        )


    # Humidity explanation
    if profile["humidity_min"] <= farm_conditions["humidity"] <= profile["humidity_max"]:

        reasons.append(
            f"✓ Your humidity ({farm_conditions['humidity']}%) "
            f"supports {crop} growing conditions"
        )


    # Soil explanation
    if farm_conditions["soil_texture"] in profile["soil_types"]:

        reasons.append(
            f"✓ Your soil texture ({farm_conditions['soil_texture']}) "
            f"aligns with {crop}-growing samples"
        )

    else:

        reasons.append(
            f"⚠ Your soil texture differs from common {crop} samples"
        )


    # Nutrient explanations

    if profile["N_min"] <= farm_conditions["N"] <= profile["N_max"]:

        reasons.append(
            f"✓ Nitrogen level ({farm_conditions['N']}) "
            f"is within the observed range for {crop}"
        )


    if profile["P_min"] <= farm_conditions["P"] <= profile["P_max"]:

        reasons.append(
            f"✓ Phosphorus level ({farm_conditions['P']}) "
            f"is suitable for {crop}"
        )


    if profile["K_min"] <= farm_conditions["K"] <= profile["K_max"]:

        reasons.append(
            f"✓ Potassium level ({farm_conditions['K']}) "
            f"is suitable for {crop}"
        )


    return reasons



# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Crop Advisor AI",
    page_icon="🌱"
)


st.title("🌱 Crop Advisor AI")

st.write(
    """
Agricultural Extension Agent Decision Support Tool.

Enter measured soil and environmental conditions to receive
crop recommendations with explanations.
"""
)


N = st.number_input("Nitrogen (N)", min_value=0.0)

P = st.number_input("Phosphorus (P)", min_value=0.0)

K = st.number_input("Potassium (K)", min_value=0.0)

temperature = st.number_input(
    "Temperature (°C)",
    value=25.0
)

humidity = st.number_input(
    "Humidity (%)",
    value=80.0
)

ph = st.number_input(
    "Soil pH",
    value=6.5
)

rainfall = st.number_input(
    "Rainfall (mm)",
    value=100.0
)

soil_texture = st.selectbox(
    "Soil Texture",
    crop_data["soil_texture"].unique()
)



if st.button("Generate Recommendation"):


    input_data = pd.DataFrame({

        "N":[N],
        "P":[P],
        "K":[K],
        "temperature":[temperature],
        "humidity":[humidity],
        "ph":[ph],
        "rainfall":[rainfall],
        "soil_texture":[soil_texture]

    })


    # Predict probabilities instead of only one class.
    # This allows us to show the top three recommended crops.
    probabilities = model.predict_proba(input_data)[0]


    crops = model.classes_


    recommendations = pd.DataFrame({

        "Crop": crops,
        "Probability": probabilities

    }).sort_values(
        by="Probability",
        ascending=False
    ).head(3)



    farm_conditions = {

        "N":N,
        "P":P,
        "K":K,
        "humidity":humidity,
        "rainfall":rainfall,
        "soil_texture":soil_texture

    }


    st.success(
        f"Recommended Crop: {recommendations.iloc[0]['Crop'].title()}"
    )


    st.write("## Crop Suitability Explanation")


    for _, row in recommendations.iterrows():

        crop = row["Crop"]

        confidence = row["Probability"] * 100


        st.subheader(
            f"{crop.title()} ({confidence:.1f}% suitability)"
        )


        explanations = generate_crop_explanation(
            crop,
            farm_conditions
        )


        for explanation in explanations:

            st.write(explanation)


        st.divider()
