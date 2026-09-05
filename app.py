
# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Crop Advisor AI",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# FILE PATHS
# =========================================================



MODEL_PATH = "crop_prediction_model2.pkl"
DATA_PATH = "original_dataset.csv"


# =========================================================
# LOAD MODEL
# =========================================================

try:

    model = joblib.load(MODEL_PATH)

except FileNotFoundError:

    st.error(
        "❌ The trained model could not be found.\n\n"
        f"Expected file: `{MODEL_PATH.name}`"
    )

    st.stop()

except Exception as e:

    st.error(
        "❌ An error occurred while loading the trained model."
    )

    st.exception(e)

    st.stop()


# =========================================================
# LOAD ORIGINAL DATASET
# =========================================================

try:

    crop_data = pd.read_csv(DATA_PATH)

except FileNotFoundError:

    st.error(
        "❌ The original dataset could not be found.\n\n"
        f"Expected file: `{DATA_PATH.name}`"
    )

    st.stop()

except Exception as e:

    st.error(
        "❌ An error occurred while loading the dataset."
    )

    st.exception(e)

    st.stop()


# =========================================================
# CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "crop_label",
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "soil"
]

missing_columns = [
    column
    for column in required_columns
    if column not in crop_data.columns
]

if missing_columns:

    st.error(
        "❌ The dataset is missing the following required columns:"
    )

    st.write(missing_columns)

    st.stop()


# =========================================================
# STANDARDIZE COLUMN NAMES
# =========================================================

crop_data = crop_data.rename(
    columns={
        "soil": "soil_texture"
    }
)


# Convert crop labels to strings so that they match
# the labels returned by the model.
crop_data["crop_label"] = (
    crop_data["crop_label"]
    .astype(str)
    .str.strip()
)


# Standardize soil texture values.
crop_data["soil_texture"] = (
    crop_data["soil_texture"]
    .astype(str)
    .str.strip()
)


# =========================================================
# CREATE CROP PROFILES
# =========================================================

crop_profiles = {}


for crop in crop_data["crop_label"].unique():

    crop_sample = crop_data[
        crop_data["crop_label"] == crop
    ]

    crop_profiles[crop] = {

        # Nitrogen
        "N_min": crop_sample["N"].min(),
        "N_max": crop_sample["N"].max(),

        # Phosphorus
        "P_min": crop_sample["P"].min(),
        "P_max": crop_sample["P"].max(),

        # Potassium
        "K_min": crop_sample["K"].min(),
        "K_max": crop_sample["K"].max(),

        # Temperature
        "temperature_min": crop_sample["temperature"].min(),
        "temperature_max": crop_sample["temperature"].max(),

        # Humidity
        "humidity_min": crop_sample["humidity"].min(),
        "humidity_max": crop_sample["humidity"].max(),

        # Soil pH
        "ph_min": crop_sample["ph"].min(),
        "ph_max": crop_sample["ph"].max(),

        # Rainfall
        "rainfall_min": crop_sample["rainfall"].min(),
        "rainfall_max": crop_sample["rainfall"].max(),

        # Soil texture
        "soil_types": crop_sample["soil_texture"].unique()
    }


# =========================================================
# EXPLANATION FUNCTION
# =========================================================

def generate_crop_explanation(crop, farm_conditions):

    """
    Generate human-readable explanations showing how the
    farmer's conditions compare with the conditions observed
    for the recommended crop in the training dataset.
    """

    # Check whether a profile exists for this crop.
    if crop not in crop_profiles:

        return [
            "⚠ A detailed explanation is not available "
            "for this crop because its profile was not found "
            "in the reference dataset."
        ]

    profile = crop_profiles[crop]

    reasons = []


    # -----------------------------------------------------
    # Rainfall
    # -----------------------------------------------------

    if (
        profile["rainfall_min"]
        <= farm_conditions["rainfall"]
        <= profile["rainfall_max"]
    ):

        reasons.append(
            f"✓ Rainfall ({farm_conditions['rainfall']:.1f} mm) "
            f"is within the observed range for {crop} "
            f"({profile['rainfall_min']:.1f}–"
            f"{profile['rainfall_max']:.1f} mm)."
        )

    else:

        reasons.append(
            f"⚠ Rainfall ({farm_conditions['rainfall']:.1f} mm) "
            f"is outside the common observed range for {crop} "
            f"({profile['rainfall_min']:.1f}–"
            f"{profile['rainfall_max']:.1f} mm)."
        )


    # -----------------------------------------------------
    # Temperature
    # -----------------------------------------------------

    if (
        profile["temperature_min"]
        <= farm_conditions["temperature"]
        <= profile["temperature_max"]
    ):

        reasons.append(
            f"✓ Temperature ({farm_conditions['temperature']:.1f} °C) "
            f"is within the observed range for {crop} "
            f"({profile['temperature_min']:.1f}–"
            f"{profile['temperature_max']:.1f} °C)."
        )

    else:

        reasons.append(
            f"⚠ Temperature ({farm_conditions['temperature']:.1f} °C) "
            f"is outside the common observed range for {crop} "
            f"({profile['temperature_min']:.1f}–"
            f"{profile['temperature_max']:.1f} °C)."
        )


    # -----------------------------------------------------
    # Humidity
    # -----------------------------------------------------

    if (
        profile["humidity_min"]
        <= farm_conditions["humidity"]
        <= profile["humidity_max"]
    ):

        reasons.append(
            f"✓ Humidity ({farm_conditions['humidity']:.1f}%) "
            f"is within the observed range for {crop} "
            f"({profile['humidity_min']:.1f}–"
            f"{profile['humidity_max']:.1f}%)."
        )

    else:

        reasons.append(
            f"⚠ Humidity ({farm_conditions['humidity']:.1f}%) "
            f"is outside the common observed range for {crop} "
            f"({profile['humidity_min']:.1f}–"
            f"{profile['humidity_max']:.1f}%)."
        )


    # -----------------------------------------------------
    # Soil pH
    # -----------------------------------------------------

    if (
        profile["ph_min"]
        <= farm_conditions["ph"]
        <= profile["ph_max"]
    ):

        reasons.append(
            f"✓ Soil pH ({farm_conditions['ph']:.2f}) "
            f"is within the observed range for {crop} "
            f"({profile['ph_min']:.2f}–"
            f"{profile['ph_max']:.2f})."
        )

    else:

        reasons.append(
            f"⚠ Soil pH ({farm_conditions['ph']:.2f}) "
            f"is outside the common observed range for {crop} "
            f"({profile['ph_min']:.2f}–"
            f"{profile['ph_max']:.2f})."
        )


    # -----------------------------------------------------
    # Soil Texture
    # -----------------------------------------------------

    if farm_conditions["soil_texture"] in profile["soil_types"]:

        reasons.append(
            f"✓ Soil texture "
            f"({farm_conditions['soil_texture']}) "
            f"matches soil types observed for {crop}."
        )

    else:

        reasons.append(
            f"⚠ Soil texture "
            f"({farm_conditions['soil_texture']}) "
            f"differs from the common soil types observed "
            f"for {crop}."
        )


    # -----------------------------------------------------
    # Nitrogen
    # -----------------------------------------------------

    if (
        profile["N_min"]
        <= farm_conditions["N"]
        <= profile["N_max"]
    ):

        reasons.append(
            f"✓ Nitrogen level ({farm_conditions['N']:.1f}) "
            f"is within the observed range for {crop} "
            f"({profile['N_min']:.1f}–"
            f"{profile['N_max']:.1f})."
        )

    else:

        reasons.append(
            f"⚠ Nitrogen level ({farm_conditions['N']:.1f}) "
            f"is outside the common observed range for {crop}."
        )


    # -----------------------------------------------------
    # Phosphorus
    # -----------------------------------------------------

    if (
        profile["P_min"]
        <= farm_conditions["P"]
        <= profile["P_max"]
    ):

        reasons.append(
            f"✓ Phosphorus level ({farm_conditions['P']:.1f}) "
            f"is within the observed range for {crop} "
            f"({profile['P_min']:.1f}–"
            f"{profile['P_max']:.1f})."
        )

    else:

        reasons.append(
            f"⚠ Phosphorus level ({farm_conditions['P']:.1f}) "
            f"is outside the common observed range for {crop}."
        )


    # -----------------------------------------------------
    # Potassium
    # -----------------------------------------------------

    if (
        profile["K_min"]
        <= farm_conditions["K"]
        <= profile["K_max"]
    ):

        reasons.append(
            f"✓ Potassium level ({farm_conditions['K']:.1f}) "
            f"is within the observed range for {crop} "
            f"({profile['K_min']:.1f}–"
            f"{profile['K_max']:.1f})."
        )

    else:

        reasons.append(
            f"⚠ Potassium level ({farm_conditions['K']:.1f}) "
            f"is outside the common observed range for {crop}."
        )


    return reasons


# =========================================================
# APPLICATION HEADER
# =========================================================

st.title("🌱 Crop Advisor AI")

st.write(
    """
### Agricultural Extension Agent Decision Support Tool

Enter measured soil and environmental conditions to receive
AI-powered crop recommendations with transparent explanations.
"""
)


st.info(
    "💡 The recommendations are generated by a trained "
    "machine-learning model. The explanations compare your "
    "farm conditions with conditions observed in the reference dataset."
)


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("🌾 Farm Conditions")


col1, col2 = st.columns(2)


with col1:

    N = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    P = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    K = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        value=25.0,
        step=0.5
    )


with col2:

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=1.0
    )

    ph = st.number_input(
        "Soil pH",
        min_value=0.0,
        max_value=14.0,
        value=6.5,
        step=0.1
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=100.0,
        step=10.0
    )

    soil_texture = st.selectbox(
        "Soil Texture",
        sorted(crop_data["soil_texture"].unique())
    )


# =========================================================
# GENERATE RECOMMENDATION
# =========================================================

if st.button(
    "🌱 Generate Recommendation",
    use_container_width=True
):

    # -----------------------------------------------------
    # Prepare model input
    # -----------------------------------------------------

    input_data = pd.DataFrame({

        "N": [N],

        "P": [P],

        "K": [K],

        "temperature": [temperature],

        "humidity": [humidity],

        "ph": [ph],

        "rainfall": [rainfall],

        "soil_texture": [soil_texture]

    })


    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    try:

        with st.spinner(
            "Analyzing your farm conditions..."
        ):

            probabilities = model.predict_proba(
                input_data
            )[0]


            # Convert model classes to strings.
            crops = [
                str(crop).strip()
                for crop in model.classes_
            ]


            recommendations = pd.DataFrame({

                "Crop": crops,

                "Probability": probabilities

            }).sort_values(
                by="Probability",
                ascending=False
            ).head(3)


    except Exception as e:

        st.error(
            "❌ The model could not generate a prediction."
        )

        st.exception(e)

        st.stop()


    # -----------------------------------------------------
    # Check recommendation
    # -----------------------------------------------------

    if recommendations.empty:

        st.warning(
            "⚠ No crop recommendation could be generated."
        )

        st.stop()


    # -----------------------------------------------------
    # Farm conditions for explanation
    # -----------------------------------------------------

    farm_conditions = {

        "N": N,

        "P": P,

        "K": K,

        "temperature": temperature,

        "humidity": humidity,

        "ph": ph,

        "rainfall": rainfall,

        "soil_texture": soil_texture

    }


    # =====================================================
    # TOP RECOMMENDATION
    # =====================================================

    recommended_crop = str(
        recommendations.iloc[0]["Crop"]
    ).strip()

    recommended_probability = (
        recommendations.iloc[0]["Probability"] * 100
    )


    st.success(
        f"🌱 Recommended Crop: "
        f"**{recommended_crop.title()}** "
        f"({recommended_probability:.1f}% suitability)"
    )


    # =====================================================
    # TOP 3 RECOMMENDATIONS
    # =====================================================

    st.subheader("🏆 Top 3 Crop Recommendations")


    for rank, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        crop = str(
            row["Crop"]
        ).strip()

        confidence = (
            row["Probability"] * 100
        )


        st.markdown(
            f"### {rank}. {crop.title()}"
        )


        st.progress(
            min(max(float(row["Probability"]), 0.0), 1.0)
        )


        st.write(
            f"**Suitability score:** "
            f"{confidence:.1f}%"
        )


        # -------------------------------------------------
        # Explanation
        # -------------------------------------------------

        explanations = generate_crop_explanation(
            crop,
            farm_conditions
        )


        st.markdown(
            "**Why this crop was recommended:**"
        )


        for explanation in explanations:

            st.write(explanation)


        st.divider()


    # =====================================================
    # INPUT SUMMARY
    # =====================================================

    with st.expander(
        "📋 View Your Farm Conditions"
    ):

        summary = pd.DataFrame({

            "Parameter": [
                "Nitrogen",
                "Phosphorus",
                "Potassium",
                "Temperature",
                "Humidity",
                "Soil pH",
                "Rainfall",
                "Soil Texture"
            ],

            "Value": [
                f"{N:.1f}",
                f"{P:.1f}",
                f"{K:.1f}",
                f"{temperature:.1f} °C",
                f"{humidity:.1f}%",
                f"{ph:.2f}",
                f"{rainfall:.1f} mm",
                soil_texture
            ]

        })


        st.table(summary)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🌱 Crop Advisor AI | Machine Learning-Based "
    "Agricultural Decision Support"
)

st.caption(
    "Note: Recommendations are model predictions and "
    "should be considered alongside local agricultural "
    "expert advice and field conditions."
)

