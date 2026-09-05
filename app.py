import streamlit as st
import pandas as pd
import joblib


model = joblib.load("crop_prediction_model2.pkl")
label_encoder = joblib.load("crop_label_encoder.pkl")


st.set_page_config(
    page_title="Crop Advisor AI",
    page_icon="🌱"
)


st.title("🌱 Crop Advisor AI")
st.subheader("Extension Agent Decision Support Tool")

st.write(
    """
This tool helps agricultural extension agents recommend suitable crops
using measured soil and environmental conditions.
"""
)

st.write(
    """
Enter the soil nutrient and environmental conditions
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

soil = st.selectbox(
    "Soil Texture",
    [
        "loamy",
        "sandy loam",
        "loamy clay",
        "clay",
        "sandy"
    ]
)


if st.button("Generate Crop Recommendation"):

    input_data = pd.DataFrame({
        "N":[N],
        "P":[P],
        "K":[K],
        "temperature":[temperature],
        "humidity":[humidity],
        "ph":[ph],
        "rainfall":[rainfall],
        "soil_texture":[soil]
    })


    probabilities = model.predict_proba(input_data)[0]

    encoded_classes = model.classes_

    crops = label_encoder.inverse_transform(
        encoded_classes.astype(int)
    )
    
    crops = [
    str(crop).strip()
    for crop in crops
    ]


    results = (
        pd.DataFrame({
            "Crop":crops,
            "Probability":probabilities
        })
        .sort_values(
            "Probability",
            ascending=False
        )
        .head(3)
    )


    st.success(
        f"Primary Recommendation: {results.iloc[0]['Crop'].title()}"
    )


    st.write("### Alternative Suitable Crops")

    for _, row in results.iterrows():

        st.write(
            f"- {row['Crop'].title()} "
            f"({row['Probability']*100:.1f}% model confidence)"
        )


    crop = results.iloc[0]["Crop"]


    st.write("### Why this crop is suitable")

    st.write(
        f"""
The recommendation for **{crop.title()}** is based on the combined
relationship between:

- available soil nutrients (Nitrogen, Phosphorus and Potassium);
- soil texture ({soil});
- temperature conditions ({temperature}°C);
- humidity level ({humidity}%);
- rainfall availability ({rainfall} mm);
- soil acidity/alkalinity (pH {ph}).

The model identified this crop as the most compatible option
based on historical crop and soil patterns in the dataset.
"""
    )
