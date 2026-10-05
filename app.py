import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Használtautó Piaci Értékbecslő", page_icon="🚘", layout="wide")

st.title("🚘 Kontinentális Használtautó Piaci Értékbecslő")
st.caption("AutoScout24 balkormányos, európai hirdetések alapján becsült reális magyar piaci érték.")

@st.cache_resource
def load_bundle():
    return joblib.load("auto_modell_nemet.pkl")

bundle = load_bundle()
model = bundle["model"]
feature_columns = bundle["features"]
brand_model_map = bundle["brand_model_map"]
gears = bundle["gears"]
fuels = bundle["fuels"]

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Jármű kiválasztása")
    makes = sorted(list(brand_model_map.keys()))
    selected_make = st.selectbox("Márka kiválasztása:", makes, index=makes.index("Skoda") if "Skoda" in makes else 0)

    available_models = brand_model_map[selected_make]
    selected_model = st.selectbox("Típus / Modell:", available_models)

    year = st.slider("Gyártási év:", min_value=2002, max_value=2024, value=2016, step=1)
    hp = st.number_input("Teljesítmény (Lóerő / LE):", min_value=50, max_value=600, value=150, step=5)

with col2:
    st.subheader("2. Műszaki paraméterek")
    mileage = st.number_input("Futott kilométer (km):", min_value=1000, max_value=400000, value=145000, step=5000)
    gear = st.selectbox("Sebességváltó:", gears)
    fuel = st.selectbox("Üzemanyag típusa:", fuels)

st.divider()

if st.button("📊 Reális piaci ár becslése", type="primary", use_container_width=True):
    input_row = {col: 0 for col in feature_columns}

    # Folytonos mezők
    input_row["year"] = year
    input_row["mileage"] = mileage
    input_row["hp"] = hp

    # Kategorikus mezők
    for col_name in [f"make_{selected_make}", f"model_{selected_model}", f"gear_{gear}", f"fuel_{fuel}"]:
        if col_name in input_row:
            input_row[col_name] = 1

    input_df = pd.DataFrame([input_row])
    predicted_price = model.predict(input_df)[0]

    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.success(f"### Becsült átlagos piaci érték:\n# **{predicted_price:,.0f} Ft**")
    with res_col2:
        lower_bound = predicted_price * 0.93
        upper_bound = predicted_price * 1.07
        st.info(f"### Reális kereskedelmi ársáv:\n**{lower_bound:,.0f} Ft – {upper_bound:,.0f} Ft**\n\n*(Műszaki állapottól, előélettől és felszereltségtől függően)*")