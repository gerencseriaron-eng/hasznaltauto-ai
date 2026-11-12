import streamlit as st
import pandas as pd
import joblib
import urllib.request
import json

st.set_page_config(
    page_title="Használtautó Piaci Értékbecslő és Ajánlatelemző",
    page_icon="🚘",
    layout="wide"
)

# --- MODELL ÉS ADATOK BETÖLTÉSE ---
@st.cache_resource
def load_bundle():
    return joblib.load("auto_modell_nemet.pkl")

@st.cache_data(ttl=3600)
def get_eur_huf_rate():
    try:
        url = "https://api.frankfurter.app/latest?from=EUR&to=HUF"
        with urllib.request.urlopen(url, timeout=3) as response:
            data = json.loads(response.read().decode())
            return float(data["rates"]["HUF"]), True
    except Exception:
        return 405.0, False

bundle = load_bundle()
model = bundle["model"]
feature_columns = bundle["features"]
brand_model_map = bundle["brand_model_map"]
gears = bundle["gears"]
fuels = bundle["fuels"]
eur_huf, is_live_rate = get_eur_huf_rate()

def predict_car_price(make, car_model, year, mileage, hp, gear, fuel):
    input_row = {col: 0 for col in feature_columns}
    input_row["year"] = year
    input_row["mileage"] = mileage
    input_row["hp"] = hp

    for col_name in [f"make_{make}", f"model_{car_model}", f"gear_{gear}", f"fuel_{fuel}"]:
        if col_name in input_row:
            input_row[col_name] = 1

    input_df = pd.DataFrame([input_row])
    return float(model.predict(input_df)[0])

# --- FEJLÉC ÉS RENDSZERINFORMÁCIÓK ---
st.title("🚘 Használtautó Piaci Értékbecslő és Ajánlatelemző")
st.caption(
    "Nyilvános európai járműhirdetésekből (AutoScout24), műszaki paraméterekből és történeti árakból tanított "
    "Random Forest alapú piaci értékbecslő és anomália-detektor."
)

with st.expander("ℹ️ Hogyan működik a rendszer és honnan származnak az adatok?"):
    info_c1, info_c2, info_c3 = st.columns(3)
    info_c1.markdown("**1. Adatforrás**\nTöbb tízezer nyilvános európai autóhirdetés történeti eladási ára és műszaki adatlapja.")
    info_c2.markdown("**2. Gépi tanulási modell**\nRandom Forest Regressor, amely a márka, modell, évjárat, futásteljesítmény, teljesítmény, váltó és üzemanyag alapján becsül.")
    rate_label = f"{eur_huf:.1f} Ft (élő MNB/ECB)" if is_live_rate else f"{eur_huf:.1f} Ft (referencia)"
    info_c3.markdown(f"**3. Aktuális árfolyam & küszöbök**\n1 EUR = {rate_label}. A rendszer a ±15%-on kívüli ajánlatokat szokatlanként jelöli.")

st.divider()

tab1, tab2 = st.tabs(["🎯 Egyedi jármű értékbecslése és ajánlat-ellenőrzés", "📋 Több hirdetés tömeges szűrése"])

# ==============================================================================
# 1. FÜL: EGYEDI ÉRTÉKBECSLÉS + ANOMÁLIA DETEKTOR + ÉRTÉKVESZTÉSI GÖRBE
# ==============================================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Jármű kiválasztása")
        makes = sorted(list(brand_model_map.keys()))
        default_make_idx = makes.index("Volkswagen") if "Volkswagen" in makes else 0
        selected_make = st.selectbox("Márka:", makes, index=default_make_idx)

        available_models = brand_model_map[selected_make]
        default_model_idx = available_models.index("Golf") if "Golf" in available_models else 0
        selected_model = st.selectbox("Típus / Modell:", available_models, index=default_model_idx)

        year = st.slider("Gyártási év:", min_value=2002, max_value=2026, value=2016, step=1)
        hp = st.number_input("Teljesítmény (Lóerő / LE):", min_value=50, max_value=650, value=150, step=5)

    with col2:
        st.subheader("2. Műszaki paraméterek és hirdetési ár")
        mileage = st.number_input("Futott kilométer (km):", min_value=1000, max_value=450000, value=145000, step=5000)

        default_gear_idx = 1 if len(gears) > 1 and str(gears[0]).startswith("-") else 0
        gear = st.selectbox("Sebességváltó:", gears, index=default_gear_idx)

        default_fuel_idx = 1 if len(fuels) > 1 and str(fuels[0]).startswith("-") else 0
        fuel = st.selectbox("Üzemanyag típusa:", fuels, index=default_fuel_idx)

        listed_price = st.number_input(
            "Vizsgálandó hirdetési ár (Ft) – írd be az ajánlat ellenőrzéséhez (0 = csak becslés):",
            min_value=0,
            max_value=150000000,
            value=0,
            step=100000
        )

    st.divider()

    if st.button("📊 Reális piaci ár becslése és ajánlat elemzése", type="primary", use_container_width=True):
        predicted_price = predict_car_price(selected_make, selected_model, year, mileage, hp, gear, fuel)
        lower_bound = predicted_price * 0.90
        upper_bound = predicted_price * 1.10

        st.subheader(f"Piaci elemzés: {selected_make} {selected_model} ({year})")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            label="Becsült reális piaci ár",
            value=f"{predicted_price:,.0f} Ft",
            delta=f"≈ {predicted_price / eur_huf:,.0f} €",
            delta_color="off"
        )
        m2.metric(
            label="Reális kereskedelmi ársáv (±10%)",
            value=f"{lower_bound / 1_000_000:.2f}M – {upper_bound / 1_000_000:.2f}M Ft",
            delta="Normál piaci szórás",
            delta_color="off"
        )

        if listed_price > 0:
            diff_ft = listed_price - predicted_price
            diff_pct = (diff_ft / predicted_price) * 100
            m3.metric(
                label="Hirdetési ár eltérése a piactól",
                value=f"{listed_price:,.0f} Ft",
                delta=f"{diff_pct:+.1f}% ({diff_ft:+,.0f} Ft)",
                delta_color="inverse"
            )
        else:
            m3.metric(
                label="Ajánlat-ellenőrzés állapota",
                value="Nincs ár megadva",
                delta="Adj meg hirdetési árat fent!",
                delta_color="off"
            )

        # --- ANOMÁLIA-JELZÉS ÉS VIZUÁLIS ÁRSKÁLA ---
        if listed_price > 0:
            st.markdown("#### 🔎 Ajánlat minősítése (Anomália-detektor)")

            # Pozíció számítása a vizuális sávon (-30% és +30% között)
            clamped_pct = max(-30.0, min(30.0, diff_pct))
            marker_pos = ((clamped_pct + 30.0) / 60.0) * 100.0

            st.markdown(
                f"""
                <div style="margin: 12px 0 20px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 12px; color: #666; margin-bottom: 4px;">
                        <span>⚠️ Szokatlanul olcsó (&lt;-15%)</span>
                        <span>✅ Reális piaci sáv (±10%)</span>
                        <span>🚨 Szokatlanul drága (&gt;+15%)</span>
                    </div>
                    <div style="position: relative; height: 18px; border-radius: 9px;
                                background: linear-gradient(90deg, #f59e0b 0%, #fde047 22%, #22c55e 33%, #22c55e 67%, #fca5a5 78%, #ef4444 100%);">
                        <div style="position: absolute; left: calc({marker_pos}% - 6px); top: -4px;
                                    width: 12px; height: 26px; background: #111827; border: 2px solid white; border-radius: 4px;">
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            if diff_pct < -15:
                st.warning(
                    f"⚠️ **SZOKATLANUL OLCSÓ AJÁNLAT ({abs(diff_pct):.1f}%-kal a becsült piaci érték alatt!)**\n\n"
                    f"A hirdetett ár **{abs(diff_ft):,.0f} Ft-tal alacsonyabb** a hasonló műszaki paraméterű járművek átlagánál. "
                    f"Ez utalhat rendkívül gyors eladási szándékra, de fokozott kockázatot is jelenthet (rejtett műszaki hiba, sérült előélet, visszatekert kilométeróra)."
                )
            elif diff_pct > 15:
                st.error(
                    f"🚨 **SZOKATLANUL DRÁGA AJÁNLAT ({diff_pct:.1f}%-kal a becsült piaci érték felett!)**\n\n"
                    f"A hirdetett ár **{diff_ft:,.0f} Ft-tal magasabb** a reális piaci szintnél. "
                    f"Az ajánlat jelentősen túlárazott, hacsak nem igazolható ritka gyári csúcsfelszereltséggel vagy gyűjtői állapottal."
                )
            elif -15 <= diff_pct < -10:
                st.info(
                    f"💡 **Kedvező árazású ajánlat ({abs(diff_pct):.1f}%-kal a piaci átlag alatt)** – "
                    f"Érdemes megtekinteni, mert a reális piaci sáv alsó határa alatt van, de még nem szélsőségesen gyanús."
                )
            elif 10 < diff_pct <= 15:
                st.info(
                    f"ℹ️ **Enyhén túlárazott ajánlat ({diff_pct:.1f}%-kal a piaci átlag felett)** – "
                    f"Kissé a szokásos kereskedelmi sáv felett van, alkupozícióval reális szintre hozható."
                )
            else:
                st.success(
                    f"✅ **REÁLIS PIACI AJÁNLAT (eltérés: {diff_pct:+.1f}%)** – "
                    f"A hirdetési ár teljes mértékben összhangban van a jármű évjáratával, futásteljesítményével és műszaki adataival."
                )

        # --- TÖRTÉNETI ÉRTÉKVESZTÉSI GÖRBE ---
        st.divider()
        st.subheader("📉 Történeti alapú értékvesztési görbe (Futásteljesítmény hatása)")
        st.caption("A modell becslése ugyanerre az autóra különböző kilométeróra-állások mellett:")

        km_points = list(range(30000, 310000, 20000))
        curve_data = []
        for km_val in km_points:
            est_p = predict_car_price(selected_make, selected_model, year, km_val, hp, gear, fuel)
            curve_data.append({"Futásteljesítmény (km)": km_val, "Becsült piaci ár (Ft)": round(est_p)})

        curve_df = pd.DataFrame(curve_data).set_index("Futásteljesítmény (km)")
        st.line_chart(curve_df, y="Becsült piaci ár (Ft)", color="#2563eb")

# ==============================================================================
# 2. FÜL: TÖBB HIRDETÉS TÖMEGES SZŰRÉSE
# ==============================================================================
with tab2:
    st.subheader("Több járműhirdetés egyidejű ellenőrzése")
    st.write(
        "Írd át a táblázat sorait vagy adj hozzá új sorokat valós hirdetések adataival, majd kattints az ellenőrzésre. "
        "A rendszer minden hirdetéshez kiszámítja a reális árat, és megjelöli a szokatlanul olcsó vagy drága tételeket."
    )

    sample_make = "Volkswagen" if "Volkswagen" in brand_model_map else sorted(list(brand_model_map.keys()))[0]
    sample_model = "Golf" if "Golf" in brand_model_map[sample_make] else brand_model_map[sample_make][0]
    sample_gear = gears[1] if len(gears) > 1 and str(gears[0]).startswith("-") else gears[0]
    sample_fuel = fuels[1] if len(fuels) > 1 and str(fuels[0]).startswith("-") else fuels[0]

    base_est = predict_car_price(sample_make, sample_model, 2016, 145000, 150, sample_gear, sample_fuel)

    sample_listings = pd.DataFrame([
        {
            "Márka": sample_make,
            "Modell": sample_model,
            "Évjárat": 2016,
            "Kilométer": 145000,
            "Lóerő": 150,
            "Váltó": sample_gear,
            "Üzemanyag": sample_fuel,
            "Hirdetési ár (Ft)": round(base_est * 0.76, -4)
        },
        {
            "Márka": sample_make,
            "Modell": sample_model,
            "Évjárat": 2016,
            "Kilométer": 145000,
            "Lóerő": 150,
            "Váltó": sample_gear,
            "Üzemanyag": sample_fuel,
            "Hirdetési ár (Ft)": round(base_est * 0.99, -4)
        },
        {
            "Márka": sample_make,
            "Modell": sample_model,
            "Évjárat": 2016,
            "Kilométer": 145000,
            "Lóerő": 150,
            "Váltó": sample_gear,
            "Üzemanyag": sample_fuel,
            "Hirdetési ár (Ft)": round(base_est * 1.24, -4)
        }
    ])

    edited_df = st.data_editor(
        sample_listings,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Márka": st.column_config.SelectboxColumn("Márka", options=sorted(list(brand_model_map.keys())), required=True),
            "Évjárat": st.column_config.NumberColumn("Évjárat", min_value=2002, max_value=2026, step=1, required=True),
            "Kilométer": st.column_config.NumberColumn("Kilométer", min_value=1000, max_value=450000, step=5000, required=True),
            "Lóerő": st.column_config.NumberColumn("Lóerő", min_value=50, max_value=650, step=5, required=True),
            "Váltó": st.column_config.SelectboxColumn("Váltó", options=gears, required=True),
            "Üzemanyag": st.column_config.SelectboxColumn("Üzemanyag", options=fuels, required=True),
            "Hirdetési ár (Ft)": st.column_config.NumberColumn("Hirdetési ár (Ft)", min_value=100000, step=50000, required=True),
        }
    )

    if st.button("🔍 Hirdetéslista elemzése", type="primary"):
        results = []
        for _, row in edited_df.dropna().iterrows():
            est = predict_car_price(
                row["Márka"],
                row["Modell"],
                int(row["Évjárat"]),
                int(row["Kilométer"]),
                int(row["Lóerő"]),
                row["Váltó"],
                row["Üzemanyag"]
            )
            offer = float(row["Hirdetési ár (Ft)"])
            diff = ((offer - est) / est) * 100

            if diff < -15:
                status = "⚠️ Szokatlanul olcsó"
            elif diff > 15:
                status = "🚨 Szokatlanul drága"
            else:
                status = "✅ Reális ajánlat"

            results.append({
                "Jármű": f"{row['Márka']} {row['Modell']} ({int(row['Évjárat'])})",
                "Futás": f"{int(row['Kilométer']):,} km",
                "Hirdetési ár": f"{offer:,.0f} Ft",
                "Becsült reális ár": f"{est:,.0f} Ft",
                "Eltérés": f"{diff:+.1f}%",
                "Minősítés": status
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
