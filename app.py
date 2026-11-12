import streamlit as st
import pandas as pd
import joblib
import urllib.request
import json
import unicodedata

st.set_page_config(
    page_title="Használtautó Piaci Értékbecslő és Ajánlatelemző",
    page_icon="🚘",
    layout="wide"
)

# --- SZÖVEG NORMALIZÁLÁS (KISBETŰ + ÉKEZETMENTESÍTÉS) ---
def normalize_text(text):
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
    )

ALIASES = {
    "vw": "volkswagen",
    "merci": "mercedes",
    "mb": "mercedes",
    "bimmer": "bmw",
    "chev": "chevrolet",
}

def expand_query(query):
    words = normalize_text(query).split()
    expanded = [ALIASES.get(w, w) for w in words]
    return expanded

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

# Felesleges / ritka üzemanyagtípusok (pl. földgáz, CNG, LPG, -/-) kiszűrése
kizart_uzemanyagok = ("földgáz", "cng", "lpg", "hidrogén", "hydrogen", "etanol", "ethanol", "-/-", "egyéb", "other")
szurt_uzemanyagok = [f for f in bundle["fuels"] if not any(k in str(f).lower() for k in kizart_uzemanyagok)]
fuels = szurt_uzemanyagok if szurt_uzemanyagok else bundle["fuels"]

eur_huf, is_live_rate = get_eur_huf_rate()

# Összes (Márka, Modell) pár előkészítése a gyorskeresőhöz
ALL_VEHICLES = []
for b_name in sorted(brand_model_map.keys()):
    for m_name in sorted(brand_model_map[b_name]):
        ALL_VEHICLES.append((b_name, m_name))

# --- KIVITEL, ÁLLAPOT ÉS PIACI KALIBRÁCIÓS SZORZÓK ---
TRIMS = {
    "Alap kivitel (pl. Trendline, Basis, Access)": 0.95,
    "Közepes / Magas felszereltség (pl. Comfortline, Highline, Elegance)": 1.00,
    "Sportos kivitel (pl. GT, GT Sport, R-Line, S-Line, M-Sport)": 1.06,
    "Csúcsmodell / Sportváltozat (pl. GTI, GTD, R, RS, AMG, M)": 1.22,
}

CONDITIONS = {
    "Kiváló / Megkímélt, szervizkönyves": 1.06,
    "Normál / Korának és futásának megfelelő": 1.00,
    "Foglalkozós / Esztétikai vagy műszaki hibás": 0.82,
}

MARKETS = {
    "Magyar piac (Használtautó.hu szintre kalibrált)": 0.92,
    "Német / Nyugat-európai piac (nyers AutoScout24)": 1.00,
}

def predict_car_price(
    make,
    car_model,
    year,
    mileage,
    hp,
    gear,
    fuel,
    ccm=1600,
    trim_label="Közepes / Magas felszereltség (pl. Comfortline, Highline, Elegance)",
    condition_label="Normál / Korának és futásának megfelelő",
    market_label="Magyar piac (Használtautó.hu szintre kalibrált)"
):
    input_row = {col: 0 for col in feature_columns}
    input_row["year"] = year
    input_row["mileage"] = mileage
    input_row["hp"] = hp

    for col_name in [f"make_{make}", f"model_{car_model}", f"gear_{gear}", f"fuel_{fuel}"]:
        if col_name in input_row:
            input_row[col_name] = 1

    input_df = pd.DataFrame([input_row])
    base_price = float(model.predict(input_df)[0])

    # 1. Évjárat-extrapoláció (az AutoScout24 tanító adathalmaz 2011-2021 közötti tartományán kívülre)
    if year < 2011:
        age_factor = 0.93 ** (2011 - year)
    elif year > 2021:
        age_factor = 1.06 ** min(year - 2021, 5)
    else:
        age_factor = 1.00

    # 2. Magas futásteljesítmény korrekció (220 000 km felett meredekebb értékvesztés)
    extra_km = max(0, mileage - 220000)
    km_factor = max(0.75, 1.0 - (extra_km / 100000.0) * 0.08)

    # 3. Hengerűrtartalom finomhangolás
    ccm_factor = 1.0 + max(-0.03, min(0.06, (ccm - 1600) / 10000.0))

    # 4. Kivitel, állapot és célpiac szorzók
    trim_factor = TRIMS.get(trim_label, 1.0)
    condition_factor = CONDITIONS.get(condition_label, 1.0)
    market_factor = MARKETS.get(market_label, 0.92)

    return base_price * age_factor * km_factor * ccm_factor * trim_factor * condition_factor * market_factor

# --- FEJLÉC ÉS RENDSZERINFORMÁCIÓK ---
st.title("🚘 Használtautó Piaci Értékbecslő és Ajánlatelemző")
st.caption(
    "Nyilvános európai járműhirdetésekből (AutoScout24), részletes műszaki paraméterekből és történeti árakból tanított "
    "hibrid Random Forest piaci értékbecslő és anomália-detektor."
)

with st.expander("ℹ️ Hogyan működik a hibrid modell és a piaci kalibráció?"):
    info_c1, info_c2, info_c3 = st.columns(3)
    info_c1.markdown("**1. Gépi tanulási alap**\nTöbb tízezer nyilvános európai autóhirdetés történeti eladási árain tanított Random Forest modell.")
    info_c2.markdown("**2. Specifikus finomhangolás**\nA felszereltségi szint (pl. GT, GTI, S-Line), hengerűrtartalom, állapot és a 2011 alatti életkor-extrapoláció pontosítja az alapbecslést.")
    rate_label = f"{eur_huf:.1f} Ft (élő árfolyam)" if is_live_rate else f"{eur_huf:.1f} Ft (referencia)"
    info_c3.markdown(f"**3. Magyar piaci szorzó & küszöbök**\n1 EUR = {rate_label}. A rendszer a ±15%-on kívüli ajánlatokat szokatlanul olcsóként vagy drágaként jelöli.")

st.divider()

tab1, tab2 = st.tabs(["🎯 Egyedi jármű értékbecslése és ajánlat-ellenőrzés", "📋 Több hirdetés tömeges szűrése"])

# ==============================================================================
# 1. FÜL: EGYEDI ÉRTÉKBECSLÉS + ANOMÁLIA DETEKTOR + ÉRTÉKVESZTÉSI GÖRBE
# ==============================================================================
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Jármű és kivitel kiválasztása")

        search_query = st.text_input(
            "🔍 Gyorskereső (írd be kis- vagy nagybetűvel, pl. golf, vw golf, octavia, bmw 320):",
            placeholder="Kezdj el gépelni (pl. golf)..."
        )

        makes = sorted(list(brand_model_map.keys()))

        if search_query.strip():
            tokens = expand_query(search_query)
            raw_norm = normalize_text(search_query)

            matched = []
            for b_name, m_name in ALL_VEHICLES:
                combo_norm = normalize_text(f"{b_name} {m_name}")
                if all(tok in combo_norm for tok in tokens):
                    m_norm = normalize_text(m_name)
                    # Prioritási sorrend: 0 = pontos egyezés, 1 = ezzel kezdődik, 2 = tartalmazza
                    if m_norm == raw_norm or combo_norm == " ".join(tokens):
                        rank = 0
                    elif m_norm.startswith(raw_norm):
                        rank = 1
                    else:
                        rank = 2
                    matched.append((rank, b_name, m_name))

            matched.sort(key=lambda x: (x[0], x[1], x[2]))

            if matched:
                options_labels = [f"{b} – {m}" for _, b, m in matched]
                chosen_label = st.selectbox(
                    f"Találatok ({len(options_labels)} db):",
                    options_labels,
                    index=0
                )
                selected_make, selected_model = chosen_label.split(" – ", 1)
            else:
                st.info("Nincs pontos egyezés a modelllistában – válaszd ki a márkát alább, a modell automatikusan 'Egyéb'-re áll.")
                selected_make = st.selectbox("Márka:", makes, index=makes.index("Volkswagen") if "Volkswagen" in makes else 0)
                selected_model = "Egyéb"
        else:
            default_make_idx = makes.index("Volkswagen") if "Volkswagen" in makes else 0
            selected_make = st.selectbox("Márka:", makes, index=default_make_idx)

            raw_models = sorted(list(brand_model_map[selected_make]))
            available_models = raw_models + ["Egyéb / Listában nem szereplő modell"]

            # Kisbetű-független keresés az alapértelmezett 'Golf' modellre
            default_model_idx = 0
            for idx, m_item in enumerate(available_models):
                if normalize_text(m_item) == "golf":
                    default_model_idx = idx
                    break

            selected_model = st.selectbox("Típus / Modell:", available_models, index=default_model_idx)

        selected_trim = st.selectbox("Kivitel / Felszereltségi szint:", list(TRIMS.keys()), index=1)

        year = st.slider("Gyártási év (generáció):", min_value=2000, max_value=2026, value=2016, step=1)

        c_eng1, c_eng2 = st.columns(2)
        with c_eng1:
            ccm = st.number_input("Hengerűrtartalom (cm³):", min_value=800, max_value=6500, value=1968, step=50)
        with c_eng2:
            hp = st.number_input("Teljesítmény (Lóerő / LE):", min_value=50, max_value=650, value=150, step=5)

    with col2:
        st.subheader("2. Műszaki adatok, állapot és hirdetési ár")
        mileage = st.number_input("Futott kilométer (km):", min_value=1000, max_value=500000, value=145000, step=5000)

        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            default_gear_idx = 1 if len(gears) > 1 and str(gears[0]).startswith("-") else 0
            gear = st.selectbox("Sebességváltó:", gears, index=default_gear_idx)
        with c_tr2:
            fuel = st.selectbox("Üzemanyag típusa:", fuels, index=0)

        selected_condition = st.selectbox("Műszaki és esztétikai állapot:", list(CONDITIONS.keys()), index=1)
        selected_market = st.selectbox("Referenciapiac:", list(MARKETS.keys()), index=0)

        listed_price = st.number_input(
            "Vizsgálandó hirdetési ár (Ft) – ajánlat ellenőrzéséhez (0 = csak becslés):",
            min_value=0,
            max_value=150000000,
            value=0,
            step=100000
        )

    st.divider()

    if st.button("📊 Reális piaci ár becslése és ajánlat elemzése", type="primary", use_container_width=True):
        predicted_price = predict_car_price(
            selected_make, selected_model, year, mileage, hp, gear, fuel,
            ccm=ccm, trim_label=selected_trim, condition_label=selected_condition, market_label=selected_market
        )
        lower_bound = predicted_price * 0.90
        upper_bound = predicted_price * 1.10

        st.subheader(f"Piaci elemzés: {selected_make} {selected_model} ({year}) – {ccm} cm³, {hp} LE")

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
                    f"A hirdetett ár **{abs(diff_ft):,.0f} Ft-tal alacsonyabb** a hasonló műszaki paraméterű és kivitelű járművek átlagánál. "
                    f"Ez utalhat gyors eladási szándékra, de fokozott kockázatot is jelenthet (rejtett műszaki hiba, sérült előélet, visszatekert kilométeróra)."
                )
            elif diff_pct > 15:
                st.error(
                    f"🚨 **SZOKATLANUL DRÁGA AJÁNLAT ({diff_pct:.1f}%-kal a becsült piaci érték felett!)**\n\n"
                    f"A hirdetett ár **{diff_ft:,.0f} Ft-tal magasabb** a reális piaci szintnél. "
                    f"Az ajánlat jelentősen túlárazott a megadott évjárathoz, futáshoz és kivitelhez képest."
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
                    f"A hirdetési ár teljes mértékben összhangban van a jármű évjáratával, kivitelével és műszaki adataival."
                )

        # --- TÖRTÉNETI ÉRTÉKVESZTÉSI GÖRBE ---
        st.divider()
        st.subheader("📉 Történeti alapú értékvesztési görbe (Futásteljesítmény hatása)")
        st.caption("A kalibrált modell becslése ugyanerre a konfigurációra különböző kilométeróra-állások mellett:")

        km_points = list(range(40000, 360000, 20000))
        curve_data = []
        for km_val in km_points:
            est_p = predict_car_price(
                selected_make, selected_model, year, km_val, hp, gear, fuel,
                ccm=ccm, trim_label=selected_trim, condition_label=selected_condition, market_label=selected_market
            )
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
        "A rendszer a magyar piaci kalibrációval kiszámítja a reális árat, és megjelöli a szokatlanul olcsó vagy drága tételeket."
    )

    sample_make = "Volkswagen" if "Volkswagen" in brand_model_map else sorted(list(brand_model_map.keys()))[0]
    sample_model = "Golf" if "Golf" in brand_model_map[sample_make] else brand_model_map[sample_make][0]
    sample_gear = gears[1] if len(gears) > 1 and str(gears[0]).startswith("-") else gears[0]
    sample_fuel = fuels[0]

    base_est = predict_car_price(sample_make, sample_model, 2016, 145000, 150, sample_gear, sample_fuel, ccm=1968)

    sample_listings = pd.DataFrame([
        {
            "Márka": sample_make,
            "Modell": sample_model,
            "Évjárat": 2016,
            "Kilométer": 145000,
            "Köbcenti": 1968,
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
            "Köbcenti": 1968,
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
            "Köbcenti": 1968,
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
            "Évjárat": st.column_config.NumberColumn("Évjárat", min_value=2000, max_value=2026, step=1, required=True),
            "Kilométer": st.column_config.NumberColumn("Kilométer", min_value=1000, max_value=500000, step=5000, required=True),
            "Köbcenti": st.column_config.NumberColumn("Köbcenti (cm³)", min_value=800, max_value=6500, step=50, required=True),
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
                row["Üzemanyag"],
                ccm=int(row["Köbcenti"])
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
                "Motor": f"{int(row['Köbcenti'])} cm³, {int(row['Lóerő'])} LE",
                "Futás": f"{int(row['Kilométer']):,} km",
                "Hirdetési ár": f"{offer:,.0f} Ft",
                "Becsült reális ár": f"{est:,.0f} Ft",
                "Eltérés": f"{diff:+.1f}%",
                "Minősítés": status
            })

        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
