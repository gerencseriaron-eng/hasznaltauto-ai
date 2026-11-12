import streamlit as st
import pandas as pd
import joblib
import urllib.request
import json
import unicodedata
import re

st.set_page_config(
    page_title="Használtautó Piaci Értékbecslő és Ajánlatelemző",
    page_icon="🚘",
    layout="wide"
)

# --- SZÖVEG NORMALIZÁLÁS ---
def normalize_text(text):
    text = str(text).lower().strip()
    return "".join(
        c for c in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(c)
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
raw_brand_model_map = bundle["brand_model_map"]
gears = bundle["gears"]

# Felesleges üzemanyagtípusok kiszűrése
kizart_uzemanyagok = ("földgáz", "cng", "lpg", "hidrogén", "hydrogen", "etanol", "ethanol", "-/-", "egyéb", "other")
szurt_uzemanyagok = [f for f in bundle["fuels"] if not any(k in str(f).lower() for k in kizart_uzemanyagok)]
fuels = szurt_uzemanyagok if szurt_uzemanyagok else bundle["fuels"]

eur_huf, is_live_rate = get_eur_huf_rate()

# Alapvető európai modellek biztosítása minden népszerű márkához
EXTRA_MODELS = {
    "Volkswagen": ["Golf", "Passat", "Polo", "Tiguan", "Touran", "Caddy", "Transporter", "Jetta", "Bora", "Scirocco", "Arteon", "T-Roc", "Touareg", "Sharan"],
    "Skoda": ["Octavia", "Superb", "Fabia", "Rapid", "Scala", "Kodiaq", "Karoq", "Kamiq", "Yeti"],
    "Audi": ["A1", "A3", "A4", "A5", "A6", "A7", "A8", "Q3", "Q5", "Q7", "TT"],
    "BMW": ["116", "118", "120", "316", "318", "320", "330", "520", "525", "530", "X1", "X3", "X5"],
    "Mercedes-Benz": ["A-Klasse", "B-Klasse", "C-Klasse", "E-Klasse", "S-Klasse", "CLA", "GLA", "GLC", "GLE"],
    "Opel": ["Astra", "Corsa", "Insignia", "Vectra", "Zafira", "Mokka", "Meriva"],
    "Ford": ["Focus", "Mondeo", "Fiesta", "Kuga", "S-Max", "Galaxy", "Mustang"],
    "Toyota": ["Corolla", "Yaris", "Auris", "Avensis", "RAV4", "C-HR", "Prius"],
    "Suzuki": ["Swift", "Vitara", "SX4", "SX4 S-Cross", "Ignis"],
    "Honda": ["Civic", "Accord", "CR-V", "Jazz"],
    "Mazda": ["3", "6", "CX-3", "CX-5"],
    "Hyundai": ["i20", "i30", "i40", "Tucson", "Santa Fe"],
    "Kia": ["Ceed", "Sportage", "Rio", "Sorento"],
    "Renault": ["Megane", "Clio", "Laguna", "Scenic", "Captur"],
    "Peugeot": ["206", "207", "208", "307", "308", "508", "3008"],
    "Seat": ["Leon", "Ibiza", "Altea", "Ateca", "Toledo"],
    "Volvo": ["S40", "S60", "S80", "V40", "V50", "V60", "V70", "XC60", "XC90"],
}

BRAND_MODELS = {}
for b, m_list in raw_brand_model_map.items():
    BRAND_MODELS[b] = set(str(m).strip() for m in m_list if str(m).strip())

for b, extra_list in EXTRA_MODELS.items():
    if b in BRAND_MODELS:
        BRAND_MODELS[b].update(extra_list)
    else:
        BRAND_MODELS[b] = set(extra_list)

MAKES_LIST = sorted(BRAND_MODELS.keys())
for b in MAKES_LIST:
    BRAND_MODELS[b] = sorted(BRAND_MODELS[b]) + ["Egyéb / Listában nem szereplő modell"]

ALL_VEHICLES = []
for b in MAKES_LIST:
    for m in BRAND_MODELS[b]:
        if not m.startswith("Egyéb"):
            ALL_VEHICLES.append((b, m))

BRAND_ALIASES = {
    "vw": "Volkswagen",
    "volkswagen": "Volkswagen",
    "skoda": "Skoda",
    "audi": "Audi",
    "bmw": "BMW",
    "merci": "Mercedes-Benz",
    "mercedes": "Mercedes-Benz",
    "opel": "Opel",
    "ford": "Ford",
    "toyota": "Toyota",
    "suzuki": "Suzuki",
    "honda": "Honda",
    "mazda": "Mazda",
    "hyundai": "Hyundai",
    "kia": "Kia",
    "renault": "Renault",
    "peugeot": "Peugeot",
    "seat": "Seat",
    "volvo": "Volvo",
}

GENERATION_YEARS = {
    ("golf", "4"): 2002,
    ("golf", "iv"): 2002,
    ("golf", "5"): 2008,
    ("golf", "v"): 2008,
    ("golf", "6"): 2011,
    ("golf", "vi"): 2011,
    ("golf", "7"): 2016,
    ("golf", "vii"): 2016,
    ("golf", "8"): 2021,
    ("golf", "viii"): 2021,
    ("passat", "b6"): 2008,
    ("passat", "b7"): 2013,
    ("passat", "b8"): 2017,
    ("octavia", "1"): 2003,
    ("octavia", "2"): 2009,
    ("octavia", "3"): 2016,
    ("octavia", "4"): 2021,
    ("a4", "b7"): 2006,
    ("a4", "b8"): 2012,
    ("a4", "b9"): 2018,
}

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

# --- SESSION STATE INICIALIZÁLÁS ---
if "make_val" not in st.session_state:
    st.session_state["make_val"] = "Volkswagen" if "Volkswagen" in MAKES_LIST else MAKES_LIST[0]
if "model_val" not in st.session_state:
    st.session_state["model_val"] = "Golf" if "Golf" in BRAND_MODELS[st.session_state["make_val"]] else BRAND_MODELS[st.session_state["make_val"]][0]
if "year_val" not in st.session_state:
    st.session_state["year_val"] = 2016
if "trim_val" not in st.session_state:
    st.session_state["trim_val"] = list(TRIMS.keys())[1]
if "search_msg" not in st.session_state:
    st.session_state["search_msg"] = ""

def apply_smart_search():
    raw_q = st.session_state.get("smart_query_input", "").strip()
    if not raw_q:
        st.session_state["search_msg"] = ""
        return

    norm_q = normalize_text(raw_q)
    tokens = re.findall(r"[a-z]+|\d+", norm_q)

    detected_brand = None
    for tok in tokens:
        if tok in BRAND_ALIASES:
            detected_brand = BRAND_ALIASES[tok]
            break

    trim_keys = list(TRIMS.keys())
    if any(t in tokens for t in ["gti", "gtd", "rs", "amg", "cupra", "opc", "type"]):
        st.session_state["trim_val"] = trim_keys[3]
    elif any(t in tokens for t in ["gt", "sport", "sline", "rline", "msport", "fr"]):
        st.session_state["trim_val"] = trim_keys[2]
    elif any(t in tokens for t in ["trendline", "basis", "alap"]):
        st.session_state["trim_val"] = trim_keys[0]

    best_score = 0
    best_pair = None

    for b_name, m_name in ALL_VEHICLES:
        if detected_brand and b_name != detected_brand:
            continue

        m_norm = normalize_text(m_name)
        m_tokens = re.findall(r"[a-z]+|\d+", m_norm)
        score = 0

        if m_norm in norm_q and len(m_norm) >= 2:
            score += 10 + len(m_norm)

        for tok in tokens:
            if tok in BRAND_ALIASES:
                continue
            if tok in m_tokens:
                score += 8
            elif len(tok) >= 3 and any(mt.startswith(tok) for mt in m_tokens):
                score += 5

        if score > best_score:
            best_score = score
            best_pair = (b_name, m_name)

    if best_pair and best_score >= 5:
        b_found, m_found = best_pair
        st.session_state["make_val"] = b_found
        st.session_state["model_val"] = m_found

        m_key = normalize_text(m_found)
        for tok in tokens:
            if (m_key, tok) in GENERATION_YEARS:
                st.session_state["year_val"] = GENERATION_YEARS[(m_key, tok)]
                break

        st.session_state["search_msg"] = f"✅ Kiválasztva: **{b_found} {m_found}** (Évjárat: {st.session_state['year_val']})"
    elif detected_brand:
        st.session_state["make_val"] = detected_brand
        st.session_state["model_val"] = "Egyéb / Listában nem szereplő modell"
        st.session_state["search_msg"] = f"ℹ️ Márka beállítva: **{detected_brand}** (a pontos modell nem szerepel a listában, 'Egyéb' mód aktív)."
    else:
        st.session_state["search_msg"] = "⚠️ Nem sikerült pontos modellt azonosítani – válaszd ki a márkát és modellt közvetlenül a két legördülő menüből."

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

    # A Random Forest fák 220 000 km felett ritka és zajos adatokon (elírásokon) alapulnak,
    # ezért az alapbecslést 220 000 km-nél rögzítjük, felette pedig folytonos görbével csökkentjük.
    rf_mileage = min(mileage, 220000)
    input_row["mileage"] = rf_mileage
    input_row["hp"] = hp

    make_col = f"make_{make}"
    if make_col in input_row:
        input_row[make_col] = 1

    target_m = normalize_text(car_model)
    for col in feature_columns:
        if col.startswith("model_") and normalize_text(col[6:]) == target_m:
            input_row[col] = 1
            break

    for col_name in [f"gear_{gear}", f"fuel_{fuel}"]:
        if col_name in input_row:
            input_row[col_name] = 1

    input_df = pd.DataFrame([input_row])
    base_price = float(model.predict(input_df)[0])

    # 1. Évjárat-extrapoláció (2011 alatti és 2021 feletti modellekre)
    if year < 2011:
        age_factor = 0.93 ** (2011 - year)
    elif year > 2021:
        age_factor = 1.06 ** min(year - 2021, 5)
    else:
        age_factor = 1.00

    # 2. Folytonos magas futásteljesítmény-csökkenés (220 000 km felett minden 10 000 km -1.5%)
    extra_km = max(0, mileage - 220000)
    km_factor = 0.985 ** (extra_km / 10000.0)

    # 3. Hengerűrtartalom finomhangolás
    ccm_factor = 1.0 + max(-0.03, min(0.06, (ccm - 1600) / 10000.0))

    # 4. Kivitel, állapot és célpiac szorzók
    trim_factor = TRIMS.get(trim_label, 1.0)
    condition_factor = CONDITIONS.get(condition_label, 1.0)
    market_factor = MARKETS.get(market_label, 0.92)

    return base_price * age_factor * km_factor * ccm_factor * trim_factor * condition_factor * market_factor

# --- FEJLÉC ---
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
    with st.form("quick_search_form", border=True):
        q_col1, q_col2 = st.columns([4, 1])
        with q_col1:
            st.text_input(
                "🔍 Gyorskereső (írd be szabadszavas formában, pl. golf 5, vw golf 5 gt, octavia rs, bmw 320, majd nyomj Entert):",
                key="smart_query_input",
                placeholder="Pl.: golf 5 gt"
            )
        with q_col2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            st.form_submit_button("Keresés", on_click=apply_smart_search, use_container_width=True)

    if st.session_state["search_msg"]:
        st.caption(st.session_state["search_msg"])

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Jármű és kivitel kiválasztása")

        selected_make = st.selectbox("Márka:", MAKES_LIST, key="make_val")

        valid_models = BRAND_MODELS[selected_make]
        if st.session_state["model_val"] not in valid_models:
            st.session_state["model_val"] = valid_models[0]

        selected_model = st.selectbox("Típus / Modell:", valid_models, key="model_val")

        selected_trim = st.selectbox("Kivitel / Felszereltségi szint:", list(TRIMS.keys()), key="trim_val")

        year = st.slider("Gyártási év (generáció):", min_value=2000, max_value=2026, key="year_val", step=1)

        c_eng1, c_eng2 = st.columns(2)
        with c_eng1:
            ccm = st.number_input("Hengerűrtartalom (cm³):", min_value=800, max_value=6500, value=1968, step=50)
        with c_eng2:
            hp = st.number_input("Teljesítmény (Lóerő / LE):", min_value=50, max_value=650, value=140, step=5)

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

        # --- TÖRTÉNETI ÉRTÉKVESZTÉSI GÖRBE (MONOTONITÁS-KÉNYSZERÍTÉSSEL) ---
        st.divider()
        st.subheader("📉 Történeti alapú értékvesztési görbe (Futásteljesítmény hatása)")
        st.caption("A kalibrált modell becslése ugyanerre a konfigurációra különböző kilométeróra-állások mellett:")

        km_points = list(range(40000, 360000, 20000))
        curve_data = []
        prev_price = None

        for km_val in km_points:
            est_p = predict_car_price(
                selected_make, selected_model, year, km_val, hp, gear, fuel,
                ccm=ccm, trim_label=selected_trim, condition_label=selected_condition, market_label=selected_market
            )
            # Monoton csökkenés biztosítása a döntési fák apró lépcsőzései ellen
            if prev_price is not None and est_p >= prev_price:
                est_p = prev_price * 0.992
            prev_price = est_p
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

    sample_make = "Volkswagen"
    sample_model = "Golf"
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
            "Márka": st.column_config.SelectboxColumn("Márka", options=MAKES_LIST, required=True),
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
