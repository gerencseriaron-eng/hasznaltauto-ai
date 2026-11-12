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

UNSET_MAKE = "-- Válassz márkát --"
UNSET_OPT = "-- Nincs megadva --"

YEARS_LIST = [UNSET_OPT] + list(range(2026, 1999, -1))
FUELS_LIST = [UNSET_OPT] + fuels
GEARS_LIST = [UNSET_OPT] + gears

# Alapvető európai modellek biztosítása minden népszerű márkához
EXTRA_MODELS = {
    "Volkswagen": ["Golf", "Passat", "Polo", "Tiguan", "Touran", "Caddy", "Transporter", "Jetta", "Bora", "Scirocco", "Arteon", "T-Roc", "Touareg", "Sharan", "ID.3", "ID.4", "up!"],
    "Skoda": ["Octavia", "Superb", "Fabia", "Rapid", "Scala", "Kodiaq", "Karoq", "Kamiq", "Yeti", "Enyaq", "Citigo"],
    "Audi": ["A1", "A3", "A4", "A5", "A6", "A7", "A8", "Q3", "Q5", "Q7", "TT", "e-tron"],
    "BMW": ["116", "118", "120", "316", "318", "320", "330", "520", "525", "530", "X1", "X3", "X5", "i3", "i4", "iX3"],
    "Mercedes-Benz": ["A-Klasse", "B-Klasse", "C-Klasse", "E-Klasse", "S-Klasse", "CLA", "GLA", "GLC", "GLE", "EQA", "EQC"],
    "Opel": ["Astra", "Corsa", "Insignia", "Vectra", "Zafira", "Mokka", "Meriva"],
    "Ford": ["Focus", "Mondeo", "Fiesta", "Kuga", "S-Max", "Galaxy", "Mustang"],
    "Toyota": ["Corolla", "Yaris", "Auris", "Avensis", "RAV4", "C-HR", "Prius"],
    "Suzuki": ["Swift", "Vitara", "SX4", "SX4 S-Cross", "Ignis"],
    "Honda": ["Civic", "Accord", "CR-V", "Jazz"],
    "Mazda": ["3", "6", "CX-3", "CX-5"],
    "Hyundai": ["i20", "i30", "i40", "Tucson", "Santa Fe", "Ioniq", "Ioniq 5", "Kona"],
    "Kia": ["Ceed", "Sportage", "Rio", "Sorento", "Niro", "EV6"],
    "Renault": ["Megane", "Clio", "Laguna", "Scenic", "Captur", "Zoe"],
    "Peugeot": ["206", "207", "208", "307", "308", "508", "3008"],
    "Seat": ["Leon", "Ibiza", "Altea", "Ateca", "Toledo"],
    "Volvo": ["S40", "S60", "S80", "V40", "V50", "V60", "V70", "XC60", "XC90"],
    "Nissan": ["Leaf", "Qashqai", "Juke", "X-Trail"],
    "Tesla": ["Model 3", "Model Y", "Model S", "Model X"],
}

COMPACT_MODELS = {
    "i3", "up!", "citigo", "mii", "zoe", "leaf", "spring", "fortwo", "forfour",
    "aygo", "yaris", "polo", "fabia", "fiesta", "corsa", "swift", "jazz",
    "206", "207", "208", "clio", "ibiza", "a1", "500", "panda", "ignis", "rio", "i20"
}

BRAND_MODELS = {}
for b, m_list in raw_brand_model_map.items():
    BRAND_MODELS[b] = set(str(m).strip() for m in m_list if str(m).strip())

for b, extra_list in EXTRA_MODELS.items():
    if b in BRAND_MODELS:
        BRAND_MODELS[b].update(extra_list)
    else:
        BRAND_MODELS[b] = set(extra_list)

MAKES_LIST = [UNSET_MAKE] + sorted(BRAND_MODELS.keys())
for b in sorted(BRAND_MODELS.keys()):
    BRAND_MODELS[b] = [UNSET_OPT] + sorted(BRAND_MODELS[b]) + ["Egyéb / Listában nem szereplő modell"]

# --- KIVITEL, ÁLLAPOT ÉS PIACI KALIBRÁCIÓS SZORZÓK ---
TRIMS = {
    UNSET_OPT: 1.00,
    "Alap kivitel (pl. Trendline, Basis, Access)": 0.95,
    "Közepes / Magas felszereltség (pl. Comfortline, Highline, Elegance)": 1.00,
    "Sportos kivitel (pl. GT, GT Sport, R-Line, S-Line, M-Sport)": 1.06,
    "Csúcsmodell / Sportváltozat (pl. GTI, GTD, R, RS, AMG, M)": 1.22,
}

CONDITIONS = {
    UNSET_OPT: 1.00,
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
    car_model=None,
    year=None,
    mileage=None,
    hp=None,
    gear=None,
    fuel=None,
    ccm=None,
    battery_kwh=None,
    is_ev=False,
    trim_label=UNSET_OPT,
    condition_label=UNSET_OPT,
    market_label="Magyar piac (Használtautó.hu szintre kalibrált)"
):
    input_row = {col: 0 for col in feature_columns}

    # Ha a felhasználó nem adott meg évjáratot/km-t/lóerőt, semleges piaci átlaggal számolunk
    eff_year = int(year) if year not in (None, UNSET_OPT) else 2016
    eff_mileage = int(mileage) if mileage is not None else 140000

    has_specific_model = car_model not in (None, UNSET_OPT) and not str(car_model).startswith("Egyéb")
    target_m = normalize_text(car_model) if has_specific_model else ""
    is_compact = target_m in COMPACT_MODELS

    default_hp = 95 if is_compact else 135
    eff_hp = int(hp) if hp is not None else default_hp

    input_row["year"] = eff_year
    input_row["mileage"] = min(eff_mileage, 220000)
    input_row["hp"] = eff_hp

    make_col = f"make_{make}"
    if make_col in input_row:
        input_row[make_col] = 1

    model_matched = False
    if has_specific_model:
        for col in feature_columns:
            if col.startswith("model_") and normalize_text(col[6:]) == target_m:
                input_row[col] = 1
                model_matched = True
                break

    if gear not in (None, UNSET_OPT):
        g_col = f"gear_{gear}"
        if g_col in input_row:
            input_row[g_col] = 1

    if fuel not in (None, UNSET_OPT):
        f_col = f"fuel_{fuel}"
        if f_col in input_row:
            input_row[f_col] = 1

    input_df = pd.DataFrame([input_row])
    base_price = float(model.predict(input_df)[0])

    # 1. Szegmens korrekció, ha konkrét modell lett kiválasztva, de nem szerepelt a .pkl oszlopai között
    if has_specific_model and not model_matched:
        segment_factor = 0.52 if is_compact else 0.85
    else:
        segment_factor = 1.00

    # 2. Évjárat-extrapoláció (2011 alatti és 2021 feletti modellekre)
    if eff_year < 2011:
        age_factor = 0.93 ** (2011 - eff_year)
    elif eff_year > 2021:
        age_factor = 1.06 ** min(eff_year - 2021, 5)
    else:
        age_factor = 1.00

    # 3. Folytonos magas futásteljesítmény-csökkenés (220 000 km felett)
    extra_km = max(0, eff_mileage - 220000)
    km_factor = 0.985 ** (extra_km / 10000.0)

    # 4. Hajtáslánc finomhangolás (csak ha a felhasználó megadta a cm³ / kWh adatot)
    if is_ev:
        if battery_kwh is not None:
            powertrain_factor = (1.0 + max(-0.08, min(0.15, (battery_kwh - 50) / 250.0))) * 0.86
        else:
            powertrain_factor = 0.86
    else:
        if ccm is not None:
            powertrain_factor = 1.0 + max(-0.03, min(0.06, (ccm - 1600) / 10000.0))
        else:
            powertrain_factor = 1.00

    # 5. Kivitel, állapot és célpiac szorzók
    trim_factor = TRIMS.get(trim_label, 1.0)
    condition_factor = CONDITIONS.get(condition_label, 1.0)
    market_factor = MARKETS.get(market_label, 0.92)

    return base_price * segment_factor * age_factor * km_factor * powertrain_factor * trim_factor * condition_factor * market_factor

# --- FEJLÉC ---
st.title("🚘 Használtautó Piaci Értékbecslő és Ajánlatelemző")
st.caption(
    "Nyilvános európai járműhirdetésekből (AutoScout24), műszaki paraméterekből és történeti árakból tanított "
    "hibrid Random Forest piaci értékbecslő és anomália-detektor. Csak azokat a mezőket töltsd ki, amelyek alapján szűrni szeretnél."
)

with st.expander("ℹ️ Hogyan működik a hibrid modell és az opcionális szűrés?"):
    info_c1, info_c2, info_c3 = st.columns(3)
    info_c1.markdown("**1. Gépi tanulási alap**\nTöbb tízezer nyilvános európai autóhirdetés történeti eladási árain tanított Random Forest modell.")
    info_c2.markdown("**2. Rugalmas szűrés**\nCsak a kiválasztott paramétereket veszi figyelembe; a meg nem adott mezőknél semleges piaci átlaggal számol.")
    rate_label = f"{eur_huf:.1f} Ft (élő árfolyam)" if is_live_rate else f"{eur_huf:.1f} Ft (referencia)"
    info_c3.markdown(f"**3. Magyar piaci szorzó & küszöbök**\n1 EUR = {rate_label}. A rendszer a ±15%-on kívüli ajánlatokat szokatlanul olcsóként vagy drágaként jelöli.")

st.divider()

# ==============================================================================
# 1. JÁRMŰ ADATAI (ALAPÉRTELMEZETTEN ÜRES / NINCS MEGADVA)
# ==============================================================================
st.subheader("1. Jármű adatai")
r1_c1, r1_c2, r1_c3 = st.columns([1, 1, 1.3])

with r1_c1:
    selected_make = st.selectbox("Márka:", MAKES_LIST, index=0)

with r1_c2:
    if selected_make == UNSET_MAKE:
        selected_model = st.selectbox("Modell:", [UNSET_OPT], index=0, disabled=True)
    else:
        available_models = BRAND_MODELS[selected_make]
        selected_model = st.selectbox(
            "Modell:",
            available_models,
            index=0,
            key=f"model_select_{selected_make}"
        )

with r1_c3:
    selected_trim = st.selectbox("Kivitel / Felszereltség:", list(TRIMS.keys()), index=0)

r2_c1, r2_c2, r2_c3 = st.columns([1, 1, 1.3])

with r2_c1:
    year = st.selectbox("Évjárat:", YEARS_LIST, index=0)

with r2_c2:
    mileage = st.number_input(
        "Futott kilométer (km):",
        min_value=0,
        max_value=600000,
        value=None,
        step=5000,
        placeholder="Nincs megadva (pl. 145000)"
    )

with r2_c3:
    selected_condition = st.selectbox("Jármű állapota:", list(CONDITIONS.keys()), index=0)

# ==============================================================================
# 2. MOTOR ÉS HAJTÁSLÁNC
# ==============================================================================
st.subheader("2. Motor és hajtáslánc")
r3_c1, r3_c2, r3_c3, r3_c4 = st.columns(4)

with r3_c1:
    fuel = st.selectbox("Üzemanyag:", FUELS_LIST, index=0)

is_ev = (fuel != UNSET_OPT) and (("elektromos" in normalize_text(fuel)) or ("elektro" in normalize_text(fuel)))

with r3_c2:
    if is_ev:
        auto_idx = next((i for i, g in enumerate(GEARS_LIST) if "auto" in normalize_text(g)), 0)
        gear = st.selectbox("Sebességváltó:", GEARS_LIST, index=auto_idx, disabled=True)
    else:
        gear = st.selectbox("Sebességváltó:", GEARS_LIST, index=0)

with r3_c3:
    if is_ev:
        ccm = None
        battery_kwh = st.number_input(
            "Akkumulátor (kWh):",
            min_value=10,
            max_value=150,
            value=None,
            step=2,
            placeholder="Nincs megadva (pl. 42)"
        )
    else:
        battery_kwh = None
        ccm = st.number_input(
            "Hengerűrtartalom (cm³):",
            min_value=500,
            max_value=7000,
            value=None,
            step=50,
            placeholder="Nincs megadva (pl. 1968)"
        )

with r3_c4:
    hp = st.number_input(
        "Teljesítmény (LE):",
        min_value=30,
        max_value=1000,
        value=None,
        step=5,
        placeholder="Nincs megadva (pl. 140)"
    )

# ==============================================================================
# 3. PIACI REFERENCIA ÉS HIRDETÉSI ÁR ELLENŐRZÉSE
# ==============================================================================
st.subheader("3. Piaci referencia és hirdetési ár ellenőrzése")
r4_c1, r4_c2 = st.columns(2)

with r4_c1:
    selected_market = st.selectbox("Referenciapiac:", list(MARKETS.keys()), index=0)

with r4_c2:
    listed_price = st.number_input(
        "Vizsgálandó hirdetési ár (Ft) – ajánlat ellenőrzéséhez:",
        min_value=0,
        max_value=200000000,
        value=None,
        step=100000,
        placeholder="Opcionális – írd be a hirdetés árát"
    )

st.divider()

# ==============================================================================
# EREDMÉNYEK ÉS ANOMÁLIA-DETEKTOR
# ==============================================================================
if st.button("📊 Reális piaci ár becslése és ajánlat elemzése", type="primary", use_container_width=True):
    if selected_make == UNSET_MAKE:
        st.warning("⚠️ Kérlek válassz ki legalább egy **Márkát** az értékbecslés indításához!")
    else:
        predicted_price = predict_car_price(
            selected_make,
            car_model=selected_model,
            year=year,
            mileage=mileage,
            hp=hp,
            gear=gear,
            fuel=fuel,
            ccm=ccm,
            battery_kwh=battery_kwh,
            is_ev=is_ev,
            trim_label=selected_trim,
            condition_label=selected_condition,
            market_label=selected_market
        )
        lower_bound = predicted_price * 0.90
        upper_bound = predicted_price * 1.10

        # Dinamikus cím összeállítása csak a ténylegesen megadott paraméterekből
        title_parts = [selected_make]
        if selected_model != UNSET_OPT:
            title_parts.append(selected_model)
        if year != UNSET_OPT:
            title_parts.append(f"({year})")

        spec_parts = []
        if fuel != UNSET_OPT:
            spec_parts.append(fuel)
        if is_ev and battery_kwh is not None:
            spec_parts.append(f"{battery_kwh} kWh")
        elif not is_ev and ccm is not None:
            spec_parts.append(f"{ccm} cm³")
        if hp is not None:
            spec_parts.append(f"{hp} LE")
        if mileage is not None:
            spec_parts.append(f"{mileage:,} km")

        header_text = " ".join(title_parts)
        if spec_parts:
            header_text += " – " + ", ".join(spec_parts)

        st.subheader(f"Piaci elemzés: {header_text}")

        # Jelzés, ha bizonyos alapadatok nem lettek kitöltve
        assumed = []
        if year == UNSET_OPT:
            assumed.append("évjárat (2016)")
        if mileage is None:
            assumed.append("futásteljesítmény (140 000 km)")
        if hp is None:
            assumed.append("teljesítmény (piaci átlag)")
        if assumed:
            st.caption(f"ℹ️ A meg nem adott mezőknél átlagos piaci referenciaértékekkel számolt a modell: {', '.join(assumed)}.")

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

        if listed_price is not None and listed_price > 0:
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
                delta="Opcionálisan megadható fent",
                delta_color="off"
            )

        # --- ANOMÁLIA-JELZÉS ÉS VIZUÁLIS ÁRSKÁLA ---
        if listed_price is not None and listed_price > 0:
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
                    f"A hirdetett ár **{abs(diff_ft):,.0f} Ft-tal alacsonyabb** a hasonló paraméterű járművek átlagánál. "
                    f"Ez utalhat gyors eladási szándékra, de fokozott kockázatot is jelenthet (rejtett műszaki hiba, sérült előélet, visszatekert kilométeróra)."
                )
            elif diff_pct > 15:
                st.error(
                    f"🚨 **SZOKATLANUL DRÁGA AJÁNLAT ({diff_pct:.1f}%-kal a becsült piaci érték felett!)**\n\n"
                    f"A hirdetett ár **{diff_ft:,.0f} Ft-tal magasabb** a reális piaci szintnél. "
                    f"Az ajánlat jelentősen túlárazott a megadott szűrőfeltételekhez képest."
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
                    f"A hirdetési ár teljes mértékben összhangban van a megadott járműparaméterekkel."
                )

        # --- TÖRTÉNETI ÉRTÉKVESZTÉSI GÖRBE ---
        st.divider()
        st.subheader("📉 Történeti alapú értékvesztési görbe (Futásteljesítmény hatása)")
        st.caption("A modell becslése a kiválasztott szűrőfeltételekre különböző kilométeróra-állások mellett:")

        km_points = list(range(40000, 360000, 20000))
        curve_data = []
        prev_price = None

        for km_val in km_points:
            est_p = predict_car_price(
                selected_make,
                car_model=selected_model,
                year=year,
                mileage=km_val,
                hp=hp,
                gear=gear,
                fuel=fuel,
                ccm=ccm,
                battery_kwh=battery_kwh,
                is_ev=is_ev,
                trim_label=selected_trim,
                condition_label=selected_condition,
                market_label=selected_market
            )
            if prev_price is not None and est_p >= prev_price:
                est_p = prev_price * 0.992
            prev_price = est_p
            curve_data.append({"Futásteljesítmény (km)": km_val, "Becsült piaci ár (Ft)": round(est_p)})

        curve_df = pd.DataFrame(curve_data).set_index("Futásteljesítmény (km)")
        st.line_chart(curve_df, y="Becsült piaci ár (Ft)", color="#2563eb")
