import io
import re
import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
from rapidfuzz import process, fuzz
import html
import json
from groq import Groq
import os
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ========== APP SETUP ==========
st.set_page_config(page_title="Car features comparison", layout="wide")
st.title("🚗 Car features comparison")
date_str = datetime.now().strftime("%Y%m%d")

# ========== NORMALIZATION / UTILITIES ==========
NORMALIZATION_MAP = {
    "Fuel-Efficiency (km/l)*": "Fuel Efficiency (km/l)",
    "wheels and tyre": "Tyre",
    "wheel and tyre": "Tyre",
    "tyre": "Tyre",
    "safety security": "Safety",
    "safety and security": "Safety",
    "safety": "Safety",
    "infotainment & connectivity": "Infotainment",
    "comfort & convenience": "Comfort & Convenience",
    "comfort and convenience": "Comfort & Convenience",
    "brake": "Brakes",
    "brakes": "Brakes",
    "suspension": "Suspension",
    "steering": "Steering",
    "wheels": "Tyre",
    "dimensions": "Dimensions",
    "fuel": "Fuel",
    "engine": "Engine",
    "transmission": "Transmission",
    "exterior":"Exterior",
    "interior":"Interior",
    "instrument panel & center fascia display":"Infotainment",
    "audio":"Audio & Entertainment",
    "air conditioner":"Air conditioning"
}

NORMALIZED_KEYS = {k.lower(): v for k, v in NORMALIZATION_MAP.items()}
NORMALIZED_VALUES = list(set(NORMALIZATION_MAP.values()))

def normalize_label(label: str, threshold: int = 86) -> str:
    if not label:
        return label

    norm = label.lower().strip()
    norm = re.sub(r"[\-&_/]", " ", norm)
    norm = re.sub(r"\s+", " ", norm).strip()

    # Exact match first
    if norm in NORMALIZED_KEYS:
        return NORMALIZED_KEYS[norm]

    # Fuzzy match against KEYS, not values
    best_match, score, _ = process.extractOne(norm, list(NORMALIZED_KEYS.keys()), scorer=fuzz.token_sort_ratio)
    if score >= threshold:
        return NORMALIZED_KEYS[best_match]

    return label.strip()


def yes_no_formatter(val):
    if pd.isna(val) or val is None:
        return "-"
    sval = str(val).strip()
    if sval.lower() in ["yes", "y", "true", "available", "standard"]:
        return "✅ Yes"
    if sval.lower() in ["no", "n", "false", "not available", "na", "-"]:
        return "❌ No"
    return sval

def wrap_text(df, first_col_width=260, other_col_width=160):
    styles = [
        {'selector': 'th', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]},
        {'selector': 'td', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]}
    ]
    table_styles = []
    for i, col in enumerate(df.columns):
        width = first_col_width if i == 0 else other_col_width
        table_styles.append({'selector': f'th.col{i}', 'props': [('min-width', f'{width}px')]})
        table_styles.append({'selector': f'td.col{i}', 'props': [('min-width', f'{width}px')]})
    return df.style.set_properties(**{'white-space': 'pre-wrap', 'word-wrap': 'break-word'}).set_table_styles(styles + table_styles)

# ========== HYUNDAI ==========
HY_HEADERS = {"User-Agent": "Mozilla/5.0"}

@st.cache_data(ttl=3600)
def hy_get_models():
    url = "https://api.hyundai.co.in/service/test-drive/getModels"
    resp = requests.get(url, headers=HY_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()

def slugify_model(name: str) -> str:
    name = name.lower()
    name = re.sub(r'^(all\s+new|new)\s+', '', name)
    name = re.sub(r'[^a-z0-9\s]', '', name)
    return name.strip().replace(" ", "-")

def hy_features_url(model_desc: str) -> str:
    return f"https://www.hyundai.com/in/en/find-a-car/{slugify_model(model_desc)}/features"

def _parse_hy_table(table_soup) -> pd.DataFrame:
    thead = table_soup.find("thead")
    tbody = table_soup.find("tbody")
    if not thead or not tbody:
        return pd.DataFrame()

    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    if not headers or len(headers) < 2:
        return pd.DataFrame()

    rows = []
    for tr in tbody.find_all("tr"):
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if not cells:
            continue
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(cells)
    df = pd.DataFrame(rows, columns=headers)
    return df

@st.cache_data(ttl=3600)
def hy_fetch_flat_features(model_desc: str):
    url = hy_features_url(model_desc)
    r = requests.get(url, headers=HY_HEADERS, timeout=40)
    if r.status_code != 200:
        return pd.DataFrame(columns=["brand","car","variant","sub_category","feature","value"])
    soup = BeautifulSoup(r.text, "html.parser")

    tables = soup.find_all("table", class_="table-hover")
    fallback_names = [
        "Engine & Trim Plan",
        "Safety", "Exterior", "Interior", "Infotainment & Connectivity", "Comfort & Convenience"
    ]

    section_titles = []
    for tbl in tables:
        title = None
        prev = tbl
        for _ in range(5):
            prev = prev.find_previous(["h2", "h3", "h4", "h5"]) if prev else None
            if prev and prev.get_text(strip=True):
                title = prev.get_text(strip=True)
                break
        section_titles.append(title or "")

    if not any(t for t in section_titles) and len(tables) == len(fallback_names):
        section_titles = fallback_names
    elif len(section_titles) != len(tables):
        section_titles = [t if t else "" for t in section_titles]
        section_titles += [""] * (len(tables) - len(section_titles))

    flat_rows = []
    for tbl, raw_title in zip(tables, section_titles):
        cat_title = (raw_title or "").strip()
        if re.search(r"engine\s*&?\s*trim", cat_title, flags=re.I):
            continue

        sub_category = normalize_label(cat_title) if cat_title else "Features"

        df = _parse_hy_table(tbl)
        if df.empty:
            continue

        feature_col = df.columns[0]
        variant_cols = list(df.columns[1:])

        for _, row in df.iterrows():
            feature = row.get(feature_col, "").strip()
            if not feature:
                continue
            for vcol in variant_cols:
                val = row.get(vcol, "")
                flat_rows.append({
                    "brand": "Hyundai",
                    "car": model_desc,
                    "variant": vcol.strip() if isinstance(vcol, str) else str(vcol),
                    "category": "Features",
                    "sub_category": normalize_label(sub_category),
                    "feature": normalize_label(feature.strip()),
                    "value": str(val).strip()
                })
    df = pd.DataFrame(flat_rows)
    df=df[df["sub_category"].str.lower() != "features"]
    return df

# ========== MARUTI ==========
MS_API = {
    "Nexa": {
        "cars_url": "https://www.nexaexperience.com/graphql/execute.json/msil-platform/NexaCarList",
        "variant_url": "https://www.nexaexperience.com/graphql/execute.json/msil-platform/VariantDetailCompare;modelCd={};channel=EXC;locale=en;"
    },
    "Arena": {
        "cars_url": "https://www.marutisuzuki.com/graphql/execute.json/msil-platform/ArenaCarList",
        "variant_url": "https://www.marutisuzuki.com/graphql/execute.json/msil-platform/ArenaVariantDetailCompare;modelCd={};channel=NRM;locale=en;"
    }
}
MS_HEADERS = {"accept": "application/json, */*", "user-agent": "Mozilla/5.0"}

@st.cache_data(ttl=3600)
def ms_fetch_cars(channel: str):
    r = requests.get(MS_API[channel]["cars_url"], headers=MS_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["data"]["carModelList"]["items"]

@st.cache_data(ttl=3600)
def ms_fetch_variant_details(channel: str, modelCd: str):
    url = MS_API[channel]["variant_url"].format(modelCd)
    r = requests.get(url, headers=MS_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data
def ms_parse_features(data, channel: str, model_desc: str):
    rows = []
    labels = {}
    try:
        labels_list = data["data"]["variantSpecificationsLabelsList"]["items"][0]
        labels = {
            key.replace("_label", ""): val
            for key, val in labels_list.items()
            if key.endswith("_label")
        }
    except Exception:
        pass

    variants = data["data"]["carModelList"]["items"][0]["variants"]
    for v in variants:
        vname = v.get("variantName", "-")
        for cat in v.get("specificationCategory", []):
            category_name = normalize_label(cat.get("categoryName", ""))
            for aspect in cat.get("specificationAspect", []):
                sub_category = normalize_label(aspect.get("categoryLabel", ""))
                for feature, value in aspect.items():
                    if feature in ("_model", "categoryLabel") or value is None:
                        continue
                    feature_label = labels.get(feature, feature)
                    feature_label = normalize_label(feature_label)
                    rows.append({
                        "brand": "Maruti Suzuki",
                        "car": model_desc,
                        "variant": vname,
                        "category": category_name,   # ✅ proper grouping
                        "sub_category": sub_category,
                        "feature": feature_label,
                        "value": str(value)
                    })
    return pd.DataFrame(rows)

# ---------- CONFIG ----------
TATA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MODELS = [
    ("tiago", "ice"),
    ("altroz", "ice"),
    ("tigor", "ice"),
    ("punch", "ice"),
    ("nexon", "ice"),
    ("harrier", "ice"),
    ("safari", "ice"),
    ("tiago", "ev"),
    ("nexon", "ev"),
    ("curvv", "ev")
]

# ---------- HELPERS ----------
def tata_features_url(model: str, fuel: str) -> str:
    if fuel == "ice":
        return f"https://cars.tatamotors.com/{model}/ice/specifications.html"
    else:  # EV
        return f"https://ev.tatamotors.com/{model}/ev/specifications.html"

def tata_normalize_label(label: str) -> str:
    return re.sub(r"\s+", "_", label.strip().lower())

# ---------- MAIN FUNCTION ----------
def tata_fetch_flat_features(model: str, fuel: str):
    url = tata_features_url(model, fuel)
    r = requests.get(url, headers=TATA_HEADERS, timeout=40)
    if r.status_code != 200:
        print(f"⚠️ Failed for {model} ({r.status_code})")
        return pd.DataFrame(columns=["brand","car","variant","category","sub_category","feature","value"])

    soup = BeautifulSoup(r.text, "html.parser")
    div = soup.find("div", class_="productspecs-results")
    if not div:
        print(f"⚠️ No JSON container for {model}")
        return pd.DataFrame(columns=["brand","car","variant","category","sub_category","feature","value"])

    data_json = div.get("data-productspecjson")
    if not data_json:
        print(f"⚠️ Empty JSON for {model}")
        return pd.DataFrame(columns=["brand","car","variant","category","sub_category","feature","value"])

    try:
        decoded_json = html.unescape(data_json)
        data = json.loads(decoded_json)
    except json.JSONDecodeError:
        print(f"⚠️ JSON decode error for {model}")
        return pd.DataFrame(columns=["brand","car","variant","category","sub_category","feature","value"])

    variants = data["results"].get("variantSpecFeature", [])
    flat_rows = []

    for variant in variants:
        variant_name = variant.get("variantLabel", "").strip()
        if not variant_name:
            continue

        # Specifications
        for spec in variant.get("productSpecifications", []):
            sub_category = tata_normalize_label(spec.get("specGroupLabel", "Specifications"))
            for item in spec.get("specList", []):
                feature = tata_normalize_label(item.get("specLabel", ""))
                value = item.get("specValue") or item.get("specVal") or item.get("value") or "N/A"
                if feature:
                    flat_rows.append({
                        "brand": "Tata",
                        "car": model,
                        "variant": variant_name,
                        "category": "Specifications",
                        "sub_category": sub_category,
                        "feature": feature,
                        "value": str(value).strip()
                    })

        # Features
        for feat in variant.get("productFeatures", []):
            sub_category = tata_normalize_label(feat.get("featureGroupLabel", "Features"))
            for item in feat.get("featureList", []):
                feature = tata_normalize_label(item.get("featureLabel", ""))
                value = item.get("featureValue") or item.get("value") or "N/A"
                if feature:
                    flat_rows.append({
                        "brand": "Tata",
                        "car": model,
                        "variant": variant_name,
                        "category": "Features",
                        "sub_category": sub_category,
                        "feature": feature,
                        "value": str(value).strip()
                    })

    df = pd.DataFrame(flat_rows)
    df = df[df["sub_category"].str.lower() != "features"]  # drop unclassified
    return df

# ========== SIDEBAR CONTROLS ==========
left, right = st.sidebar.columns([1,1])
with left:
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared. Fresh data will be fetched.")

with st.spinner("Fetching car lists..."):
    # Hyundai
    hy_models_raw = hy_get_models()
    hy_models = [m for m in hy_models_raw if m.get("enabled")]
    hy_options = {f"Hyundai — {m['description']}": m for m in hy_models}

    # Maruti Suzuki (Nexa + Arena)
    nexa_cars = ms_fetch_cars("Nexa")
    arena_cars = ms_fetch_cars("Arena")
    ms_cars = []
    for c in nexa_cars:
        c["channel"] = "Nexa"
        ms_cars.append(c)
    for c in arena_cars:
        c["channel"] = "Arena"
        ms_cars.append(c)
    ms_options = {f"Maruti — {c['modelDesc']}": c for c in ms_cars}

    # Tata
    tata_models = MODELS  # Defined earlier in your code
    tata_cars = []
    for model, fuel in tata_models:
        tata_cars.append({
            "model": model,
            "fuel": fuel
        })
    tata_options = {f"Tata — {c['model'].capitalize()}": c for c in tata_cars}


all_options = list(hy_options.keys()) + list(ms_options.keys()) + list(tata_options.keys())

# ---------- BRAND FILTER ----------
BRANDS = ["Hyundai", "Maruti Suzuki", "Tata"]
selected_brands = st.sidebar.multiselect("Select Brand(s)", options=BRANDS, default=[])

# ---------- BUILD OPTIONS BASED ON SELECTED BRANDS ----------
all_options = []
if "Hyundai" in selected_brands:
    all_options += list(hy_options.keys())
if "Maruti Suzuki" in selected_brands:
    all_options += list(ms_options.keys())
if "Tata" in selected_brands:
    all_options += list(tata_options.keys())

selected_names = st.sidebar.multiselect("Select Car(s)", options=all_options)

if not selected_names:
    st.info("👈 Select at least one car from the sidebar to begin.")
    st.stop()

# ========== BUILD UNIFIED ROWS ==========
all_rows = []
with st.spinner("Fetching detailed features/specifications..."):
    for name in selected_names:
        if name.startswith("Hyundai — "):
            m = hy_options[name]
            model_desc = m["description"]
            hy_df = hy_fetch_flat_features(model_desc)
            if not hy_df.empty:
                all_rows.append(hy_df)

        elif name.startswith("Maruti — "):
            c = ms_options[name]
            data = ms_fetch_variant_details(c["channel"], c["modelCd"])
            ms_df = ms_parse_features(data, c["channel"], c["modelDesc"])
            if not ms_df.empty:
                all_rows.append(ms_df)

        elif name.startswith("Tata — "):
            c = tata_options[name]
            model = c["model"]
            fuel = c["fuel"]
            tata_df = tata_fetch_flat_features(model, fuel)
            if not tata_df.empty:
                all_rows.append(tata_df)

if not all_rows:
    st.warning("No feature/spec data found for the selected cars.")
    st.stop()

combined_df = pd.concat(all_rows, ignore_index=True)

for col in ["sub_category", "feature"]:
    combined_df[col] = combined_df[col].apply(lambda x: normalize_label(str(x)) if pd.notnull(x) else x)

combined_df["variant_label"] = combined_df["car"] + " | " + combined_df["variant"].fillna("-")

first_sel = selected_names[0]
with st.container():
    st.markdown("### ℹ️ Selected Cars")
    st.write(", ".join(selected_names))




# ========== VARIANT FILTER ADDED HERE ==========
all_variants = combined_df["variant_label"].unique()
selected_variants = st.sidebar.multiselect(
    "Select Variant(s) (optional)",
    options=sorted(all_variants),
    default=[]
)

if selected_variants:
    combined_df = combined_df[combined_df["variant_label"].isin(selected_variants)]

# ========== FILTERS ==========
search_term = st.text_input("🔎 Search features (matches feature text)", key="search_all")

# Always define df_view
if search_term:
    df_view = combined_df[combined_df["feature"].str.contains(search_term, case=False, na=False)]
else:
    df_view = combined_df.copy()


show_diff_only = st.checkbox("🔀 Show only differences across selected variants", value=False)
if show_diff_only:
    if len(selected_variants) <= 1:
        st.warning("Select at least two variants to compare differences.")
    else:
        mask = df_view.groupby(["category", "sub_category", "feature"])["value"].transform("nunique") > 1
        df_view = df_view[mask]



# ========== DOWNLOADS ==========
export_cols = ["brand","car","variant","category","sub_category","feature","value"]

export_df = df_view[export_cols].copy().sort_values(["sub_category","feature","brand","car","variant"])

col_title, col_dl = st.columns([4,2])
with col_title:
    st.subheader("📋 Variant Comparison")
with col_dl:
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=f"unified_comparison_{date_str}.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c2:
        out_xlsx = io.BytesIO()
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="Comparison")
        st.download_button(
            "⬇️ Excel",
            data=out_xlsx.getvalue(),
            file_name=f"unified_comparison_{date_str}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ========== DISPLAY ==========
if df_view.empty:
    st.warning("No rows after filters.")
else:
    main_cats = ["Features", "Technical Specifications"]
    feat_df = df_view[df_view["category"].str.contains("Features", case=False, na=False)]
    tech_df = df_view[~df_view["category"].str.contains("Features", case=False, na=False)]


    for df_block, title in [(tech_df, "⚙️ Technical Specifications"), (feat_df, "🎛️ Features")]:
        if not df_block.empty:
            st.markdown(f"## {title}")

            subcats = list(df_block["sub_category"].unique())

            # Special rule: Fuel always below Dimensions in Technical Specs
            if title.startswith("⚙️ Technical"):
                if "Dimensions" in subcats and "Fuel" in subcats:
                    subcats.remove("Fuel")
                    dim_index = subcats.index("Dimensions")
                    subcats.insert(dim_index + 1, "Fuel")

            # Create a tab per sub-category
            tabs = st.tabs(subcats)
            for i, subcat in enumerate(subcats):
                with tabs[i]:
                    st.markdown(f"### {subcat}")
                    sub_block = df_block[df_block["sub_category"] == subcat].copy()
                    sub_pivot = sub_block.pivot_table(
                        index="feature",
                        columns="variant_label",
                        values="value",
                        aggfunc="first"  # or "max" / "min"
                    ).reset_index()

                    # Apply Yes/No formatter
                    for col in sub_pivot.columns[1:]:
                        sub_pivot[col] = sub_pivot[col].apply(
                            lambda x: yes_no_formatter(str(x)) if pd.notnull(x) else "Not Mentioned"
                        )

                    st.data_editor(
                        wrap_text(sub_pivot, first_col_width=250, other_col_width=150),
                        use_container_width=True,
                        disabled=True,
                        key=f"comparison_{title}_{subcat}"
                    )


st.title("AI Car Query Assistant")
st.write("Ask anything about cars specs in our dataset.")
user_query = st.text_input("Enter your query:")

if user_query:

    prompt = f"""
    You are a helpful assistant.
    you will answer all queries using below car specifications dataset and if required you can use internet data to answer also if answer is not in the dataset.
    You should be able to unify the features name across different brands
    Here is the car dataset:
    {combined_df}

    Answer this question: {user_query}
    """

    # Call Groq model
    with st.spinner("Generating response..."):
        response = groq_client.chat.completions.create(
            model="moonshotai/Kimi-K2-Instruct-0905",
            messages=[{"role": "user", "content": prompt}]
        )

    # Display answer
    answer = response.choices[0].message.content
    st.subheader("Answer:")
    st.write(answer)
