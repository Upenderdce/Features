import requests
import pandas as pd
import streamlit as st
import io
from datetime import datetime
import re
from rapidfuzz import process, fuzz   # for similarity search

date_str = datetime.now().strftime("%Y%m%d")

# --- Normalization rules for categories & subcategories ---
NORMALIZATION_MAP = {
    "Fuel-Efficiency (km/l)*": "Fuel Efficiency (km/l)*",
    "wheels and tyre": "Tyre",
    "wheel and tyre": "Tyre",
    "tyre": "Tyre",
    "safety security": "Safety",
    "safety and security": "Safety",
    "safety": "Safety",
}

# Prepare lowercase lookup keys
NORMALIZED_KEYS = {k.lower(): v for k, v in NORMALIZATION_MAP.items()}
NORMALIZED_VALUES = list(set(NORMALIZATION_MAP.values()))  # unique target values


def normalize_label(label: str, threshold: int = 85) -> str:
    """Normalize labels using direct map + similarity search"""
    if not label:
        return label

    norm = label.lower().strip()
    norm = re.sub(r"[\-&_/]", " ", norm)      # replace special chars with space
    norm = re.sub(r"\s+", " ", norm).strip()  # collapse multiple spaces

    # Direct match first
    if norm in NORMALIZED_KEYS:
        return NORMALIZED_KEYS[norm]

    # Similarity search against known normalized values
    best_match, score, _ = process.extractOne(
        norm,
        NORMALIZED_VALUES,
        scorer=fuzz.token_sort_ratio
    )
    if score >= threshold:
        return best_match

    # Fallback: original cleaned label
    return label.strip()


# --- Page Config ---
st.set_page_config(page_title="Maruti Suzuki Cars Feature Explorer", layout="wide")

# --- API URLs ---
API_CONFIG = {
    "Nexa": {
        "cars_url": "https://www.nexaexperience.com/graphql/execute.json/msil-platform/NexaCarList",
        "variant_url": "https://www.nexaexperience.com/graphql/execute.json/msil-platform/VariantDetailCompare;modelCd={};channel=EXC;locale=en;"
    },
    "Arena": {
        "cars_url": "https://www.marutisuzuki.com/graphql/execute.json/msil-platform/ArenaCarList",
        "variant_url": "https://www.marutisuzuki.com/graphql/execute.json/msil-platform/ArenaVariantDetailCompare;modelCd={};channel=NRM;locale=en;"
    }
}

HEADERS = {
    "accept": "application/json, */*",
    "user-agent": "Mozilla/5.0",
}


# --- Fetch functions ---
@st.cache_data(ttl=3600)
def fetch_cars(channel: str):
    url = API_CONFIG[channel]["cars_url"]
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["data"]["carModelList"]["items"]


@st.cache_data(ttl=3600)
def fetch_variant_details(channel: str, modelCd: str):
    url = API_CONFIG[channel]["variant_url"].format(modelCd)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data
def parse_features(data, channel: str):
    rows = []

    # Try to extract feature label mapping if present
    labels = {}
    try:
        labels_list = data["data"]["variantSpecificationsLabelsList"]["items"][0]
        labels = {key.replace("_label", ""): val for key, val in labels_list.items() if key.endswith("_label")}
    except Exception as e:
        print(f"⚠️ Could not parse labels for {channel}: {e}")

    variants = data["data"]["carModelList"]["items"][0]["variants"]

    for v in variants:
        vname = v.get("variantName")
        for cat in v.get("specificationCategory", []):
            category_name = normalize_label(cat.get("categoryName", ""))
            for aspect in cat.get("specificationAspect", []):
                sub_category = normalize_label(aspect.get("categoryLabel", ""))
                for feature, value in aspect.items():
                    if feature in ("_model", "categoryLabel") or value is None:
                        continue
                    # Apply label mapping (works for both Nexa & Arena now)
                    feature_label = labels.get(feature, feature)
                    feature_label = normalize_label(feature_label)
                    rows.append({
                        "variant": vname,
                        "category": category_name,
                        "sub_category": sub_category,
                        "feature": feature_label,
                        "value": str(value)
                    })

    return pd.DataFrame(rows)


# --- Helper functions ---
def yes_no_formatter(val):
    if pd.isna(val):
        return "-"
    if isinstance(val, str):
        val_low = val.lower()
        if val_low in ["yes", "true"]:
            return "✅ Yes"
        elif val_low in ["no", "false"]:
            return "❌ No"
    return str(val)


def wrap_text(df, first_col_width=250, other_col_width=150):
    styles = [
        {'selector': 'th', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]},
        {'selector': 'td', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]}
    ]
    table_styles = []
    for i, col in enumerate(df.columns):
        width = first_col_width if i == 0 else other_col_width
        table_styles.append({'selector': f'th.col{i}', 'props': [('min-width', f'{width}px')]})
        table_styles.append({'selector': f'td.col{i}', 'props': [('min-width', f'{width}px')]})
    return df.style.set_properties(**{'white-space': 'pre-wrap', 'word-wrap': 'break-word'}) \
        .set_table_styles(styles + table_styles)


# --- Streamlit UI ---
st.title("🚗 Maruti Suzuki Cars Feature Explorer & Variant Comparison")

# Brand selector (single brand)
brand = st.sidebar.radio("Select Brand", ["Maruti Suzuki"])

if st.button("🔄 Scrape Latest Data from Website"):
    st.cache_data.clear()
    st.success("Cache cleared. New data will be fetched automatically on next selection.")

# Fetch cars from both channels
with st.spinner("Fetching Maruti Suzuki car list..."):
    nexa_cars = fetch_cars("Nexa")
    arena_cars = fetch_cars("Arena")

# Merge & tag
cars = []
for c in nexa_cars:
    c["channel"] = "Nexa"
    cars.append(c)
for c in arena_cars:
    c["channel"] = "Arena"
    cars.append(c)

# Unique options with channel label
car_options = {f"{c['modelDesc']} ({c['channel']})": c for c in cars}

# --- Multi-car selection ---
selected_car_names = st.sidebar.multiselect(
    "Select Car(s) for Comparison",
    options=list(car_options.keys())
)

if not selected_car_names:
    st.warning("Please select at least one car")
else:
    all_dfs = []
    for name in selected_car_names:
        car = car_options[name]
        if name == selected_car_names[0]:
            col_image, col_info = st.columns([1, 2])
            with col_image:
                if car.get("carImage", {}).get("_publishUrl"):
                    st.image(car["carImage"]["_publishUrl"], width=300)
            with col_info:
                st.markdown(f"### {car['modelDesc']} ({car.get('bodyType', '-')}) [{car['channel']}]")
                st.markdown(f"**Price:** ₹{car.get('exShowroomPrice', 0):,}")
                if car.get("brochure", {}).get("_publishUrl"):
                    st.markdown(f"[📘 Download Brochure]({car['brochure']['_publishUrl']})")
                col1, col2, col3 = st.columns(3)
                col1.metric("Price (Ex-showroom)", f"₹{car.get('exShowroomPrice', 0):,}")
                col2.metric("Body Type", car.get('bodyType', '-'))
                col3.metric("Variants", len(car.get("variants", [])))

        with st.spinner(f"Fetching {name} specs..."):
            data = fetch_variant_details(car["channel"], car["modelCd"])
            df = parse_features(data, car["channel"])
            df["car"] = name
            df["variant_label"] = df["car"] + " | " + df["variant"]
            all_dfs.append(df)

    # === Combine ===
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # --- Variant selection ---
    variants = sorted(combined_df["variant_label"].unique())
    selected_variants = st.sidebar.multiselect(
        "Select Variant(s) to Compare",
        options=variants,
        default=variants
    )

    if not selected_variants:
        st.warning("Select at least one variant")
    else:
        # --- Variant Comparison (only workflow now) ---

        comp_df = combined_df[combined_df["variant_label"].isin(selected_variants)].copy()
        search_term_comp = st.text_input("🔎 Search features", key="comp_search")
        if search_term_comp:
            comp_df = comp_df[comp_df["feature"].str.contains(search_term_comp, case=False, na=False)]

        show_diff_only = st.checkbox("🔀 Show only differences", value=False)
        if show_diff_only and len(selected_variants) > 1:
            mask = comp_df.groupby(["category", "sub_category", "feature"])["value"].transform("nunique") > 1
            comp_df = comp_df[mask]

        tech_df = comp_df[comp_df["category"].str.contains("Technical", case=False, na=False)]
        feat_df = comp_df[~comp_df["category"].str.contains("Technical", case=False, na=False)]

        # --- Downloads ---
        comparison_download = pd.DataFrame()
        for df_block in [tech_df, feat_df]:
            if not df_block.empty:
                for subcat in df_block["sub_category"].unique():
                    sub_block = df_block[df_block["sub_category"] == subcat].copy()
                    sub_pivot = sub_block.pivot(
                        index=["category", "sub_category", "feature"],
                        columns="variant_label",
                        values="value"
                    ).reset_index()
                    comparison_download = pd.concat([comparison_download, sub_pivot], ignore_index=True)

        col_title, col_buttons = st.columns([4, 2])
        with col_title:
            st.subheader("Variant Comparison")
        with col_buttons:
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ CSV",
                    data=comparison_download.to_csv(index=False).encode("utf-8"),
                    file_name=f"comparison_{date_str}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with c2:
                output2 = io.BytesIO()
                with pd.ExcelWriter(output2, engine="openpyxl") as writer:
                    comparison_download.to_excel(writer, index=False, sheet_name="Comparison")
                st.download_button(
                    "⬇️ Excel",
                    data=output2.getvalue(),
                    file_name=f"comparison_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        # --- Display tables with tabs ---
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
                        sub_pivot = sub_block.pivot(
                            index="feature",
                            columns="variant_label",
                            values="value"
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
