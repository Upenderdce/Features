import requests
import pandas as pd
import streamlit as st
import io
from datetime import datetime
date_str = datetime.now().strftime("%Y%m%d")

# --- Page Config ---
st.set_page_config(page_title="Nexa Car Features", layout="wide")

# --- API URLs ---
CARS_URL = "https://www.nexaexperience.com/graphql/execute.json/msil-platform/NexaCarList"
VARIANT_URL = (
    "https://www.nexaexperience.com/graphql/execute.json/"
    "msil-platform/VariantDetailCompare;modelCd={};channel=EXC;locale=en;"
)

HEADERS = {
    "accept": "application/json, */*",
    "user-agent": "Mozilla/5.0",
}

# --- Fetch functions ---
@st.cache_data
def fetch_cars():
    r = requests.get(CARS_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()["data"]["carModelList"]["items"]

@st.cache_data
def fetch_variant_details(modelCd: str):
    url = VARIANT_URL.format(modelCd)
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

@st.cache_data
def parse_features(data):
    labels = {}
    try:
        labels_list = data["data"]["variantSpecificationsLabelsList"]["items"][0]
        labels = {key.replace("_label", ""): val for key, val in labels_list.items() if key.endswith("_label")}
    except Exception as e:
        print("⚠️ Could not parse labels:", e)

    variants = data["data"]["carModelList"]["items"][0]["variants"]
    rows = []
    for v in variants:
        vname = v.get("variantName")
        for cat in v.get("specificationCategory", []):
            category_name = cat.get("categoryName")
            for aspect in cat.get("specificationAspect", []):
                sub_category = aspect.get("categoryLabel")
                for feature, value in aspect.items():
                    if feature in ("_model", "categoryLabel") or value is None:
                        continue
                    feature_label = labels.get(feature, feature)
                    rows.append({
                        "variant": vname,
                        "category": category_name,
                        "sub_category": sub_category,
                        "feature": feature_label,
                        "value": str(value)
                    })
    df = pd.DataFrame(rows)
    return df

# --- Helper functions ---
def yes_no_formatter(val):
    if isinstance(val, str):
        val_low = val.lower()
        if val_low in ["yes", "true"]:
            return "✅ Yes"
        elif val_low in ["no", "false"]:
            return "❌ No"
    return val if val != "nan" else "-"


def wrap_text(df, first_col_width=250, other_col_width=150):
    # Apply text wrapping
    styles = [
        {'selector': 'th', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]},
        {'selector': 'td', 'props': [('white-space', 'pre-wrap'), ('word-wrap', 'break-word')]}
    ]

    # Set column widths
    table_styles = []
    for i, col in enumerate(df.columns):
        width = first_col_width if i == 0 else other_col_width
        table_styles.append({'selector': f'th.col{i}', 'props': [('min-width', f'{width}px')]})
        table_styles.append({'selector': f'td.col{i}', 'props': [('min-width', f'{width}px')]})

    return df.style.set_properties(**{'white-space': 'pre-wrap', 'word-wrap': 'break-word'}) \
        .set_table_styles(styles + table_styles)


# --- Streamlit UI ---

st.title("🚗 Nexa Cars Feature Explorer & Variant Comparison")

# --- Scrape Latest Data Button ---
if st.button("🔄 Scrape Latest Data from Website"):
    st.cache_data.clear()  # Clear cached fetch functions
    st.success("Cache cleared. New data will be fetched automatically on next selection.")


with st.spinner("Fetching Nexa car list..."):
    cars = fetch_cars()
car_options = {c["modelDesc"]: c for c in cars}

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
            col_image, col_info = st.columns([1, 2])  # Image left, info right

            with col_image:
                st.image(car["carImage"]["_publishUrl"], width=300)

            with col_info:
                st.markdown(f"### {car['modelDesc']} ({car['bodyType']})")
                st.markdown(f"**Price:** ₹{car['exShowroomPrice']:,}")
                if car.get("brochure", {}).get("_publishUrl"):
                    st.markdown(f"[📘 Download Brochure]({car['brochure']['_publishUrl']})")
                col1, col2, col3 = st.columns(3)
                col1.metric("Price (Ex-showroom)", f"₹{car['exShowroomPrice']:,}")
                col2.metric("Body Type", car['bodyType'])
                col3.metric("Variants", len(car["variants"]))

        with st.spinner(f"Fetching {name} specs..."):
            data = fetch_variant_details(car["modelCd"])
            df = parse_features(data)
            df["car"] = name
            df["variant_label"] = df["car"] + " | " + df["variant"]
            all_dfs.append(df)

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
        tab1, tab2 = st.tabs(["🔍 Feature Explorer", "📊 Variant Comparison"])

        # --- Feature Explorer ---
        with tab1:
            # Filter and split data
            explore_df = combined_df.copy()
            search_term = st.text_input("🔎 Search features", key="explorer_search")
            if search_term:
                explore_df = explore_df[explore_df["feature"].str.contains(search_term, case=False, na=False)]

            tech_df = explore_df[explore_df["category"].str.contains("Technical", case=False, na=False)]
            feat_df = explore_df[~explore_df["category"].str.contains("Technical", case=False, na=False)]

            # Prepare pivoted version for download
            explorer_download = pd.DataFrame()
            for df_block in [tech_df, feat_df]:
                if not df_block.empty:
                    for subcat in df_block["sub_category"].unique():
                        sub_block = df_block[df_block["sub_category"] == subcat].copy()
                        sub_pivot = sub_block.pivot(
                            index=["category", "sub_category", "feature"],
                            columns="variant_label",
                            values="value"
                        ).reset_index()
                        explorer_download = pd.concat([explorer_download, sub_pivot], ignore_index=True)

            # Top buttons
            col_title, col_buttons = st.columns([4, 1])
            with col_title:
                st.subheader("Feature Explorer (Side by Side)")
            with col_buttons:
                st.download_button(
                    "⬇️ CSV",
                    data=explorer_download.to_csv(index=False).encode("utf-8"),
                    file_name=f"feature_explorer_{date_str}.csv",
                    mime="text/csv"
                )
                output1 = io.BytesIO()
                with pd.ExcelWriter(output1, engine="openpyxl") as writer:
                    explorer_download.to_excel(writer, index=False, sheet_name="FeatureExplorer")
                st.download_button(
                    "⬇️ Excel",
                    data=output1.getvalue(),
                    file_name=f"feature_explorer_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Display tables
            for df_block, title in [(tech_df, "⚙️ Technical Specifications"), (feat_df, "🎛️ Features")]:
                if not df_block.empty:
                    st.markdown(f"## {title}")
                    for subcat in df_block["sub_category"].unique():
                        st.markdown(f"**{subcat}**")
                        sub_block = df_block[df_block["sub_category"] == subcat].copy()
                        sub_pivot = sub_block.pivot(
                            index="feature",
                            columns="variant_label",
                            values="value"
                        ).reset_index()
                        for col in sub_pivot.columns[1:]:
                            sub_pivot[col] = sub_pivot[col].apply(
                                lambda x: yes_no_formatter(str(x)) if pd.notnull(x) else "Not Mentioned"
                            )
                        st.dataframe(wrap_text(sub_pivot, first_col_width=250, other_col_width=150),
                                     use_container_width=True)

        with tab2:
            # Filter selected variants
            comp_df = combined_df[combined_df["variant_label"].isin(selected_variants)].copy()

            # Search feature in Variant Comparison
            search_term_comp = st.text_input("🔎 Search features in Variant Comparison", key="comp_search")
            if search_term_comp:
                comp_df = comp_df[comp_df["feature"].str.contains(search_term_comp, case=False, na=False)]

            # Show only differences if checked
            show_diff_only = st.checkbox("🔀 Show only differences", value=False)
            if show_diff_only and len(selected_variants) > 1:
                mask = comp_df.groupby(["category", "sub_category", "feature"])["value"].transform("nunique") > 1
                comp_df = comp_df[mask]

            # Split into Technical and Features
            tech_df = comp_df[comp_df["category"].str.contains("Technical", case=False, na=False)]
            feat_df = comp_df[~comp_df["category"].str.contains("Technical", case=False, na=False)]

            # Prepare pivoted version for download
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

            # Top buttons
            col_title, col_buttons = st.columns([4, 1])
            with col_title:
                st.subheader("Variant Comparison")
            with col_buttons:
                st.download_button(
                    "⬇️ CSV",
                    data=comparison_download.to_csv(index=False).encode("utf-8"),
                    file_name=f"comparison_{date_str}.csv",
                    mime="text/csv"
                )
                output2 = io.BytesIO()
                with pd.ExcelWriter(output2, engine="openpyxl") as writer:
                    comparison_download.to_excel(writer, index=False, sheet_name="Comparison")
                st.download_button(
                    "⬇️ Excel",
                    data=output2.getvalue(),
                    file_name=f"comparison_{date_str}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            # Display tables
            for df_block, title in [(tech_df, "⚙️ Technical Specifications"), (feat_df, "🎛️ Features")]:
                if not df_block.empty:
                    st.markdown(f"## {title}")
                    for subcat in df_block["sub_category"].unique():
                        st.markdown(f"**{subcat}**")
                        sub_block = df_block[df_block["sub_category"] == subcat].copy()
                        sub_pivot = sub_block.pivot(
                            index="feature",
                            columns="variant_label",
                            values="value"
                        ).reset_index()
                        for col in sub_pivot.columns[1:]:
                            sub_pivot[col] = sub_pivot[col].apply(
                                lambda x: yes_no_formatter(str(x)) if pd.notnull(x) else "Not Mentioned"
                            )
                        st.dataframe(wrap_text(sub_pivot, first_col_width=250, other_col_width=150), use_container_width=True)


