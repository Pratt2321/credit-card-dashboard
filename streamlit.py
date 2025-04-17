import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import base64

# Set page configuration to wide mode
st.set_page_config(layout="wide", page_title="Visualizing the Credit Card Landscape: From Market Offerings to Beginner-Friendly Recommendations")

# Title and Author Section
st.title("Unlocking Beginner‑Friendly Credit Cards")
st.markdown("**Author**: Pratham Pradhan | **Affiliation**: CMSE at MSU")
st.markdown(
    """
    <h2 style="font-size:32px; margin-bottom:0;">Introduction</h2>
    <p style="font-size:20px;">
    Credit cards are ubiquitous tools in personal finance, yet their varied fee structures, eligibility requirements, and reward programs can overwhelm new cardholders. This dashboard leverages the Consumer Financial Protection Bureau's (CFPB) 2024 Terms of Credit Card Plans survey to map out the landscape of credit card offerings across major issuers. We begin with a treemap comparing annual fees from the top‑25 institutions versus others, proceed to two side‑by‑side analyses of a stacked bar chart showing how many top issuers waive fees versus charge them and a histogram illustrating the typical grace‑period length, and conclude with a Sankey‑style funnel that isolates truly beginner‑friendly cards (no required credit score, no annual fee, no foreign‑transaction fee, yet still offering rewards). By combining broad market overviews with a targeted “funnel” analysis, this work helps both consumers and analysts quickly identify products that balance accessibility with benefits.
        </p>
""",
    unsafe_allow_html=True
)

# --------------------------- Data Loading --------------------------- #
@st.cache_data
def load_data():
    try:
        # Attempt to load the dataset from an Excel file
        df = pd.read_excel('cfpb_tccp-data_2024-06-30.xlsx', header=0, skiprows=9)
        # Ensure the key column is present or create it from similar columns
        if "Issued by Top 25 Institution" not in df.columns:
            for col in df.columns:
                if "top 25" in col.lower():
                    df["Issued by Top 25 Institution"] = df[col].map({
                        True: "Top 25 Institutions",
                        False: "Not Top 25 Institutions"
                    })
                    break
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create a sample dataframe if loading fails
        data = {
            "Institution Name": ["Bank A", "Bank B", "Bank C"] * 10,
            "Product Name": ["Card X", "Card Y", "Card Z"] * 10,
            "Annual Fee": [0, 95, 195] * 10,
            "Issued by Top 25 Institution": [True, True, False] * 10,
            "Foreign Transaction Fees?": ["no", "yes", "no"] * 10,
            "Targeted Credit Tiers": ["no credit score", "good", "excellent"] * 10,
            "Rewards": ["Cash back", "Points", "Miles"] * 10,
            "Other Rewards": ["Sign-up bonus", "", "Travel insurance"] * 10,
            "Grace Period": [21, 25, 30] * 10
        }
        return pd.DataFrame(data)

df = load_data()

# Ensure numeric columns are properly formatted
df["Annual Fee"] = pd.to_numeric(df["Annual Fee"], errors="coerce").fillna(0)
df["Grace Period"] = pd.to_numeric(df["Grace Period"], errors="coerce")

# Standardize the institution flag column for plotting (if stored as boolean)
if df["Issued by Top 25 Institution"].dtype == bool:
    df["Issued by Top 25 Institution"] = df["Issued by Top 25 Institution"].map({
        True: "Top 25 Institutions",
        False: "Not Top 25 Institutions"
    })

# ------------------------- Visualization Functions ------------------------- #
@st.cache_data
def create_heatmap(df_input):
    """
    Creates a treemap (styled as a heatmap) that visualizes credit card offerings,
    broken down by institution (Top 25 vs. Not Top 25) and their respective annual fees.
    """
    # Use the existing column mapping if not already a string
    temp_df = df_input.copy()
    if temp_df["Issued by Top 25 Institution"].dtype == bool:
        temp_df["Issued by Top 25 Institution"] = temp_df["Issued by Top 25 Institution"].map({
            True: "Top 25 Institutions",
            False: "Not Top 25 Institutions"
        })
        
    fig = px.treemap(
        data_frame=temp_df,
        path=["Issued by Top 25 Institution", "Institution Name", "Product Name"],
        values="Annual Fee",
        color="Issued by Top 25 Institution",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=["Annual Fee"]
    )
    fig.update_traces(
        hovertemplate="**%{label}**<br>Annual Fee: $%{value}<extra></extra>"
    )
    fig.update_layout(
        title="Treemap of Credit Card Offerings Based on Annual Fees (Top 25 vs. Others)",
        margin=dict(t=50, l=25, r=25, b=25)
    )
    return fig

@st.cache_data
def create_stacked_barplot(df_input):
    """
    Creates a stacked bar chart for the top 10 institutions that displays the proportion of
    credit cards with and without an annual fee.
    """
    temp_df = df_input.copy()
    temp_df["Annual Fee Label"] = temp_df["Annual Fee"].apply(lambda x: "No Annual Fee" if x == 0 else "Has Annual Fee")
    
    fee_counts = temp_df.groupby(["Institution Name", "Annual Fee Label"]).size().unstack(fill_value=0)
    # Select top 10 institutions by card count
    top10_institutions = fee_counts.sum(axis=1).sort_values(ascending=False).head(10)
    fee_counts_top10 = fee_counts.loc[top10_institutions.index]
    
    # Calculate proportions
    fee_props = fee_counts_top10.div(fee_counts_top10.sum(axis=1), axis=0)
    fee_props_sorted = fee_props.sort_values(by="No Annual Fee", ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.get_cmap("Set2").colors[:2]
    fee_props_sorted[["No Annual Fee", "Has Annual Fee"]].plot(
        kind="bar", stacked=True, color=colors, ax=ax
    )
    plt.title("Proportion of Cards with No Annual Fee vs. Annual Fee (Top 10 Institutions)", pad=45)
    plt.xlabel("Institution Name")
    plt.ylabel("Proportion of Cards")
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Annual Fee Status", loc='upper center', bbox_to_anchor=(0.5, 1.23), ncol=2)
    # Remove unnecessary spines for a clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

@st.cache_data
def create_histogram(df_input):
    """
    Creates a histogram of the credit card grace periods along with a kernel density estimate.
    A vertical red dashed line indicates the median grace period.
    """
    temp_df = df_input.copy()
    temp_df["Grace Period"] = pd.to_numeric(temp_df["Grace Period"], errors="coerce")
    grace_df = temp_df.dropna(subset=["Grace Period"])
    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(grace_df["Grace Period"], bins=20, kde=True, color="#CCCCCC", ax=ax)
    median_val = grace_df["Grace Period"].median()
    plt.axvline(median_val, color="red", linestyle="--", label=f"Median: {median_val:.0f} days")
    plt.title("Distribution of Credit Card Grace Periods")
    plt.xlabel("Grace Period (Days)")
    plt.ylabel("Number of Cards")
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig

@st.cache_data
def create_sankey(df_input):
    """
    Creates a Sankey diagram that highlights a beginner-friendly credit card path:
    from Top 25 Institutions, filtering by no required credit score,
    then by no annual fee, followed by no foreign transaction fees and lastly rewarding cards.
    """
    temp_df = df_input.copy()
    # Ensure proper numeric conversion and clean strings for fees
    temp_df["Annual Fee"] = pd.to_numeric(temp_df["Annual Fee"], errors="coerce").fillna(0)
    temp_df["Foreign Transaction Fees?"] = temp_df["Foreign Transaction Fees?"].astype(str).str.strip().str.lower()
    
    # Filter for Top 25 Institutions; if none are flagged, use top 5 institutions as fallback
    top25_df = temp_df[temp_df["Issued by Top 25 Institution"] == "Top 25 Institutions"].copy()
    if top25_df.empty:
        top_institutions = temp_df["Institution Name"].value_counts().nlargest(5).index
        top25_df = temp_df[temp_df["Institution Name"].isin(top_institutions)].copy()
        top25_df["Issued by Top 25 Institution"] = "Top 25 Institutions"
        
    # Create indicator columns
    top25_df["No Score"] = top25_df["Targeted Credit Tiers"].astype(str).str.contains("no credit score", case=False, na=False)
    top25_df["Annual Fee Label"] = top25_df["Annual Fee"].apply(lambda x: "No Annual Fee" if x == 0 else "Has Annual Fee")
    top25_df["Foreign Fee Label"] = top25_df["Foreign Transaction Fees?"].apply(lambda x: "No FTF" if x == "no" else "Has FTF")
    
    # Define nodes for the Sankey diagram
    labels = [
        "Top 25 Institutions",
        "No Credit Score Required",
        "Some Credit Score Required",
        "No Annual Fee",
        "Has Annual Fee",
        "No FTF",
        "Has FTF",
        "Rewards",
        "Minimal to no Rewards"
    ]
    label_map = {label: i for i, label in enumerate(labels)}
    
    # Set node positions for layout purposes
    x_coords = [0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8]
    y_coords = [0.5, 0.75, 0.25, 0.7, 0.3, 0.7, 0.3, 0.6, 0.4]
    
    # Level 1: Top 25 Institutions to Credit Score Requirement
    no_score_count = int(top25_df["No Score"].sum())
    has_score_count = int(len(top25_df) - no_score_count)
    s1_source = [label_map["Top 25 Institutions"]] * 2
    s1_target = [label_map["No Credit Score Required"], label_map["Some Credit Score Required"]]
    s1_value = [no_score_count, has_score_count]
    
    # Level 2: For those with no credit score requirement, split by Annual Fee
    no_score_df = top25_df[top25_df["No Score"]].copy()
    annual_counts = no_score_df["Annual Fee Label"].value_counts()
    s2_source = [label_map["No Credit Score Required"]] * 2
    s2_target = [label_map["No Annual Fee"], label_map["Has Annual Fee"]]
    s2_value = [
        int(annual_counts.get("No Annual Fee", 0)),
        int(annual_counts.get("Has Annual Fee", 0))
    ]
    
    # Level 3: For no annual fee cards, split by Foreign Transaction Fee
    no_annual_df = no_score_df[no_score_df["Annual Fee"] == 0].copy()
    ftf_counts = no_annual_df["Foreign Fee Label"].value_counts()
    s3_source = [label_map["No Annual Fee"]] * 2
    s3_target = [label_map["No FTF"], label_map["Has FTF"]]
    s3_value = [
        int(ftf_counts.get("No FTF", 0)),
        int(ftf_counts.get("Has FTF", 0))
    ]
    
    # Level 4: For cards with no FTF, determine if rewards exist.
    # IMPORTANT: Use the same label ("No FTF") as in the previous step.
    no_ftf_df = no_annual_df[no_annual_df["Foreign Fee Label"] == "No FTF"].copy()
    # Clean the rewards columns
    no_ftf_df["Rewards"] = no_ftf_df["Rewards"].astype(str).str.strip()
    no_ftf_df["Other Rewards"] = no_ftf_df["Other Rewards"].astype(str).str.strip()
    # Instead of using .apply (which led to the error), use np.where to create a vectorized condition.
    condition = (~no_ftf_df["Rewards"].str.lower().isin(["", "nan"])) | (~no_ftf_df["Other Rewards"].str.lower().isin(["", "nan"]))
    no_ftf_df["Reward Combo"] = np.where(condition, "Rewards", "Minimal to no Rewards")
    
    reward_counts = no_ftf_df["Reward Combo"].value_counts()
    s4_source = [label_map["No FTF"]] * 2
    s4_target = [label_map["Rewards"], label_map["Minimal to no Rewards"]]
    s4_value = [
        int(reward_counts.get("Rewards", 0)),
        int(reward_counts.get("Minimal to no Rewards", 0))
    ]
    
    # Combine all the levels
    sources = s1_source + s2_source + s3_source + s4_source
    targets = s1_target + s2_target + s3_target + s4_target
    values  = s1_value  + s2_value  + s3_value  + s4_value
    
    # Define node colors (highlighting the main funnel)
    highlight_colors = {
        "Top 25 Institutions": "#440154",
        "No Credit Score Required": "#46327e",
        "No Annual Fee": "#31688e",
        "No FTF": "#35b779",
        "Rewards": "#fde725"
    }
    node_colors = [highlight_colors[label] if label in highlight_colors else "lightgrey" for label in labels]
    
    # Define link colors emphasizing main funnel transitions
    highlight_link_pairs = [
        ("Top 25 Institutions", "No Credit Score Required"),
        ("No Credit Score Required", "No Annual Fee"),
        ("No Annual Fee", "No FTF"),
        ("No FTF", "Rewards")
    ]
    highlight_link_colors = [
        "#440154",
        "#46327e",
        "#31688e",
        "#35b779"
    ]
    link_colors = []
    for s, t in zip(sources, targets):
        src_label = labels[s]
        tgt_label = labels[t]
        if (src_label, tgt_label) in highlight_link_pairs:
            idx = highlight_link_pairs.index((src_label, tgt_label))
            link_colors.append(highlight_link_colors[idx])
        else:
            link_colors.append("rgba(200,200,200,0.2)")
    
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
            x=x_coords,
            y=y_coords,
            color=node_colors,
            hovertemplate='%{label}<extra></extra>'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            hovertemplate='%{value:.0f} cards<extra></extra>'
        )
    ))
    fig.update_layout(
        title_text="Visualizing Beginner-friendly Cards: From No Credit Score to Rewards",
        font_size=10,
        margin=dict(t=125, l=25, r=75, b=25),
        annotations=[
            dict(
                x=.8,
                y=0.15,
                xref='paper',
                yref='paper',
                text="*FTF: Foreign Transaction Fee",
                showarrow=False,
                font=dict(size=12, color="white"),
                align="left"
            )
        ]
    )
    return fig

@st.cache_data
def convert_df_to_csv(dataframe):
    return dataframe.to_csv(index=False).encode('utf-8')

# --------------------------- Main Dashboard Layout --------------------------- #
# 1. Top Section: Heatmap (Treemap)
st.markdown("### Treemap: Credit Card Offerings by Annual Fee")
heatmap_fig = create_heatmap(df)
st.plotly_chart(heatmap_fig, use_container_width=True)

# 2. Middle Section: Two Columns (Stacked Bar & Histogram)
st.markdown("### Institution Annual Fee Distribution & Grace Period Histogram")
col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Stacked Bar Chart: Annual Fee Proportions (Top 10 Institutions)")
    barplot_fig = create_stacked_barplot(df)
    st.pyplot(barplot_fig)
with col2:
    st.markdown("#### Histogram: Credit Card Grace Periods")
    hist_fig = create_histogram(df)
    st.pyplot(hist_fig)

# 3. Bottom Section: Sankey Diagram
st.markdown("### Sankey Diagram: Beginner-friendly Credit Card Path")
sankey_fig = create_sankey(df)
st.plotly_chart(sankey_fig, use_container_width=True)
st.markdown(
    '<p style="font-size:20px; color:white; margin-top:-0.5rem;">'
    'There were 18 cards in the last filtered list. Some of the notable names include the Capital One SavorOne for Students, the Discover It Secured Credit Card, and the Chase Freedom Rise. '
    '</p>',
    unsafe_allow_html=True
)

# --------------------- Interactive Filters --------------------- #
with st.expander("Interactive Filters"):
    st.markdown("Filter the dataset to update the visualizations below.")
    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_institutions = st.multiselect(
            "Select Institution(s)",
            options=sorted(df["Institution Name"].unique()),
            default=[]
        )
    with filter_col2:
        min_fee = int(df["Annual Fee"].min())
        max_fee = int(df["Annual Fee"].max())
        fee_range = st.slider(
            "Select Annual Fee Range",
            min_value=min_fee,
            max_value=max_fee,
            value=(min_fee, max_fee)
        )
    
    # Apply filters to the dataframe
    filtered_df = df.copy()
    if selected_institutions:
        filtered_df = filtered_df[filtered_df["Institution Name"].isin(selected_institutions)]
    filtered_df = filtered_df[(filtered_df["Annual Fee"] >= fee_range[0]) & 
                              (filtered_df["Annual Fee"] <= fee_range[1])]
    
    st.write(f"Showing {len(filtered_df)} of {len(df)} credit cards")
    
    if len(filtered_df) > 0 and len(filtered_df) < len(df):
        st.markdown("#### Filtered Visualizations")
        # Re-create visualizations based on the filtered data
        filtered_heatmap = create_heatmap(filtered_df)
        filtered_barplot = create_stacked_barplot(filtered_df)
        filtered_hist = create_histogram(filtered_df)
        filtered_sankey = create_sankey(filtered_df)
        
        st.plotly_chart(filtered_heatmap, use_container_width=True)
        filtered_col1, filtered_col2 = st.columns(2)
        with filtered_col1:
            st.pyplot(filtered_barplot)
        with filtered_col2:
            st.pyplot(filtered_hist)
        st.plotly_chart(filtered_sankey, use_container_width=True)
        
# --------------------------- Conclusion & References --------------------------- #
st.markdown(
    """
    ## Key Takeaways
    - Top institutions offer a wide array of credit cards with differing fee structures; this visualization shows that even among recognized issuers, many cards carry an annual fee.  
    - The stacked bar chart and histogram help clarify institutional differences and the typical grace period range (mostly clustered around 21–25 days).  
    - The Sankey diagram reveals a steep funnel from top issuers to truly beginner-friendly products that combine no required credit score, no annual fee, and no foreign transaction fees before offering rewards.
    
    ## References
    1. Terms of Credit Card Plans (TCCP) survey (2024). Consumer Financial Protection Bureau. https://www.consumerfinance.gov/data-research/credit-card-data/terms-credit-card-plans-survey/
    2. Plotly Technologies Inc. (2015). Collaborative data science. https://plot.ly
    """
)

# --------------------------- Data Download --------------------------- #
csv_data = convert_df_to_csv(df)
st.download_button(
    label="Download Raw Data",
    data=csv_data,
    file_name="credit_card_data.csv",
    mime="text/csv"
)
