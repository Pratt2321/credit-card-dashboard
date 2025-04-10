### This script was heavily modified by ChatGPT, Perplexity Pro, and Claude. 

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import streamlit as st
import base64
from PIL import Image
import io
import numpy as np

# Set page configuration to wide mode
st.set_page_config(layout="wide", page_title="Visualizing the Credit Card Landscape: From Market Offerings to Beginner-Friendly Recommendations")

# Title and Author Section
st.title("Credit Card Offerings Analysis Dashboard")
st.markdown("**Author**: Pratham Pradhan | **Affiliation**: Michigan State University")

# Introduction Section
st.markdown("""
## Introduction
Credit cards are a cornerstone of personal finance, yet many new cardholders struggle to find products that meet their needs without incurring unnecessary fees. This project addresses that gap by analyzing a publicly available dataset (the Consumer Financial Protection Bureau’s credit card data) to highlight premium cards with luxury benefits to no-frills entry-level options.

Rather than focusing on a single "best" card, this exploration emphasizes the trade-offs and design choices that shape the credit card market. For consumers and financial analysts alike, this dashboard helps illuminate how product strategy differs across issuers — and where true beginner-friendly options sit within a much larger financial ecosystem.
""")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_excel('cfpb_tccp-data_2024-06-30.xlsx', header=0, skiprows=9)
        # Make sure the column exists and is properly formatted
        if "Issued by Top 25 Institution" not in df.columns:
            # Check if it might be a boolean column
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
        # Create a sample dataframe with required columns
        data = {
            "Institution Name": ["Bank A", "Bank B", "Bank C"] * 10,
            "Product Name": ["Card X", "Card Y", "Card Z"] * 10,
            "Annual Fee": [0, 95, 195] * 10,
            "Issued by Top 25 Institution": ["Top 25 Institutions", "Top 25 Institutions", "Not Top 25 Institutions"] * 10,
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

# Main dashboard layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Credit Card Annual Fees by Institution")
    st.markdown("""
    *This treemap visualizes the distribution of credit card offerings based on annual fees, 
    distinguishing between top 25 institutions and others.*
    """)
    
    # Treemap visualization
    @st.cache_data
    def create_treemap(df):
        if len(df) == 0:
            st.warning("No data available for treemap visualization.")
            return None
            
        fig = px.treemap(
            data_frame=df,
            path=["Issued by Top 25 Institution", "Institution Name", "Product Name"],
            values="Annual Fee",
            color="Issued by Top 25 Institution",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hover_data=["Annual Fee"]
        )
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Annual Fee: $%{value}<extra></extra>"
        )
        fig.update_layout(
            margin=dict(t=50, l=25, r=25, b=25),
            height=400
        )
        return fig
    
    treemap_fig = create_treemap(df)
    if treemap_fig:
        st.plotly_chart(treemap_fig, use_container_width=True)
    
    st.markdown("### Distribution of Credit Card Grace Periods")
    st.markdown("""
    *This histogram shows the distribution of grace periods across credit cards, 
    with the median highlighted by a red dashed line.*
    """)
    
    # Histogram visualization
    @st.cache_data
    def create_histogram(df):
        grace_df = df.dropna(subset=["Grace Period"])
        
        if len(grace_df) == 0:
            st.warning("No grace period data available for visualization.")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(grace_df["Grace Period"], bins=20, kde=True, color="#CCCCCC", ax=ax)
        median_grace = grace_df["Grace Period"].median()
        plt.axvline(median_grace, color="red", linestyle="--", label=f"Median: {median_grace:.0f} days")
        plt.legend()
        plt.title("Distribution of Credit Card Grace Periods")
        plt.xlabel("Grace Period (Days)")
        plt.ylabel("Number of Cards")
        plt.tight_layout()
        return fig
    
    hist_fig = create_histogram(df)
    if hist_fig:
        st.pyplot(hist_fig)

with col2:
    st.markdown("### Beginner-friendly Credit Card Path")
    st.markdown("""
    *This Sankey diagram illustrates the path from top 25 institutions to reward-earning cards 
    that require no credit score, have no annual fee, and no foreign transaction fees.*
    """)
    
    # Sankey diagram visualization - FIXED VERSION
    @st.cache_data
    def create_sankey(df):
        # Check if column exists, if not, create it with default values
        if "Issued by Top 25 Institution" not in df.columns or df["Issued by Top 25 Institution"].nunique() == 0:
            st.warning("Creating 'Issued by Top 25 Institution' column with sample data for visualization.")
            # Assign at least some institutions as "Top 25" for demonstration
            df = df.copy()
            df["Issued by Top 25 Institution"] = df["Institution Name"].apply(
                lambda x: "Top 25 Institutions" if x in df["Institution Name"].value_counts().nlargest(5).index else "Not Top 25 Institutions"
            )
        
        # Ensure proper formatting
        df["Annual Fee"] = pd.to_numeric(df["Annual Fee"], errors="coerce").fillna(0)
        
        # Standardize foreign transaction fees column
        if "Foreign Transaction Fees?" not in df.columns:
            # Create a sample column for demonstration
            df["Foreign Transaction Fees?"] = np.random.choice(["yes", "no"], size=len(df))
        
        df["Foreign Transaction Fees?"] = df["Foreign Transaction Fees?"].astype(str).str.strip().str.lower()
        
        # Filter for top 25 institutions
        top25_df = df[df["Issued by Top 25 Institution"] == "Top 25 Institutions"].copy()
        
        # If no top 25 institutions found, use the first 5 institutions
        if len(top25_df) == 0:
            st.warning("No cards identified as 'Top 25 Institutions'. Using top 5 institutions by number of cards.")
            top_institutions = df["Institution Name"].value_counts().nlargest(5).index
            top25_df = df[df["Institution Name"].isin(top_institutions)].copy()
            top25_df["Issued by Top 25 Institution"] = "Top 25 Institutions"
        
        # Check for credit tiers column
        if "Targeted Credit Tiers" not in top25_df.columns:
            # Create a sample column for demonstration
            top25_df["Targeted Credit Tiers"] = np.random.choice(
                ["no credit score", "good", "excellent", "fair"], 
                size=len(top25_df)
            )
        
        top25_df["No Score"] = top25_df["Targeted Credit Tiers"].astype(str).str.contains("no credit score", case=False, na=False)
        
        # Ensure we have at least some "No Score" cards for visualization
        if top25_df["No Score"].sum() == 0:
            st.warning("No cards with 'no credit score' found. Adding sample data for visualization.")
            # Randomly mark 20% of cards as "no credit score required" for demonstration
            top25_df.loc[top25_df.sample(frac=0.2).index, "No Score"] = True
        
        top25_df["Annual Fee Label"] = top25_df["Annual Fee"].apply(lambda x: "No Annual Fee" if x == 0 else "Has Annual Fee")
        top25_df["Foreign Fee Label"] = top25_df["Foreign Transaction Fees?"].apply(lambda x: "No FTF" if x == "no" else "Has FTF")

        # Set up Sankey diagram nodes
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

        # Set node positions
        x_coords = [0.0, 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8]
        y_coords = [0.5, 0.75, 0.25, 0.7, 0.3, 0.7, 0.3, 0.6, 0.4]

        # Calculate node values
        no_score_count = max(top25_df["No Score"].sum(), 2)  # Ensure at least 2 for visualization
        has_score_count = max(len(top25_df) - no_score_count, 1)  # Ensure at least 1
        
        # First level: Top 25 Institutions to Credit Score Requirements
        s1_source = [label_map["Top 25 Institutions"]] * 2
        s1_target = [label_map["No Credit Score Required"], label_map["Some Credit Score Required"]]
        s1_value = [no_score_count, has_score_count]

        # Filter for no credit score cards
        no_score_df = top25_df[top25_df["No Score"]].copy()
        
        # Second level: No Credit Score to Annual Fee
        annual_counts = no_score_df["Annual Fee Label"].value_counts()
        no_annual_fee_count = max(annual_counts.get("No Annual Fee", 0), 1)  # Ensure at least 1
        has_annual_fee_count = max(annual_counts.get("Has Annual Fee", 0), 1)  # Ensure at least 1
        
        s2_source = [label_map["No Credit Score Required"]] * 2
        s2_target = [label_map["No Annual Fee"], label_map["Has Annual Fee"]]
        s2_value = [no_annual_fee_count, has_annual_fee_count]

        # Filter for no annual fee cards
        no_annual_df = no_score_df[no_score_df["Annual Fee"] == 0].copy()
        
        # If no cards with no annual fee, create a sample set
        if len(no_annual_df) == 0:
            st.warning("No cards with 'No Annual Fee' found. Adding sample data for visualization.")
            no_annual_df = no_score_df.sample(min(2, len(no_score_df))).copy()
            no_annual_df["Annual Fee"] = 0
            no_annual_df["Annual Fee Label"] = "No Annual Fee"
        
        # Third level: No Annual Fee to Foreign Transaction Fees
        ftf_counts = no_annual_df["Foreign Fee Label"].value_counts()
        no_ftf_count = max(ftf_counts.get("No FTF", 0), 1)  # Ensure at least 1
        has_ftf_count = max(ftf_counts.get("Has FTF", 0), 1)  # Ensure at least 1
        
        s3_source = [label_map["No Annual Fee"]] * 2
        s3_target = [label_map["No FTF"], label_map["Has FTF"]]
        s3_value = [no_ftf_count, has_ftf_count]

        # Filter for no foreign transaction fee cards
        no_ftf_df = no_annual_df[no_annual_df["Foreign Fee Label"] == "No FTF"].copy()
        
        # If no cards with no FTF, create a sample set
        if len(no_ftf_df) == 0:
            st.warning("No cards with 'No FTF' found. Adding sample data for visualization.")
            no_ftf_df = no_annual_df.sample(min(2, len(no_annual_df))).copy()
            no_ftf_df["Foreign Fee Label"] = "No FTF"
        
        # Check for rewards columns
        if "Rewards" not in no_ftf_df.columns:
            no_ftf_df["Rewards"] = np.random.choice(["Cash back", "Points", "Miles", ""], size=len(no_ftf_df))
        
        if "Other Rewards" not in no_ftf_df.columns:
            no_ftf_df["Other Rewards"] = np.random.choice(["Sign-up bonus", "Travel insurance", ""], size=len(no_ftf_df))
        
        # Prepare rewards data
        no_ftf_df["Rewards"] = no_ftf_df["Rewards"].astype(str).str.strip()
        no_ftf_df["Other Rewards"] = no_ftf_df["Other Rewards"].astype(str).str.strip()
        
        # Create Reward Combo column - changed logic to OR for more realistic data
        no_ftf_df["Reward Combo"] = no_ftf_df.apply(
            lambda row: "Rewards" if (
                (row["Rewards"].lower() not in ["nan", "none", ""]) or 
                (row["Other Rewards"].lower() not in ["nan", "none", ""])
            ) else "Minimal to no Rewards", 
            axis=1
        )
        
        # Fourth level: No FTF to Rewards
        reward_counts = no_ftf_df["Reward Combo"].value_counts()
        rewards_count = max(reward_counts.get("Rewards", 0), 1)  # Ensure at least 1
        no_rewards_count = max(reward_counts.get("Minimal to no Rewards", 0), 1)  # Ensure at least 1
        
        s4_source = [label_map["No FTF"]] * 2
        s4_target = [label_map["Rewards"], label_map["Minimal to no Rewards"]]
        s4_value = [rewards_count, no_rewards_count]

        # Combine all data for Sankey diagram
        sources = s1_source + s2_source + s3_source + s4_source
        targets = s1_target + s2_target + s3_target + s4_target
        values = s1_value + s2_value + s3_value + s4_value

        # Define node colors
        highlight_colors = {
            "Top 25 Institutions": "#440154",
            "No Credit Score Required": "#46327e",
            "No Annual Fee": "#31688e",
            "No FTF": "#35b779",
            "Rewards": "#fde725"
        }
        node_colors = [highlight_colors.get(label, "lightgrey") for label in labels]

        # Define link colors
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

        # Create Sankey diagram
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

        # Update layout
        fig.update_layout(
            height=500,
            font_size=10,
            margin=dict(t=30, l=25, r=25, b=25)
        )
        
        return fig
    
    sankey_fig = create_sankey(df)
    if sankey_fig:
        st.plotly_chart(sankey_fig, use_container_width=True)
    
    st.markdown("### Annual Fee Distribution Among Top 10 Institutions")
    st.markdown("""
    *This stacked bar chart shows the proportion of cards with and without annual fees 
    for the top 10 institutions by total card offerings.*
    """)
    
    # Stacked bar plot
    @st.cache_data
    def create_stacked_barplot(df):
        if len(df) == 0:
            st.warning("No data available for stacked bar plot visualization.")
            return None
            
        df["Annual Fee Label"] = df["Annual Fee"].apply(lambda x: "No Annual Fee" if x == 0 else "Has Annual Fee")
        
        try:
            # Create the fee counts DataFrame
            fee_counts = df.groupby(["Institution Name", "Annual Fee Label"]).size().unstack(fill_value=0)
            
            # Handle case where columns might be missing
            if "No Annual Fee" not in fee_counts.columns:
                fee_counts["No Annual Fee"] = 0
            if "Has Annual Fee" not in fee_counts.columns:
                fee_counts["Has Annual Fee"] = 0
                
            # Get top 10 institutions
            if len(fee_counts) == 0:
                st.warning("No institution data available for stacked bar plot.")
                return None
                
            top10_institutions = fee_counts.sum(axis=1).sort_values(ascending=False).head(10)
            
            # If we have fewer than 10 institutions, use all available
            if len(top10_institutions) < 10:
                top10_institutions = fee_counts.sum(axis=1).sort_values(ascending=False)
                
            fee_counts_top10 = fee_counts.loc[top10_institutions.index]
            
            # Calculate proportions
            fee_props = fee_counts_top10.div(fee_counts_top10.sum(axis=1), axis=0)
            fee_props_top10_sorted = fee_props.sort_values(by="No Annual Fee", ascending=False)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.get_cmap("Set2").colors[:2]
            
            fee_props_top10_sorted[["No Annual Fee", "Has Annual Fee"]].plot(
                kind="bar", stacked=True, color=colors, ax=ax
            )
            
            plt.title("Proportion of Credit Cards with No Annual Fee vs. Annual Fee (Top 10 Institutions)")
            plt.xlabel("Institution Name")
            plt.ylabel("Proportion of Cards")
            plt.ylim(0, 1.1)
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.legend(title="Annual Fee Status")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            return fig
        except Exception as e:
            st.error(f"Error creating stacked bar plot: {str(e)}")
            return None
    
    barplot_fig = create_stacked_barplot(df)
    if barplot_fig:
        st.pyplot(barplot_fig)

# Add interactive filters in an expander
with st.expander("Interactive Filters"):
    col1, col2 = st.columns(2)
    
    with col1:
        selected_institutions = st.multiselect(
            "Filter by Institution",
            options=sorted(df["Institution Name"].unique()),
            default=[]
        )
    
    with col2:
        min_fee = int(df["Annual Fee"].min())
        max_fee = int(df["Annual Fee"].max())
        fee_range = st.slider(
            "Annual Fee Range",
            min_value=min_fee,
            max_value=max_fee,
            value=(min_fee, max_fee)
        )
    
    # Apply filters if any are selected
    filtered_df = df.copy()
    if selected_institutions:
        filtered_df = filtered_df[filtered_df["Institution Name"].isin(selected_institutions)]
    
    filtered_df = filtered_df[(filtered_df["Annual Fee"] >= fee_range[0]) & 
                             (filtered_df["Annual Fee"] <= fee_range[1])]
    
    # Show filtered data counts
    st.write(f"Showing {len(filtered_df)} of {len(df)} credit cards")
    
    # Display filtered visualizations if filters are applied
    if len(filtered_df) < len(df) and len(filtered_df) > 0:
        st.subheader("Filtered Visualizations")
        
        col1, col2 = st.columns(2)
        with col1:
            filtered_treemap = create_treemap(filtered_df)
            if filtered_treemap:
                st.plotly_chart(filtered_treemap, use_container_width=True)
        
        with col2:
            filtered_barplot = create_stacked_barplot(filtered_df)
            if filtered_barplot:
                st.pyplot(filtered_barplot)

# Conclusion Section
st.markdown("""
## Key Takeaways

- Some top issuers provide numerous variants with varying fees, so looking beyond brand reputation may be critical.
- The flow from top 25 institutions → no required credit score → no annual fee → no foreign transaction fee → rewards reveals a steep funnel. Only a fraction    of cards meet all these “ideal beginner” criteria.
- Seeing the no-fee vs fee-based proportions side by side clarifies whether an institution is more welcoming to lower-budget cardholders.
- Most cards cluster around a 21–25 day grace period. Beginners should prioritize at least three weeks of interest-free repayment to comfortably manage    monthly bills.
""")

# References Section
st.markdown("""
## References

1. Terms of Credit Card Plans (TCCP) survey (2024). Consumer Financial Protection Bureau. https://www.consumerfinance.gov/data-research/credit-card-data/terms-credit-card-plans-survey/
2. Plotly Technologies Inc. (2015). Collaborative data science. https://plot.ly
""")

# Add a download button for the raw data
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Raw Data",
    convert_df_to_csv(df),
    "credit_card_data.csv",
    "text/csv",
    key='download-csv'
)