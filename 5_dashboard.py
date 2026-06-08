"""
Step 5: Streamlit Dashboard
Author: Student
Course: Retail Analytics Project

This script builds a premium interactive dashboard to explore customer segments,
predict individual customer churn via the FastAPI backend, and understand model decisions.

Tabs:
1. Customer Segment Analysis (PCA Scatter, Sizes, Radar Comparison)
2. Live Churn Predictor (Input sliders, calls API, displays colored risk gauge & suggestions)
3. Model Interpretation (SHAP global importance plots and explanations)

To run this dashboard:
streamlit run 5_dashboard.py
"""

import os
import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration with premium tab title and icon
st.set_page_config(
    page_title="Retail Customer Segmentation & Churn Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling (Inter font, dark glassmorphism cards, micro-interactions)
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        /* Apply fonts */
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        /* Custom Header Styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, #1f77b4, #2ca02c);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        
        .sub-header {
            font-size: 1.1rem;
            color: #666;
            margin-bottom: 2rem;
        }
        
        /* Metric Card Container */
        .metric-card {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }
        
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #1f77b4;
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
        }
        
        /* Risk Badges */
        .risk-gauge-bg {
            background-color: #f0f2f6;
            border-radius: 10px;
            height: 24px;
            width: 100%;
            margin-top: 10px;
        }
        
        .risk-gauge-fill {
            border-radius: 10px;
            height: 24px;
            text-align: center;
            color: white;
            font-size: 0.8rem;
            font-weight: bold;
            line-height: 24px;
        }
        
        /* Custom buttons styling */
        div.stButton > button {
            background: linear-gradient(135deg, #1f77b4 0%, #175a8a 100%);
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(31, 119, 180, 0.2);
        }
        
        div.stButton > button:hover {
            transform: scale(1.03);
            box-shadow: 0 6px 12px rgba(31, 119, 180, 0.4);
            background: linear-gradient(135deg, #175a8a 0%, #114368 100%);
        }
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# Data & API Config
# ----------------------------------------------------
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
SEGMENTED_DATA_PATH = "data/processed/segmented_customers.csv.gz"

# Title bar
st.markdown('<div class="main-header">Retail Customer Segmentation & Churn Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">An End-to-End Analytics and Predictive Modeling System for Olist Brazilian E-Commerce</div>', unsafe_allow_html=True)

# ----------------------------------------------------
# Main Tabs
# ----------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Customer Segments Overview", "🔮 Churn Predictor", "🧠 Model Explainability & SHAP"])

# ----------------------------------------------------
# TAB 1: Customer Segments Overview
# ----------------------------------------------------
with tab1:
    st.header("Customer Segmentation (Unsupervised Learning)")
    st.markdown("""
        We applied **K-Means Clustering** on standardized Recency, Frequency, and Monetary (RFM) features to segment **96,096 unique customers** into 5 distinct groups.
    """)
    
    # Load dataset
    if os.path.exists(SEGMENTED_DATA_PATH):
        df_segmented = pd.read_csv(SEGMENTED_DATA_PATH)
        
        # 1. Metric Cards Row
        total_customers = len(df_segmented)
        avg_recency_all = df_segmented['recency'].mean()
        avg_frequency_all = df_segmented['frequency'].mean()
        avg_monetary_all = df_segmented['monetary'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Customers</div>
                    <div class="metric-value">{total_customers:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Recency</div>
                    <div class="metric-value">{avg_recency_all:.1f} days</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Avg Order Frequency</div>
                    <div class="metric-value">{avg_frequency_all:.2f} orders</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Average Total Spend</div>
                    <div class="metric-value">R$ {avg_monetary_all:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # 2. Main Plots
        plot_col1, plot_col2 = st.columns([6, 5])
        
        with plot_col1:
            st.subheader("2D PCA Projection of Customer Segments")
            st.markdown("We reduced the scaled 3D RFM features to 2D using Principal Component Analysis (PCA) to visualize the clusters.")
            
            # Downsample for faster Plotly rendering (10k points)
            df_plot_sample = df_segmented.sample(n=min(10000, len(df_segmented)), random_state=42)
            
            colors_map = {
                'Champions': '#2ca02c',
                'Loyal Customers': '#1f77b4',
                'New Customers': '#ff7f0e',
                'At-Risk Customers': '#d62728',
                'Lost Customers': '#7f7f7f'
            }
            
            fig_pca = px.scatter(
                df_plot_sample, x='pca_1', y='pca_2',
                color='segment', color_discrete_map=colors_map,
                hover_data=['recency', 'frequency', 'monetary'],
                opacity=0.6,
                labels={'pca_1': 'Principal Component 1', 'pca_2': 'Principal Component 2'}
            )
            fig_pca.update_layout(
                legend_title_text='Customer Segments',
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
        with plot_col2:
            st.subheader("Customer Segment Distribution")
            segment_counts = df_segmented['segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Customers']
            
            # Pie Chart
            fig_pie = px.pie(
                segment_counts, values='Customers', names='Segment',
                color='Segment', color_discrete_map=colors_map,
                hole=0.4
            )
            fig_pie.update_layout(margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        # 3. Centroids and Radar Chart
        st.subheader("RFM Profile Comparison (Radar Chart)")
        st.markdown("""
            Since Recency, Frequency, and Monetary have drastically different scales, 
            we **Min-Max scale the centroids** [0, 1] to visualize relative strengths without distortion.
        """)
        
        centroids = df_segmented.groupby('segment')[['recency', 'frequency', 'monetary']].mean()
        
        # Normalized Centroids for Radar Chart
        min_vals = centroids.min()
        max_vals = centroids.max()
        norm_centroids = (centroids - min_vals) / (max_vals - min_vals)
        
        categories = ['Recency (Days Since Last Order)', 'Frequency (Total Orders)', 'Monetary (Spend in R$)']
        
        fig_radar = go.Figure()
        for segment_name, row in norm_centroids.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['recency'], row['frequency'], row['monetary'], row['recency']],
                theta=categories + [categories[0]],
                fill='toself',
                name=segment_name,
                line_color=colors_map.get(segment_name)
            ))
            
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            margin=dict(l=40, r=40, t=20, b=20)
        )
        
        radar_col1, radar_col2 = st.columns([5, 6])
        with radar_col1:
            st.plotly_chart(fig_radar, use_container_width=True)
        with radar_col2:
            st.write("")
            st.write("**Centroid Summary Statistics Table**")
            # Format and show centroids table
            centroids_table = centroids.copy()
            centroids_table.columns = ['Mean Recency (Days)', 'Mean Frequency (Orders)', 'Mean Spend (R$)']
            centroids_table['Customer Count'] = df_segmented['segment'].value_counts()
            centroids_table = centroids_table.round(2)
            st.dataframe(centroids_table)
            
    else:
        st.warning("⚠️ Segmented data files are missing. Please run `1_data_prep.py` and `2_segmentation.py` to populate this tab.")

# ----------------------------------------------------
# TAB 2: Churn Predictor
# ----------------------------------------------------
with tab2:
    st.header("Real-Time Churn Risk Predictor")
    st.markdown("""
        Enter details about a customer below. The dashboard will query the **FastAPI backend** 
        to compute the customer's churn probability (no purchase in the next 180 days) and identify their segment.
    """)
    
    # Check if API is running
    api_online = True
    try:
        requests.get(API_URL, timeout=2)
    except requests.exceptions.RequestException:
        api_online = False
        st.error("⚠️ **FastAPI Backend is Offline.** Please start the API by running `uvicorn 4_api:app --reload` in your terminal to enable predictions.")
        
    st.subheader("Customer Characteristics Input")
    
    # Input layouts
    in_col1, in_col2, in_col3 = st.columns(3)
    
    with in_col1:
        recency = st.slider("Recency (Days since last purchase)", 1, 730, 45, help="Smaller value means more recently active.")
        frequency = st.slider("Frequency (Total orders placed)", 1, 30, 2, help="Number of unique order IDs.")
        monetary = st.number_input("Monetary Value (Total spend in R$)", min_value=1.0, max_value=20000.0, value=150.0, step=10.0)
        
    with in_col2:
        avg_review_score = st.slider("Average Review Score", 1.0, 5.0, 4.5, step=0.1, help="Mean score given by the customer in product reviews.")
        preferred_payment_method = st.selectbox(
            "Preferred Payment Method", 
            ["credit_card", "boleto", "voucher", "debit_card", "unknown"]
        )
        
    with in_col3:
        cancellation_rate = st.slider("Order Cancellation Rate", 0.0, 1.0, 0.0, step=0.05, help="Proportion of orders that were canceled.")
        # Auto-calculate AOV to assist the user
        avg_order_value = monetary / frequency
        st.metric("Calculated Average Order Value (AOV)", f"R$ {avg_order_value:.2f}")

    st.write("")
    
    # Run Prediction Button
    if st.button("🔮 Calculate Churn Risk"):
        if not api_online:
            st.error("Cannot predict: FastAPI backend is offline.")
        else:
            with st.spinner("Calling API model endpoints..."):
                payload = {
                    "recency": float(recency),
                    "frequency": int(frequency),
                    "monetary": float(monetary),
                    "avg_order_value": float(avg_order_value),
                    "avg_review_score": float(avg_review_score),
                    "preferred_payment_method": preferred_payment_method,
                    "cancellation_rate": float(cancellation_rate)
                }
                
                try:
                    response = requests.post(f"{API_URL}/predict", json=payload)
                    res_data = response.json()
                    
                    # Unpack response
                    prob = res_data["churn_probability"]
                    pred = res_data["churn_prediction"]
                    segment = res_data["segment_label"]
                    
                    st.success("Analysis Complete!")
                    
                    # UI Results Row
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Prediction Results")
                        st.markdown(f"**Customer Segment**: `{segment}`")
                        
                        # Set color based on risk levels
                        if prob < 0.3:
                            risk_color = "#2ca02c" # Green
                            risk_label = "LOW RISK"
                        elif prob < 0.7:
                            risk_color = "#ff7f0e" # Orange
                            risk_label = "MEDIUM RISK"
                        else:
                            risk_color = "#d62728" # Red
                            risk_label = "HIGH RISK"
                            
                        st.markdown(f"**Churn Risk Level**: <span style='color:{risk_color}; font-weight:bold;'>{risk_label}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Exact Churn Probability**: `{prob:.2%}`")
                        
                        # Horizontal Risk Bar
                        st.markdown(f"""
                            <div class="risk-gauge-bg">
                                <div class="risk-gauge-fill" style="background-color: {risk_color}; width: {prob*100:.1f}%;">
                                    {prob*100:.1f}% Risk
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    with res_col2:
                        st.subheader("Recommended Retention Action")
                        
                        # Generate tailored business action based on segment & churn risk
                        if pred == 1:
                            if segment == "Champions":
                                st.markdown("""
                                    🚨 **High Churn Risk Champion!**
                                    This customer was one of your most valuable spenders. 
                                    * **Retention Plan**: Send an exclusive VIP offer, offer direct phone line/concierge support, or issue a high-value discount code immediately.
                                """)
                            elif segment == "At-Risk Customers" or segment == "Lost Customers":
                                st.markdown("""
                                    ⚠️ **Standard Churn/Inactive Customer.**
                                    * **Retention Plan**: Re-engage via standard reactivation email campaigns. Offer generic win-back coupon (e.g., 'We miss you! Here is R$ 20 off').
                                """)
                            else:
                                st.markdown("""
                                    📉 **At-Risk Customer.**
                                    * **Retention Plan**: Offer personalized discounts based on their preferred payment method. Highlight popular items in their category.
                                """)
                        else:
                            if segment == "Champions":
                                st.markdown("""
                                    🌟 **Active Champion!**
                                    * **Action**: Reward loyalty without giving unnecessary discounts. Invite to early product launches or loyalty clubs.
                                """)
                            elif segment == "New Customers":
                                st.markdown("""
                                    🌱 **Active New Customer.**
                                    * **Action**: Send onboarding welcome emails, educate on customer support, and offer a coupon for their second purchase to build frequency.
                                """)
                            else:
                                st.markdown("""
                                    ✅ **Active Customer.**
                                    * **Action**: Maintain standard marketing cadences. Send periodic feedback surveys.
                                """)
                                
                except Exception as e:
                    st.error(f"API Request Failed: {e}")

# ----------------------------------------------------
# TAB 3: Model Explainability & SHAP
# ----------------------------------------------------
with tab3:
    st.header("Explainable AI (XAI) & Feature Importance")
    st.markdown("""
        A critical step in machine learning is understanding *why* models make certain predictions. 
        We use **SHAP (SHapley Additive exPlanations)** values based on cooperative game theory to measure feature contributions.
    """)
    
    col_xai1, col_xai2 = st.columns([6, 5])
    
    with col_xai1:
        st.subheader("Global Feature Importance (SHAP)")
        st.markdown("This beeswarm plot shows how each feature value impacts the model output (churn risk).")
        
        # Display the saved SHAP summary image
        shap_img_path = "reports/shap_importance.png"
        if os.path.exists(shap_img_path):
            st.image(shap_img_path, caption="SHAP Summary Plot (XGBoost)", use_column_width=True)
        else:
            st.warning("⚠️ SHAP plot image not found. Please run `3_churn_prediction.py` to generate it.")
            
    with col_xai2:
        st.subheader("Interpreting Churn Drivers")
        st.markdown("""
            **Key Insights from the SHAP Analysis:**
            
            1. **Cancellation Rate**:
               Customers with higher order cancellation rates have a strongly positive SHAP value, making it one of the largest drivers of churn. Unresolved canceled orders indicate poor experience.
               
            2. **Review Score**:
               Lower review scores (1 or 2 stars) push the model towards predicting churn. Positive customer reviews correlate heavily with customer retention.
               
            3. **Frequency**:
               Single-purchase customers (frequency=1) represent the vast majority of churned users. Encouraging customers to place their *second* order is the highest ROI retention strategy.
               
            4. **Monetary Spend & AOV**:
               Higher monetary spend slightly reduces churn risk, showing that big-ticket buyers tend to stay active longer, though frequency remains a stronger predictor than raw spend.
               
            5. **Segment Labels**:
               If a customer belongs to the `Lost Customers` or `At-Risk Customers` segment, this segment membership acts as a strong positive driver for churn prediction. Conversely, being in the `Champions` segment pushes the probability down.
        """)
        
        st.write("")
        st.subheader("Model Performance Summary")
        
        # Show model metrics comparison image if it exists
        model_img_path = "reports/model_comparison.png"
        if os.path.exists(model_img_path):
            st.image(model_img_path, caption="Logistic Regression vs Random Forest vs XGBoost Curves", use_column_width=True)
        else:
            st.info("Performance comparison plots will show here once `3_churn_prediction.py` is completed.")
