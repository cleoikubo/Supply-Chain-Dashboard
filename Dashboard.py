import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from supply_chain_engine import SupplyChainEngine


st.set_page_config(
 page_title="Supply Chain Intelligence",
 layout="wide"
)

st.title("Supply Chain Intelligence Dashboard")


uploaded_file = st.file_uploader(
 "Upload Supply Chain Dataset",
 type=["csv", "xlsx"]
)

if uploaded_file:

 engine = SupplyChainEngine(uploaded_file)

 # -----------------------------
 # KPI SECTION
 # -----------------------------
 kpis = engine.get_kpis()

 col1, col2, col3 = st.columns(3)

 col1.metric("Total Orders",
 int(kpis["total_orders"]))

 col2.metric("Average Lead Time",
 round(kpis["avg_lead_time"], 2))

 col3.metric("Average Fulfillment Rate",
 round(kpis["avg_fulfillment"], 2))

 # -----------------------------
 # FILTERS
 # -----------------------------
 st.sidebar.header("Filters")

 supplier_filter = st.sidebar.selectbox(
 "Select Supplier",
 ["All"] + list(engine.df["supplier"].unique())
 )

 if supplier_filter != "All":
   engine.df = engine.df[
 engine.df["supplier"] == supplier_filter
 ]

 # -----------------------------
 # MONTHLY TREND
 # -----------------------------
 st.subheader("Monthly Order Trend")

 trend = engine.monthly_trend()

 fig1, ax1 = plt.subplots()

 sns.lineplot(
 data=trend,
 x="month",
 y="quantity_ordered",
 ax=ax1
 )

 plt.xticks(rotation=45)
 st.pyplot(fig1)

 # -----------------------------
 # DISTRIBUTION
 # -----------------------------
 st.subheader("Lead Time Distribution")

 fig2, ax2 = plt.subplots()

 sns.histplot(
 engine.df["lead_time_days"],
 kde=True,
 ax=ax2
 )

 st.pyplot(fig2)

 # -----------------------------
 # CLUSTERING
 # -----------------------------
 st.subheader("Supplier Clustering")

 cluster_number = st.slider(
 "Number of Clusters",
 2,
 6,
 3
 )

 cluster_df = engine.cluster_suppliers(cluster_number)

 fig3, ax3 = plt.subplots()

 sns.scatterplot(
 data=cluster_df,
 x="lead_time_days",
 y="cost_per_unit",
 hue="cluster",
 size="fulfillment_rate",
 ax=ax3
 )

 st.pyplot(fig3)
 st.dataframe(cluster_df)

 # -----------------------------
 # ANOMALY DETECTION
 # -----------------------------
 st.subheader("Anomaly Detection")

 anomalies = engine.detect_anomalies()

 fig4, ax4 = plt.subplots()

 sns.scatterplot(
 data=engine.df,
 x="lead_time_days",
 y="cost_per_unit",
 hue="anomaly",
 ax=ax4
 )

 st.pyplot(fig4)
 st.dataframe(anomalies)

 # -----------------------------
 # RISK CLASSIFICATION
 # -----------------------------
 st.subheader("Supplier Risk Classification")

 risk_df, accuracy = engine.classify_risk()

 fig5, ax5 = plt.subplots()

 sns.countplot(
 data=risk_df,
 x="risk",
 ax=ax5
 )

 st.pyplot(fig5)

 st.write("Model Accuracy:", round(accuracy, 3))

 # -----------------------------
 # AUTOMATED INSIGHTS
 # -----------------------------
 st.subheader("Automated Insights")

 if kpis["avg_lead_time"] > 18:
    st.warning(
 "Average lead time is high. Consider evaluating suppliers."
 )

 if kpis["avg_fulfillment"] < 0.85:
    st.error(
 "Fulfillment rate is below optimal threshold."
 )

 if len(anomalies) > 0:
    st.info(
 "Anomalies detected in supply chain data."
 )