import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class SupplyChainEngine:
    def __init__(self, file): self.df = self.load_data(file) , self.prepare_features()

# -----------------------------
# DATA LOADING
# -----------------------------
def load_data(self, file):
    filename = file.name.lower()
    if filename.endswith(".xlsx"):
        df = pd.read_excel(file)
    elif filename.endswith("csv"):
        df = pd.read_csv(file)
    else:
        raise ValueError("Unsupported file type")
    df["date"] = pd.to_datetime(df["date"])
    return df

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def prepare_features(self):
    self.df["fulfillment_rate"] = (
self.df["quantity_received"] /
self.df["quantity_ordered"]
)
    self.df["month"] = (
self.df["date"]
.dt.to_period("M")
.astype(str)
)

# -----------------------------
# KPI METRICS
# -----------------------------
def get_kpis(self): 
    return {
"total_orders": self.df["quantity_ordered"].sum(),
"avg_lead_time": self.df["lead_time_days"].mean(),
"avg_fulfillment": self.df["fulfillment_rate"].mean()
}

# -----------------------------
# MONTHLY TREND
# -----------------------------
def monthly_trend(self):
    return (
self.df.groupby("month")["quantity_ordered"]
.sum()
.reset_index()
)

# -----------------------------
# SUPPLIER SUMMARY
# -----------------------------
def supplier_summary(self):
    return (
self.df.groupby("supplier")
.agg({
"quantity_ordered": "sum",
"quantity_received": "sum",
"lead_time_days": "mean",
"cost_per_unit": "mean",
"fulfillment_rate": "mean"
})
.reset_index()
)

# -----------------------------
# CLUSTERING
# -----------------------------
def cluster_suppliers(self, n_clusters=3):
    supplier_df = self.supplier_summary()
    features = supplier_df[
["lead_time_days",
"cost_per_unit",
"fulfillment_rate"]
] 
    scaler = StandardScaler() , 
    scaled = scaler.fit_transform(features)
    kmeans = KMeans( 
        n_clusters = n_clusters,
random_state=42,
n_init=10)
    supplier_df["cluster"] = (
kmeans.fit_predict(scaled)
)
    return supplier_df

# -----------------------------
# Anomalies Detection
# -----------------------------
def detect_anomalies(self):
    features = self.df[
        ["lead_time_days",
"cost_per_unit",
"fulfillment_rate"]]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    iso = IsolationForest(
contamination=0.05,
random_state=42
) 
    self.df["anomaly"] = (
iso.fit_predict(scaled)
)
    return self.df[self.df["anomaly"] == -1]

# -----------------------------
# RISK CLASSIFICATION
# -----------------------------
def classify_risk(self):
    df_model = self.df.copy()
    conditions = [
    (df_model["lead_time_days"] > 20),
(df_model["fulfillment_rate"] < 0.8)]
    df_model["risk"] = np.select(conditions,["High", "Medium"],default="Low")
    features = df_model[
["lead_time_days",
"cost_per_unit",
"fulfillment_rate"]]
    labels = df_model["risk"]
    X_train, X_test, y_train, y_test = train_test_split(features,labels,
test_size=0.2,random_state=42) 
    model = RandomForestClassifier() 
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return df_model, accuracy