import os
import logging
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI
from google.cloud import storage
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-cluster-api")

# Config from environment
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_BLOB_NAME = os.getenv("GCS_BLOB_NAME")

CUSTOMER_COL = "Customer"
REVENUE_COL = "Net Value"
COMPANY_COL_CANDIDATES = [
    "Company Code", "company code", "company_code", "ccode to be billed", "c_code", "ccode"
]

app = FastAPI(title="Simple Customer Clustering API")

def find_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols_lc = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lc:
            return cols_lc[name.lower()]
    return None

def load_df_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    if not bucket_name or not blob_name:
        raise ValueError("GCS_BUCKET_NAME and GCS_BLOB_NAME must be set")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info("Downloading CSV from GCS bucket %s blob %s", bucket_name, blob_name)
    data_string = blob.download_as_text()
    return pd.read_csv(pd.io.common.StringIO(data_string), dtype=str)

def cluster_customers(df: pd.DataFrame) -> pd.DataFrame:
    company_col = find_col(df, COMPANY_COL_CANDIDATES)
    if company_col is None:
        company_col = "company_code"
        df[company_col] = "UNKNOWN"

    df[REVENUE_COL] = (
        df[REVENUE_COL]
        .str.replace(",", "", regex=False)
        .str.replace(r"[^\d\.\-]", "", regex=True)
        .astype(float)
        .fillna(0)
    )
    agg = (
        df.groupby([company_col, CUSTOMER_COL], dropna=False)[REVENUE_COL]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={REVENUE_COL: "total_revenue"})
    )
    agg["total_revenue"] = agg["total_revenue"].fillna(0.0).astype(float)

    def cluster_one_company(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["rev_pos"] = g["total_revenue"].clip(lower=0)
        if g[CUSTOMER_COL].nunique() < 3:
            labels = np.where(
                g["rev_pos"].rank(method="first", ascending=False) <= 1,
                "high_revenue",
                "low_revenue",
            )
            g["cluster_name"] = labels
            g["cluster_id"] = g["cluster_name"].map({"high_revenue": 0, "low_revenue": 1}).fillna(1).astype(int)
            return g.drop(columns=["rev_pos"])
        X = np.log1p(g["rev_pos"]).to_numpy().reshape(-1, 1)
        km = KMeans(n_clusters=2, n_init=10, random_state=42)
        g["km_id"] = km.fit_predict(X)
        means = g.groupby("km_id")["rev_pos"].mean().sort_values(ascending=False)
        order = {cid: idx for idx, cid in enumerate(means.index)}
        g["cluster_id"] = g["km_id"].map(order)
        g["cluster_name"] = g["cluster_id"].map({0: "high_revenue", 1: "low_revenue"})
        return g.drop(columns=["rev_pos", "km_id"])

    clustered_list = []
    for _, g in agg.groupby(company_col, dropna=False):
        clustered_list.append(cluster_one_company(g))
    clustered_df = pd.concat(clustered_list, ignore_index=True)
    clustered_df = clustered_df.rename(
        columns={company_col: "company_code", CUSTOMER_COL: "customer"}
    )
    return clustered_df[["company_code", "customer", "total_revenue", "cluster_name", "cluster_id"]]

@app.on_event("startup")
async def startup_event():
    global clustered_data
    try:
        df = load_df_from_gcs(GCS_BUCKET_NAME, GCS_BLOB_NAME)
        logger.info("Loaded CSV with shape %s", df.shape)
        clustered_data = cluster_customers(df)
        logger.info("Clustering done with %d rows", len(clustered_data))
    except Exception as e:
        logger.error("Failed to load or cluster data: %s", e)
        clustered_data = pd.DataFrame()

@app.get("/")
async def health():
    return {"status": "ok", "clustered_rows": len(clustered_data)}

@app.get("/clustered-data")
async def get_clustered_data():
    return clustered_data.to_dict(orient="records")
