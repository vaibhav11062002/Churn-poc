import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from collections import Counter
from itertools import combinations
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from google.cloud import storage  # For GCS

from google import genai  # Google Gen AI SDK for Vertex AI

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("customer-insights")

# Config -- insert API key here as requested
API_KEY = "AQ.Ab8RN6Js_U257WMEUfBO4rOK3gXLe0elpojCzXT5vnUb0uYxjQ"  # Your Vertex AI key

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_BLOB_NAME = os.getenv("GCS_BLOB_NAME")

VERTEX_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "sonic-name-471217-d8")
VERTEX_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "europe-west1")

# Initialize Gen AI client with hardcoded API key for Vertex AI access
client = genai.Client(
    vertexai=True,
    api_key=API_KEY,
    project=VERTEX_PROJECT,
    location=VERTEX_LOCATION,
)

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")

CUSTOMER_COL = "Customer"
REVENUE_COL = "Net Value"
COMPANY_COL_CANDIDATES = [
    "Company Code", "company code", "company_code", "ccode to be billed", "c_code", "ccode"
]
SALES_DOC_CANDIDATES = [
    "Sales Document", "Sales Document Number", "Billing Document", "Billing Doc",
    "Invoice Number", "Invoice", "Document Number"
]

KEEP_COLS = [
    "Billing Date", "Created On", "Item Description",
    "Material Group", "Distribution Channel", "Terms of Payment",
    "Order Quantity", "Net Price", "Net Value", "Document Currency"
]

# -- Helper functions (unchanged, same as before) --

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lc:
            return cols_lc[name.lower()]
    return None

def resilient_read_csv(filepath_or_buffer: Union[str, Path, io.StringIO]) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp1252", "latin1"]
    for enc in encodings:
        try:
            return pd.read_csv(filepath_or_buffer, dtype=str, encoding=enc, engine="c", sep=",", low_memory=False)
        except Exception:
            if isinstance(filepath_or_buffer, io.StringIO):
                filepath_or_buffer.seek(0)
            pass
    for enc in encodings:
        try:
            return pd.read_csv(filepath_or_buffer, dtype=str, encoding=enc, engine="python", sep=None)
        except Exception:
            if isinstance(filepath_or_buffer, io.StringIO):
                filepath_or_buffer.seek(0)
            pass
    raise ValueError("Failed to parse CSV after multiple strategies.")

def load_df_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    if not bucket_name or not blob_name:
        raise ValueError("GCS_BUCKET_NAME and GCS_BLOB_NAME must be set.")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    logger.info("Downloading CSV from GCS bucket %s blob %s", bucket_name, blob_name)
    data_string = blob.download_as_text()
    return resilient_read_csv(io.StringIO(data_string))

def sanitize_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("|", "/")
    return s

def norm_date(d):
    if pd.isna(d):
        return ""
    val = pd.to_datetime(d, errors="coerce")
    return val.strftime("%Y%m%d") if pd.notna(val) else ""

def norm_num(x, ndigits=2):
    if pd.isna(x):
        return ""
    try:
        return str(round(float(x), ndigits))
    except (ValueError, TypeError):
        return ""

def build_codebook(series: pd.Series) -> Dict[str, str]:
    vc = series.dropna().astype(str).value_counts()
    return {v: f"x{i+1}" for i, v in enumerate(vc.index.tolist())}

def full_transaction_block_for_customer(cust_df: pd.DataFrame) -> Tuple[str, int]:
    g = cust_df.copy()
    date_cols = [c for c in ["Billing Date", "Created On"] if c in g.columns]
    date_col = date_cols[0] if date_cols else None
    if date_col:
        g = g.sort_values(by=[date_col]).reset_index(drop=True)
    keep_cols = [c for c in KEEP_COLS if c in g.columns]
    codebooks: Dict[str, Dict[str, str]] = {}
    cat_cols = [c for c in ["Item Description", "Material Group", "Distribution Channel", "Terms of Payment", "Document Currency"] if c in g.columns]
    for c in cat_cols:
        codebooks[c] = build_codebook(g[c])
    header = "COLUMNS|" + "|".join([sanitize_text(c) for c in keep_cols])
    code_header = "CODES|" + json.dumps(codebooks, separators=(",", ":"), ensure_ascii=False)
    data_lines = []
    for _, row in g.iterrows():
        fields = []
        for c in keep_cols:
            if c in ["Billing Date", "Created On"]:
                fields.append(norm_date(row.get(c)))
            elif c in ["Order Quantity", "Net Price", "Net Value"]:
                fields.append(norm_num(row.get(c)))
            elif c in cat_cols:
                raw = sanitize_text(row.get(c))
                fields.append(codebooks[c].get(raw, "x0"))
            else:
                fields.append(sanitize_text(row.get(c)))
        data_lines.append("ROW|" + "|".join(fields))
    compact_text = "\n".join([header, code_header] + data_lines)
    return compact_text, len(data_lines)

def quarter_key_from_period_str(s: str) -> str:
    return s.replace("Q", "-Q")

def compute_aggregates_for_customer(cust_df: pd.DataFrame) -> Dict[str, Any]:
    a: Dict[str, Any] = {}
    g = cust_df.copy()
    date_cols = [c for c in ["Billing Date", "Created On"] if c in g.columns]
    date_col = date_cols[0] if date_cols else None
    if "Document Currency" in g.columns:
        cur_counts = g["Document Currency"].dropna().astype(str).value_counts()
        a["currency_mode"] = (cur_counts.index[0] if not cur_counts.empty else "")
    else:
        a["currency_mode"] = ""
    if "Net Value" in g.columns:
        g["Net Value"] = pd.to_numeric(g["Net Value"], errors="coerce")
    a["total_revenue_local"] = float(g["Net Value"].fillna(0).sum()) if "Net Value" in g.columns else 0.0
    revenue_by_year: Dict[str, float] = {}
    revenue_by_quarter: Dict[str, float] = {}
    if date_col:
        g[date_col] = pd.to_datetime(g[date_col], errors="coerce")
        g_valid = g[pd.notna(g[date_col])].copy()
        if "Net Value" in g_valid.columns:
            rev_series = g_valid.groupby(g_valid[date_col].dt.year)["Net Value"].sum(min_count=1)
            revenue_by_year = {str(int(k)): float(v) for k, v in rev_series.fillna(0).to_dict().items()}
            q_series = g_valid.groupby(g_valid[date_col].dt.to_period("Q"))["Net Value"].sum(min_count=1)
            revenue_by_quarter = {quarter_key_from_period_str(str(k)): float(v) for k, v in q_series.fillna(0).to_dict().items()}
        dts = g_valid[date_col].sort_values().dropna().astype("datetime64[ns]")
        diffs = dts.diff().dt.days.dropna()
        a["median_days_between_orders"] = float(np.median(diffs)) if not diffs.empty else 0.0
    else:
        a["median_days_between_orders"] = 0.0
    a["revenue_by_year"] = revenue_by_year
    a["revenue_by_quarter"] = revenue_by_quarter
    item_col = None
    if "Item Description" in g.columns:
        item_col = "Item Description"
    elif "Material Group" in g.columns:
        item_col = "Material Group"
    a["top_materials_by_revenue"] = []
    a["price_stats"] = []
    a["top_copurchase_pairs"] = []
    if item_col and "Net Value" in g.columns:
        totals = g.groupby(item_col)["Net Value"].sum(min_count=1).fillna(0).sort_values(ascending=False)
        grand = float(totals.sum()) if not totals.empty else 0.0
        top_rows = totals.head(10)
        top_materials = []
        for mat, val in top_rows.items():
            pct = (float(val) / grand) if grand > 0 else 0.0
            top_materials.append({"material": str(mat), "revenue": float(val), "share": round(pct, 4)})
        a["top_materials_by_revenue"] = top_materials
        if "Net Price" in g.columns:
            price_stats = []
            for mat, sub in g.groupby(item_col):
                prices = pd.to_numeric(sub["Net Price"], errors="coerce").dropna()
                if prices.empty:
                    continue
                avg = float(prices.mean())
                std = float(prices.std(ddof=0)) if len(prices) > 1 else 0.0
                cv = (std / avg) if avg > 0 else 0.0
                price_stats.append({"material": str(mat), "avg_price": round(avg, 4), "cv": round(cv, 4)})
            if top_materials:
                top_set = {t["material"] for t in top_materials}
                price_stats_sorted = sorted(price_stats, key=lambda d: (d["material"] not in top_set, d["material"]))
                a["price_stats"] = price_stats_sorted[:20]
            else:
                a["price_stats"] = price_stats[:20]
        date_key = None
        if "Billing Date" in g.columns and g["Billing Date"].notna().any():
            date_key = "Billing Date"
        elif "Created On" in g.columns and g["Created On"].notna().any():
            date_key = "Created On"
        pair_counts = Counter()
        if date_key:
            g[date_key] = pd.to_datetime(g[date_key], errors="coerce")
            for _, sub in g.groupby(g[date_key].dt.date):
                mats = sorted(set(sub[item_col].dropna().astype(str).tolist()))
                if len(mats) < 2:
                    continue
                for a_mat, b_mat in combinations(mats, 2):
                    pair_counts[(a_mat, b_mat)] += 1
        top_pairs = [{"a": a_m, "b": b_m, "count": int(c)} for (a_m, b_m), c in pair_counts.most_common(5)]
        a["top_copurchase_pairs"] = top_pairs
    a["total_records"] = int(len(g))
    if date_col and g[date_col].notna().any():
        a["distinct_days"] = int(g[date_col].dt.date.nunique())
    else:
        a["distinct_days"] = 0
    return a

def cluster_one_company(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["rev_pos"] = g["total_revenue"].clip(lower=0)
    if g[CUSTOMER_COL].nunique() < 3:
        ranks = g["rev_pos"].rank(method="first", ascending=False)
        labels = np.where(
            ranks <= 1, "high_revenue",
            np.where(ranks <= 2, "mixed_revenue", "low_revenue")
        )
        g["cluster_name"] = labels
        g["cluster_id"] = g["cluster_name"].map({"high_revenue": 0, "mixed_revenue": 1, "low_revenue": 2}).fillna(2).astype(int)
        return g.drop(columns=["rev_pos"])
    X = np.log1p(g["rev_pos"]).to_numpy().reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=10, random_state=42)
    g["km_id"] = km.fit_predict(X)
    means = g.groupby("km_id")["rev_pos"].mean().sort_values(ascending=False)
    order = {cid: idx for idx, cid in enumerate(means.index)}
    g["cluster_id"] = g["km_id"].map(order)
    g["cluster_name"] = g["cluster_id"].map({0: "high_revenue", 1: "mixed_revenue", 2: "low_revenue"})
    return g.drop(columns=["rev_pos", "km_id"])

# Load and cluster on startup
@app.on_event("startup")
async def startup_event():
    global raw_df, clustered_data
    raw_df = load_df_from_gcs(GCS_BUCKET_NAME, GCS_BLOB_NAME)
    for col in ["Created On", "Billing Date"]:
        if col in raw_df.columns:
            raw_df[col] = pd.to_datetime(raw_df[col], errors="coerce")
    if CUSTOMER_COL not in raw_df.columns:
        raise ValueError(f"Missing required column: {CUSTOMER_COL}")
    company_col = find_col(raw_df, COMPANY_COL_CANDIDATES)
    if company_col is None:
        company_col = "company_code"
        raw_df[company_col] = "UNKNOWN"
    if REVENUE_COL not in raw_df.columns:
        raise ValueError(f"Missing revenue column: {REVENUE_COL}")
    s = raw_df[REVENUE_COL].astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.\-]", "", regex=True)
    raw_df[REVENUE_COL] = pd.to_numeric(s, errors="coerce").fillna(0.0)
    clustered_list = []
    agg = (
        raw_df.groupby([company_col, CUSTOMER_COL], dropna=False)[REVENUE_COL]
              .sum(min_count=1)
              .reset_index()
              .rename(columns={REVENUE_COL: "total_revenue"})
    )
    for _, g in agg.groupby(company_col, dropna=False):
        clustered_list.append(cluster_one_company(g))
    clustered_data = pd.concat(clustered_list, ignore_index=True)
    clustered_data = clustered_data.rename(
        columns={company_col: "company_code", CUSTOMER_COL: "customer"}
    )
    logger.info("Loaded raw_df with %d rows and performed clustering into %d records", len(raw_df), len(clustered_data))

PROMPT_TEMPLATE = """
You are a data analyst...
... (Use your full prompt template here, unchanged)
""".strip()

def build_main_prompt(customer_id: str, known_total_revenue: float,
                      aggregates_json: Dict[str, Any], compact_block: str,
                      context_json: Dict[str, Any]) -> str:
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("[[CUSTOMER_ID]]", json.dumps(customer_id, ensure_ascii=False))
    prompt = prompt.replace("[[KNOWN_TOTAL_REVENUE]]", str(round(float(known_total_revenue or 0.0), 4)))
    prompt = prompt.replace("[[AGGREGATES_JSON]]", json.dumps(aggregates_json, separators=(",", ":"), ensure_ascii=False))
    prompt = prompt.replace("[[CONTEXT_JSON]]", json.dumps(context_json, separators=(",", ":"), ensure_ascii=False))
    prompt = prompt.replace("[[COMPACT_BLOCK]]", compact_block)
    return prompt

def try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if m:
        snippet = m.group(0)
        try:
            return json.loads(snippet)
        except Exception:
            pass
    return None

def coerce_to_schema_with_cluster(obj: Dict[str, Any], customer_id: str, cluster_name: str) -> Dict[str, Any]:
    # Same as before; trim strings and force schema compliance.
    # (Include your definition here unchanged)
    ...

app = FastAPI(title="Customer Clustering and Insights API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "rows": int(len(raw_df)), "clustered_rows": int(len(clustered_data))}

@app.get("/clustered-data")
async def get_clustered_data():
    return clustered_data.to_dict(orient="records")

@app.get("/customer-insights/{customer_id}")
async def get_customer_insights(customer_id: str, debug: bool = Query(False)):
    cust_df = raw_df[raw_df[CUSTOMER_COL].astype(str) == str(customer_id)]
    if cust_df.empty:
        raise HTTPException(status_code=404, detail="Customer not found")
    known_total_revenue = float(cust_df[REVENUE_COL].fillna(0).sum())
    aggregates_json = compute_aggregates_for_customer(cust_df)
    compact, nlines = full_transaction_block_for_customer(cust_df)
    row = clustered_data[clustered_data["customer"] == str(customer_id)].head(1).to_dict("records")
    ctx = row[0] if row else {}
    context_json = {
        "cluster_name": ctx.get("cluster_name", ""),
        "revenue_rank_in_cluster": int(ctx.get("revenue_rank_in_cluster", 0) or 0),
        "purchasing_frequency": int(ctx.get("purchasing_frequency", 0) or 0),
        "known_total_revenue_from_cluster": float(ctx.get("total_revenue", known_total_revenue)),
    }
    prompt = build_main_prompt(
        customer_id=customer_id,
        known_total_revenue=known_total_revenue,
        aggregates_json=aggregates_json,
        compact_block=compact,
        context_json=context_json,
    )
    raw_text = ""
    try:
        resp = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )
        raw_text = (resp.text or "")
        parsed = try_parse_json(raw_text)
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        parsed = None
    cluster_name = context_json.get("cluster_name", "")
    if isinstance(parsed, dict):
        coerced = coerce_to_schema_with_cluster(parsed, customer_id=customer_id, cluster_name=cluster_name)
        if debug:
            return {"result": coerced, "debug": {"raw_text_head": raw_text[:400], "parsed": True}}
        return coerced
    fallback = coerce_to_schema_with_cluster({}, customer_id=customer_id, cluster_name=cluster_name)
    if debug:
        return {"result": fallback, "debug": {"raw_text_head": raw_text[:400], "parsed": False}}
    return fallback
