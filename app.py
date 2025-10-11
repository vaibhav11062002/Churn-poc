# app.py
# FastAPI on Cloud Run with:
# - CSV load from GCS
# - Customer clustering
# - Gemini LLM insights (service account OAuth2)
# - Proper CORS for React frontend
# - Detailed try/except logging

import os
import re
import json
import logging
from typing import Dict, Any, Tuple, List, Optional
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sklearn.cluster import KMeans
from google.cloud import storage
from google.oauth2 import service_account
import google.generativeai as genai

# =========================
# Logging
# =========================
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("customer-insights")

# =========================
# Config (env)
# =========================
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_BLOB_NAME = os.getenv("GCS_BLOB_NAME")

# CORS: comma-separated origins, e.g. "http://localhost:3000,https://your-frontend.app"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",") if o.strip()]
ALLOW_CREDENTIALS = os.getenv("ALLOW_CREDENTIALS", "false").lower() == "true"

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")

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

# =========================
# LLM Auth via Service Account JSON in env
# =========================
credentials_json_str = os.getenv("GEMINI_API_CREDENTIALS_JSON")
if not credentials_json_str:
    raise RuntimeError("GEMINI_API_CREDENTIALS_JSON environment variable is required for LLM auth")

try:
    keyfile_dict = json.loads(credentials_json_str)
    credentials = service_account.Credentials.from_service_account_info(keyfile_dict)
    genai.configure(credentials=credentials)
    model = genai.GenerativeModel(MODEL_NAME)
    logger.info("Configured Google Generative AI client with service account credentials")
except Exception as e:
    logger.exception("Failed to configure Generative AI client: %s", e)
    raise

# =========================
# Helpers
# =========================
def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols_lc = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in cols_lc:
            return cols_lc[name.lower()]
    return None

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
    except Exception:
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
            elif c in ["Item Description", "Material Group", "Distribution Channel", "Terms of Payment", "Document Currency"]:
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

    if item_col:
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
            for dt, sub in g.groupby(g[date_key].dt.date):
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

# =========================
# Prompt template (unchanged)
# =========================
PROMPT_TEMPLATE = """
You are a data analyst. Return ONLY a single JSON object matching the schema below (no markdown, no code blocks, no extra keys).

Task:
- Produce high-ROI retention insights grounded in ordering cadence, value, price sensitivity, product mix, and full transaction trends.

JSON schema to return:
{
  "customer": "<string>",
  "cluster": "<string>",                // echo the provided context.cluster_name; do NOT infer
  "churn": "yes|no" "predict this after analysing the whole transaction trend not just revenue.",
  "churn_analysis": "<max 20 words>" "predict this after analysing the whole transaction trend not just revenue and the quarterly revenue, also look for monthly revenue.",
  "retention_strategies": "<max 20 words>" "predict this after analysing the whole transaction trend.",
  "Retention_offers": "<max 20 words>" "predict this after analysing the whole transaction trend.",
  "Purchase_details": "<materials bought frequently + revenue, max 20 words>",
  "revenue_by_year": { "YYYY": number, "...": number },
  "revenue_by_quarter": { "YYYY-QN": number, "...": number },
  "trend_of_sales": "<max 40 words>",
  "product_combination": "<max 20 words>",
  "best_price_by_material": [ { "material": "<code or name>", "suggested_price": <number>, "discount": "<e.g. 5-10%>" } ],
  "observation": "<min 75 words>" "predict this after analysing the whole transaction trend.",
  "recommendation": "<min 75 words>" "predict this after analysing the whole transaction trend."
}

Inputs:
- customer_id = [[CUSTOMER_ID]]
- known_total_revenue = [[KNOWN_TOTAL_REVENUE]]
- aggregates_json = [[AGGREGATES_JSON]]
- context = [[CONTEXT_JSON]]  // includes cluster_name, revenue_rank_in_cluster, purchasing_frequency
- recent_history (FULL, all transactions for this customer; compacted):
  - The first line begins with COLUMNS| and lists field order.
  - The second line begins with CODES| and is a JSON map of categorical value→short code (unlisted→x0 as OTHER).
  - Each subsequent line begins with ROW| and provides one transaction aligned to COLUMNS.

Recent history (compressed):
[[COMPACT_BLOCK]]

Rules:
1) Use aggregates_json for totals, cadence, top materials, price stats, and combinations; recent_history is for cross-checking details.
2) For best_price_by_material: suggested_price = avg_price from aggregates_json.price_stats; discount: CV≥0.15→"8-12%", 0.07–0.15→"5-8%", else→"3-5%".
3) trend_of_sales: capture patterns across months and quarters; avoid relying solely on a single quarter.
4) observation and recommendation: each must be at least 75 words, business-specific and grounded in data.
5) If any field cannot be determined, use "" for strings, {} or [] for objects/arrays, and 0 for numbers.
6) Output MUST be a single JSON object exactly per schema; no extra keys.
""".strip()

def build_main_prompt(customer_id: str,
                      known_total_revenue: float,
                      aggregates_json: Dict[str, Any],
                      compact_block: str,
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
    out = {
        "customer": str(obj.get("customer", customer_id or "")),
        "cluster": str(cluster_name or obj.get("cluster", "")),
        "churn": str(obj.get("churn", "")),
        "churn_analysis": str(obj.get("churn_analysis", obj.get("reason_churn_decision", ""))),
        "retention_strategies": str(obj.get("retention_strategies", obj.get("how_to_retain", ""))),
        "Retention_offers": str(obj.get("Retention_offers", obj.get("offers_we_can_provide", ""))),
        "Purchase_details": str(obj.get("Purchase_details", obj.get("details", ""))),
        "revenue_by_year": {},
        "revenue_by_quarter": {},
        "trend_of_sales": str(obj.get("trend_of_sales", obj.get("trend_of_buying", ""))),
        "product_combination": str(obj.get("product_combination", "")),
        "best_price_by_material": [],
        "observation": str(obj.get("observation", "")),
        "recommendation": str(obj.get("recommendation", "")),
    }
    for dict_key, legacy in [("revenue_by_year", "revenue_by_year"), ("revenue_by_quarter", "revenue_by_quarter")]:
        rb = obj.get(dict_key, obj.get(legacy, {}))
        if isinstance(rb, dict):
            fixed = {}
            for k, v in rb.items():
                try:
                    fixed[str(k)] = float(v)
                except Exception:
                    fixed[str(k)] = 0.0
            out[dict_key] = fixed
    bp = obj.get("best_price_by_material", obj.get("best_price_by_material", []))
    if isinstance(bp, list):
        cleaned = []
        for item in bp:
            if not isinstance(item, dict):
                continue
            cleaned.append({
                "material": str(item.get("material", "")),
                "suggested_price": float(item.get("suggested_price", 0) or 0),
                "discount": str(item.get("discount", "")),
            })
        out["best_price_by_material"] = cleaned
    def trim_words(s: str, n: int) -> str:
        toks = str(s).split()
        return " ".join(toks[:n])
    out["churn_analysis"] = trim_words(out["churn_analysis"], 20)
    out["retention_strategies"] = trim_words(out["retention_strategies"], 20)
    out["Retention_offers"] = trim_words(out["Retention_offers"], 20)
    out["Purchase_details"] = trim_words(out["Purchase_details"], 20)
    out["trend_of_sales"] = trim_words(out["trend_of_sales"], 40)
    out["product_combination"] = trim_words(out["product_combination"], 20)
    churn = out["churn"].strip().lower()
    if churn not in {"yes", "no"}:
        out["churn"] = ""
    return out

# =========================
# GCS CSV load
# =========================
def load_df_from_gcs(bucket_name: str, blob_name: str) -> pd.DataFrame:
    try:
        if not bucket_name or not blob_name:
            raise ValueError("GCS_BUCKET_NAME and GCS_BLOB_NAME must be set")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        logger.info("Downloading CSV from GCS bucket %s blob %s", bucket_name, blob_name)
        data_string = blob.download_as_text()
        df = pd.read_csv(pd.io.common.StringIO(data_string), dtype=str)
        logger.info("CSV loaded with shape=%s", df.shape)
        return df
    except Exception as e:
        logger.exception("Failed to download or parse CSV from GCS: %s", e)
        raise

# =========================
# Cluster logic (3 clusters)
# =========================
def cluster_one_company(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["rev_pos"] = g["total_revenue"].clip(lower=0)
    if g["Customer"].nunique() < 3:
        ranks = g["rev_pos"].rank(method="first", ascending=False)
        labels = np.where(
            ranks <= 1, "high_revenue",
            np.where(ranks <= 2, "mixed_revenue", "low_revenue")
        )
        g["cluster_name"] = labels
        g["cluster_id"] = g["cluster_name"].map({"high_revenue": 0, "mixed_revenue": 1, "low_revenue": 2}).fillna(2).astype(int)
        return g.drop(columns=["rev_pos"])
    X = np.log1p(g["rev_pos"]).to_numpy().reshape(-1, 1)
    km = KMeans(n_clusters=3, n_init=50, random_state=42)
    g["km_id"] = km.fit_predict(X)
    means = g.groupby("km_id")["rev_pos"].mean().sort_values(ascending=False)
    order = {cid: idx for idx, cid in enumerate(means.index)}
    g["cluster_id"] = g["km_id"].map(order)
    g["cluster_name"] = g["cluster_id"].map({0: "high_revenue", 1: "mixed_revenue", 2: "low_revenue"})
    return g.drop(columns=["rev_pos", "km_id"])

# =========================
# Load data & cluster at startup
# =========================
try:
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

    if "Net Price" in raw_df.columns:
        p = raw_df["Net Price"].astype(str).str.replace(",", "", regex=False).str.replace(r"[^\d\.\-]", "", regex=True)
        raw_df["Net Price"] = pd.to_numeric(p, errors="coerce")

    raw_df[CUSTOMER_COL] = raw_df[CUSTOMER_COL].astype(str)

    logger.info("Non-null counts: Customer=%d, Net Value=%d, Created On=%d, Billing Date=%d",
        raw_df[CUSTOMER_COL].notna().sum(),
        raw_df[REVENUE_COL].notna().sum(),
        raw_df["Created On"].notna().sum() if "Created On" in raw_df.columns else -1,
        raw_df["Billing Date"].notna().sum() if "Billing Date" in raw_df.columns else -1
    )

    agg = (
        raw_df.groupby([company_col, CUSTOMER_COL], dropna=False)[REVENUE_COL]
              .sum(min_count=1)
              .reset_index()
              .rename(columns={REVENUE_COL: "total_revenue"})
    )
    agg["total_revenue"] = agg["total_revenue"].fillna(0.0).astype(float)

    sales_doc_col = find_col(raw_df, SALES_DOC_CANDIDATES)
    if sales_doc_col and sales_doc_col in raw_df.columns:
        pf = (
            raw_df.groupby([company_col, CUSTOMER_COL], dropna=False)[sales_doc_col]
                  .nunique(dropna=True)
                  .reset_index()
                  .rename(columns={sales_doc_col: "purchasing_frequency"})
        )
    else:
        pf = agg[[company_col, CUSTOMER_COL]].copy()
        pf["purchasing_frequency"] = 0

    clustered_list = []
    for _, g in agg.groupby(company_col, dropna=False):
        clustered_list.append(cluster_one_company(g))
    clustered = pd.concat(clustered_list, ignore_index=True)

    clustered["revenue_rank_in_cluster"] = (
        clustered.groupby([company_col, "cluster_id"])["total_revenue"]
                 .rank(method="dense", ascending=False)
                 .astype(int)
    )

    clustered = (
        clustered.merge(pf, on=[company_col, CUSTOMER_COL], how="left")
                 .rename(columns={company_col: "company_code", CUSTOMER_COL: "customer"})
    )

    clustered_data = clustered[[
        "company_code", "customer", "total_revenue", "cluster_name",
        "revenue_rank_in_cluster", "purchasing_frequency"
    ]].copy()

    logger.info("clustered_data rows=%d, sample=%s", len(clustered_data), clustered_data.head(3).to_dict("records"))

except Exception as e:
    logger.exception("Error during data load and clustering: %s", e)
    raw_df = pd.DataFrame()
    clustered_data = pd.DataFrame()

# =========================
# FastAPI app + CORS
# =========================
app = FastAPI(title="Customer Clustering and Insights API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok", "rows": int(len(raw_df)), "clustered_rows": int(len(clustered_data))}

@app.get("/clustered-data")
async def get_clustered_data():
    logger.info("GET /clustered-data")
    return clustered_data.to_dict(orient="records")

# =========================
# LLM endpoint
# =========================
def build_main_prompt(customer_id: str,
                      known_total_revenue: float,
                      aggregates_json: Dict[str, Any],
                      compact_block: str,
                      context_json: Dict[str, Any]) -> str:
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace("[[CUSTOMER_ID]]", json.dumps(customer_id, ensure_ascii=False))
    prompt = prompt.replace("[[KNOWN_TOTAL_REVENUE]]", str(round(float(known_total_revenue or 0.0), 4)))
    prompt = prompt.replace("[[AGGREGATES_JSON]]", json.dumps(aggregates_json, separators=(",", ":"), ensure_ascii=False))
    prompt = prompt.replace("[[CONTEXT_JSON]]", json.dumps(context_json, separators=(",", ":"), ensure_ascii=False))
    prompt = prompt.replace("[[COMPACT_BLOCK]]", compact_block)
    return prompt

@app.get("/customer-insights/{customer_id}")
async def get_customer_insights(customer_id: str, debug: bool = Query(False)):
    logger.info("GET /customer-insights/%s debug=%s", customer_id, debug)
    try:
        cust_df = raw_df[raw_df[CUSTOMER_COL].astype(str) == str(customer_id)]
        if cust_df.empty:
            raise HTTPException(status_code=404, detail="Customer not found")

        nonnull_created = int(cust_df["Created On"].notna().sum()) if "Created On" in cust_df.columns else 0
        nonnull_billed = int(cust_df["Billing Date"].notna().sum()) if "Billing Date" in cust_df.columns else 0
        known_total_revenue = float(cust_df[REVENUE_COL].fillna(0).sum())
        n_items = int(cust_df["Item Description"].notna().sum()) if "Item Description" in cust_df.columns else 0
        n_prices = int(cust_df["Net Price"].notna().sum()) if "Net Price" in cust_df.columns else 0

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

        logger.info(
            "cust_rows=%d rev_sum=%.2f all_rows=%d created_nz=%d billed_nz=%d items=%d prices=%d prompt_chars=%d",
            len(cust_df), known_total_revenue, nlines, nonnull_created, nonnull_billed, n_items, n_prices, len(prompt)
        )

        raw_text = ""
        parsed = None
        try:
            resp = model.generate_content(prompt)
            raw_text = (resp.text or "")
            logger.info("LLM raw_text_len=%d", len(raw_text))
            logger.debug("LLM raw_text_head=%s", raw_text[:400].replace("\n", " "))
            parsed = try_parse_json(raw_text)
        except Exception as e:
            logger.error("LLM call failed: %s", e)

        cluster_name = context_json.get("cluster_name", "")
        if isinstance(parsed, dict):
            coerced = coerce_to_schema_with_cluster(parsed, customer_id=customer_id, cluster_name=cluster_name)
            if debug:
                return {
                    "result": coerced,
                    "debug": {
                        "cust_rows": int(len(cust_df)),
                        "known_total_revenue": known_total_revenue,
                        "all_rows": int(nlines),
                        "nonnull_created": nonnull_created,
                        "nonnull_billed": nonnull_billed,
                        "n_items": n_items,
                        "n_prices": n_prices,
                        "raw_text_head": raw_text[:400],
                        "parsed": True,
                    },
                }
            return coerced

        # Fallback if parsing failed
        fallback = {
            "customer": customer_id,
            "cluster": cluster_name,
            "churn": "",
            "churn_analysis": "",
            "retention_strategies": "",
            "Retention_offers": "",
            "Purchase_details": "",
            "revenue_by_year": {},
            "revenue_by_quarter": {},
            "trend_of_sales": "",
            "product_combination": "",
            "best_price_by_material": [],
            "observation": "",
            "recommendation": "",
        }
        if debug:
            return {
                "result": fallback,
                "debug": {
                    "cust_rows": int(len(cust_df)),
                    "known_total_revenue": known_total_revenue,
                    "all_rows": int(nlines),
                    "nonnull_created": nonnull_created,
                    "nonnull_billed": nonnull_billed,
                    "n_items": n_items,
                    "n_prices": n_prices,
                    "raw_text_head": raw_text[:400],
                    "parsed": False,
                },
            }
        return fallback

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in get_customer_insights for customer %s: %s", customer_id, e)
        raise HTTPException(status_code=500, detail="Internal server error")
