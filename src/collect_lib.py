"""
Data-collection helpers for the Vodafone coverage-gap project (v2: no API key needed).

Sources
-------
1. Ofcom drive-test measurements (4G, open data, OGL): real RSRP per operator along roads.
2. OpenCelliD (CC-BY-SA 4.0): crowdsourced cell locations per operator (MCC 234).
3. ONS LAD boundaries (Dec 2024): spatial join to local authorities.
4. Ofcom Connected Nations 2025 LA file: validation (operator-count based).

Ofcom 2023 drive-test files are "wide": one row per measurement point, with columns per
operator suffix (rsrp_top1_ee, rsrp_top1_o2, rsrp_top1_3uk, rsrp_top1_vf, ...).

Design: drive-test points -> 1 km British National Grid cells (per operator),
then cell centroids -> LA. Every Ofcom vehicle measures all four networks at the
same place and time, so comparing operators within a grid cell is a fair
head-to-head comparison.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

UK_MNC = {  # MCC 234
    2: "O2", 10: "O2", 11: "O2",
    15: "Vodafone",
    20: "Three",
    30: "EE", 31: "EE", 32: "EE", 33: "EE", 34: "EE",
}
OPS = ["EE", "O2", "Three", "Vodafone"]
RSRP_OUTDOOR_DBM = -105   # Ofcom 4G outdoor threshold
RSRP_INDOOR_DBM = -95     # Ofcom 4G indoor / in-car threshold
GRID_M = 1000

# ---------------------------------------------------------------- downloads
def scrape_links(page_url: str) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    html = requests.get(page_url, timeout=60, headers={"User-Agent": "Mozilla/5.0"}).text
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for a in soup.find_all("a", href=True):
        href = requests.compat.urljoin(page_url, a["href"])
        if re.search(r"\.(zip|csv|gz|xlsx|pdf)(\?|$)", href, re.I) or "siteassets" in href:
            rows.append({"text": " ".join(a.get_text(" ").split()), "url": href})
    return pd.DataFrame(rows).drop_duplicates("url")


def download(url: str, dest: Path, chunk=1 << 22) -> Path:
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"Already downloaded: {dest.name}")
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        done = 0
        with open(tmp, "wb") as f:
            for b in r.iter_content(chunk):
                f.write(b); done += len(b)
                if total and done % (chunk * 64) < chunk:
                    print(f"  {done/1e9:.2f}/{total/1e9:.2f} GB")
        tmp.rename(dest)
    print(f"Saved {dest} ({dest.stat().st_size/1e6:.1f} MB)")
    return dest


def csv_members(path: Path) -> list[str]:
    """CSV files inside a zip (or the path itself if it is a CSV)."""
    if zipfile.is_zipfile(path):
        return [m for m in zipfile.ZipFile(path).namelist() if m.lower().endswith(".csv")]
    return [str(path)]


def open_csv(path: Path, member: str):
    if zipfile.is_zipfile(path):
        return zipfile.ZipFile(path).open(member)
    return open(member, "rb")

# ---------------------------------------------------------------- drive test
def _find(cols, *pats, exclude=()):
    for p in pats:
        for c in cols:
            lc = c.lower().strip()
            if re.fullmatch(p, lc) and not any(re.search(e, lc) for e in exclude):
                return c
    return None


# Real 2023 files: one row per measurement, operator-suffixed columns, e.g.
# latitude, longitude, month_year, rsrp_top1_3uk, rsrp_top1_ee, rsrp_top1_o2, rsrp_top1_<vodafone>...
# top1 = strongest cell. Blank RSRP for an operator = no cell detected there -> counted as NOT covered.

def _suffix_to_op(s):
    s = s.lower()
    if s in ("3uk", "h3g", "three", "3", "h3"): return "Three"
    if s in ("ee", "eeuk"): return "EE"
    if s in ("o2", "o2uk", "tf", "vmo2"): return "O2"
    if s.startswith(("vf", "vod", "vo")): return "Vodafone"
    return None


def detect_wide_cols(cols):
    lat = _find(cols, r"lat(itude)?"); lon = _find(cols, r"lon(g|gitude)?")
    date = _find(cols, r"month_year", r"date", r"month.*")
    ops = {}
    for c in cols:
        m = re.fullmatch(r"rsrp_top1_(.+)", c.lower().strip())
        if m and _suffix_to_op(m.group(1)):
            ops[_suffix_to_op(m.group(1))] = c
    if not (lat and lon and ops):
        raise ValueError(f"Wide format not recognised. Columns: {list(cols)[:40]}")
    unknown = [c for c in cols if re.fullmatch(r"rsrp_top1_(.+)", c.lower()) and c not in ops.values()]
    return {"lat": lat, "lon": lon, "date": date, "ops": ops, "unmapped_rsrp_cols": unknown}


def aggregate_drive_test(path, chunksize=1_000_000, sample_frac=0.01, seed=0):
    from pyproj import Transformer
    tf = Transformer.from_crs(4326, 27700, always_xy=True)
    parts, samples = [], []
    for member in csv_members(path):
        with open_csv(path, member) as fh:
            head = pd.read_csv(fh, nrows=0).columns
        if not any(re.fullmatch(r"rsrp_top1_.+", c.lower()) for c in head):
            raise ValueError(f"{member}: not wide format; columns {list(head)[:20]}")
        cm = detect_wide_cols(head)
        print(f"{Path(member).name}: operators {cm['ops']}  unmapped={cm['unmapped_rsrp_cols']}")
        usecols = [cm["lat"], cm["lon"]] + ([cm["date"]] if cm["date"] else []) + list(cm["ops"].values())
        with open_csv(path, member) as fh:
            for i, ch in enumerate(pd.read_csv(fh, usecols=usecols, chunksize=chunksize, low_memory=False)):
                ch = ch.rename(columns={cm["lat"]: "lat", cm["lon"]: "lon"})
                ch["lat"] = pd.to_numeric(ch["lat"], errors="coerce"); ch["lon"] = pd.to_numeric(ch["lon"], errors="coerce")
                ch = ch[ch["lat"].between(49.8, 61) & ch["lon"].between(-8.7, 2)]
                R = pd.DataFrame({op: pd.to_numeric(ch[c], errors="coerce") for op, c in cm["ops"].items()}, index=ch.index)
                R = R.where((R >= -160) & (R <= -30))
                keep = R.notna().any(axis=1)          # vehicle measured something at this point
                ch, R = ch[keep], R[keep]
                x, y = tf.transform(ch["lon"].to_numpy(), ch["lat"].to_numpy())
                gx = (x // GRID_M).astype("int32"); gy = (y // GRID_M).astype("int32")
                for op in R.columns:
                    r = R[op].to_numpy()
                    d = pd.DataFrame({"gx": gx, "gy": gy, "operator": op, "n": 1,
                                      "n_sig": ~np.isnan(r), "rsrp_sum": np.nan_to_num(r),
                                      "cov_out": np.nan_to_num(r, nan=-999) >= RSRP_OUTDOOR_DBM,
                                      "cov_in": np.nan_to_num(r, nan=-999) >= RSRP_INDOOR_DBM})
                    parts.append(d.groupby(["gx", "gy", "operator"], as_index=False)[["n", "n_sig", "rsrp_sum", "cov_out", "cov_in"]].sum())
                if sample_frac:
                    s = ch.join(R.add_prefix("rsrp_")).sample(frac=sample_frac, random_state=seed + i)
                    samples.append(s.drop(columns=list(cm["ops"].values()), errors="ignore"))
                print(f"  chunk {i}: {len(ch):,} rows")
    grid = pd.concat(parts).groupby(["gx", "gy", "operator"], as_index=False).sum()
    grid["rsrp_mean"] = grid["rsrp_sum"] / grid["n_sig"].replace(0, np.nan)
    grid["cov_out_share"] = grid["cov_out"] / grid["n"]
    grid["cov_in_share"] = grid["cov_in"] / grid["n"]
    grid["signal_seen_share"] = grid["n_sig"] / grid["n"]
    pts = pd.concat(samples, ignore_index=True) if samples else None
    return grid, pts


# ---------------------------------------------------------------- boundaries
LAD_SERVICES = [
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Local_Authority_Districts_December_2024_Boundaries_UK_BGC/FeatureServer/0",
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/LAD_DEC_24_UK_BSC/FeatureServer/0",
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/Local_Authority_Districts_December_2024_Boundaries_UK_BSC/FeatureServer/0",
]


def load_lad_boundaries(cache: Path):
    import geopandas as gpd
    cache = Path(cache)
    if cache.exists():
        return gpd.read_file(cache)
    last_err = None
    for svc in LAD_SERVICES:
        try:
            feats, offset = [], 0
            while True:
                r = requests.get(f"{svc}/query", params={
                    "where": "1=1", "outFields": "*", "outSR": 4326, "f": "geojson",
                    "resultOffset": offset, "resultRecordCount": 200}, timeout=120)
                r.raise_for_status()
                js = r.json()
                if "features" not in js:
                    raise RuntimeError(str(js)[:300])
                feats += js["features"]
                if len(js["features"]) < 200:
                    break
                offset += 200
            gdf = gpd.GeoDataFrame.from_features(feats, crs=4326)
            cd = next(c for c in gdf.columns if re.fullmatch(r"lad\d{2}cd", c.lower()))
            nm = next(c for c in gdf.columns if re.fullmatch(r"lad\d{2}nm", c.lower()))
            gdf = gdf.rename(columns={cd: "lad_cd", nm: "lad_nm"})[["lad_cd", "lad_nm", "geometry"]]
            gdf.to_file(cache, driver="GeoJSON")
            print(f"LAD boundaries: {len(gdf)} from {svc.split('/services/')[1]}")
            return gdf
        except Exception as e:  # try next service
            last_err = e
            print("  failed:", svc.split("/services/")[1], "->", str(e)[:120])
    raise RuntimeError(f"All LAD services failed: {last_err}. Download the GeoJSON manually from geoportal.statistics.gov.uk to {cache}")


def grid_to_lad(grid: pd.DataFrame, lad):
    import geopandas as gpd
    cells = grid[["gx", "gy"]].drop_duplicates()
    pts = gpd.GeoDataFrame(cells, geometry=gpd.points_from_xy((cells.gx + .5) * GRID_M, (cells.gy + .5) * GRID_M), crs=27700)
    j = gpd.sjoin(pts, lad.to_crs(27700)[["lad_cd", "lad_nm", "geometry"]], how="left", predicate="within")
    # cells whose centroid is offshore -> nearest LA
    miss = j["lad_cd"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss.values], lad.to_crs(27700)[["lad_cd", "lad_nm", "geometry"]], how="left")
        near = near[~near.index.duplicated()]
        j.loc[near.index, ["lad_cd", "lad_nm"]] = near[["lad_cd", "lad_nm"]].values
    return grid.merge(j[["gx", "gy", "lad_cd", "lad_nm"]].drop_duplicates(["gx", "gy"]), on=["gx", "gy"], how="left")

# ---------------------------------------------------------------- gap metrics
def grid_gap_table(grid: pd.DataFrame, min_n=20) -> pd.DataFrame:
    """Wide grid table: one row per 1 km cell, operator coverage side by side, Vodafone gap."""
    g = grid[grid["n"] >= min_n]
    wide = g.pivot_table(index=["gx", "gy", "lad_cd", "lad_nm"], columns="operator",
                         values=["cov_out_share", "cov_in_share", "rsrp_mean", "n"])
    wide.columns = [f"{v}_{op}" for v, op in wide.columns]
    wide = wide.reset_index()
    need = [f"cov_out_share_{o}" for o in ("Vodafone", "EE", "O2")]
    wide = wide.dropna(subset=need)                    # head-to-head cells only
    for m in ("cov_out_share", "cov_in_share", "rsrp_mean"):
        wide[f"best_rival_{m}"] = wide[[f"{m}_EE", f"{m}_O2"]].max(axis=1)
        wide[f"vo_gap_{m}"] = wide[f"best_rival_{m}"] - wide[f"{m}_Vodafone"]   # >0: Vodafone behind
        if f"{m}_Three" in wide:
            wide[f"{m}_VodafoneThree"] = wide[[f"{m}_Vodafone", f"{m}_Three"]].max(axis=1)
    wide["vo_notspot_out"] = 1 - wide["cov_out_share_Vodafone"]
    wide["vo_notspot_in"] = 1 - wide["cov_in_share_Vodafone"]
    return wide


def lad_gap_table(wide: pd.DataFrame, min_cells=5) -> pd.DataFrame:
    metrics = [c for c in wide.columns if c.startswith(("cov_", "rsrp_", "best_rival_", "vo_"))]
    la = wide.groupby(["lad_cd", "lad_nm"])[metrics].mean()
    la["n_cells"] = wide.groupby(["lad_cd", "lad_nm"]).size()
    la["share_cells_vo_behind"] = wide.assign(b=wide["vo_gap_cov_out_share"] > 0.1).groupby(["lad_cd", "lad_nm"])["b"].mean()
    la = la[la["n_cells"] >= min_cells].reset_index()
    return la.sort_values("vo_gap_cov_out_share", ascending=False)

# ---------------------------------------------------------------- OpenCelliD
OCID_COLS = ["radio", "mcc", "net", "area", "cell", "unit", "lon", "lat",
             "range", "samples", "changeable", "created", "updated", "averageSignal"]


def load_opencellid(path: Path) -> pd.DataFrame:
    peek = pd.read_csv(path, nrows=1, header=None)
    has_header = str(peek.iloc[0, 0]).lower() == "radio"
    df = pd.read_csv(path, header=0 if has_header else None, names=None if has_header else OCID_COLS, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    df = df[(df["mcc"] == 234)].copy()
    df["operator"] = df["net"].map(UK_MNC)
    df = df.dropna(subset=["operator", "lat", "lon"])
    df["updated_dt"] = pd.to_datetime(df["updated"], unit="s", errors="coerce")
    return df


def opencellid_by_lad(cells: pd.DataFrame, lad, radios=("LTE", "NR"), site_round=3) -> pd.DataFrame:
    """Cells and approximate sites (cells snapped to ~100 m) per LA x operator."""
    import geopandas as gpd
    c = cells[cells["radio"].isin(radios)].copy()
    c["site"] = c["lat"].round(site_round).astype(str) + "_" + c["lon"].round(site_round).astype(str)
    g = gpd.GeoDataFrame(c, geometry=gpd.points_from_xy(c.lon, c.lat), crs=4326)
    j = gpd.sjoin(g, lad[["lad_cd", "geometry"]], how="inner", predicate="within")
    out = j.groupby(["lad_cd", "operator"]).agg(cells=("cell", "size"), sites=("site", "nunique")).reset_index()
    wide = out.pivot(index="lad_cd", columns="operator", values=["cells", "sites"]).fillna(0)
    wide.columns = [f"ocid_{v}_{op}" for v, op in wide.columns]
    area = lad.to_crs(27700).set_index("lad_cd").area / 1e6
    wide = wide.join(area.rename("area_km2"), how="right").fillna(0)
    for op in OPS:
        if f"ocid_sites_{op}" in wide:
            wide[f"ocid_site_density_{op}"] = wide[f"ocid_sites_{op}"] / wide["area_km2"]
    if {"ocid_sites_EE", "ocid_sites_O2", "ocid_sites_Vodafone"} <= set(wide.columns):
        wide["ocid_vo_site_share"] = wide["ocid_sites_Vodafone"] / wide[
            ["ocid_sites_EE", "ocid_sites_O2", "ocid_sites_Vodafone"]].sum(axis=1).replace(0, np.nan)
    return wide.reset_index()
