"""
Stage 3 helpers: grid features (terrain, population, coast), spatial stats, spatial-CV models.

Grid = 1 km British National Grid cells (gx, gy = floor(easting/1000), floor(northing/1000)),
the same cells built in Stage 1.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GRID_M = 1000

# ------------------------------------------------------------------ helpers
def _to_grid(lon, lat):
    from pyproj import Transformer
    tf = Transformer.from_crs(4326, 27700, always_xy=True)
    x, y = tf.transform(lon, lat)
    return (np.floor(x / GRID_M)).astype("int64"), (np.floor(y / GRID_M)).astype("int64")


def cell_centroids_lonlat(gx, gy):
    from pyproj import Transformer
    tf = Transformer.from_crs(27700, 4326, always_xy=True)
    return tf.transform((np.asarray(gx) + .5) * GRID_M, (np.asarray(gy) + .5) * GRID_M)


def fetch(url, dest: Path, timeout=120):
    """Download once; return path, or None on 404 (e.g. all-sea DEM tile)."""
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    if dest.with_suffix(".missing").exists():
        return None
    r = requests.get(url, timeout=timeout, stream=True)
    if r.status_code in (403, 404):
        dest.with_suffix(".missing").touch()
        return None
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    with open(tmp, "wb") as f:
        for b in r.iter_content(1 << 20):
            f.write(b)
    tmp.rename(dest)
    return dest


def _raster_pixels(path, block=512):
    """Yield (lon, lat, value) arrays for a north-up lat/lon GeoTIFF, in row blocks."""
    import rasterio
    with rasterio.open(path) as src:
        T, nod, H, W = src.transform, src.nodata, src.height, src.width
        lon_c = T.c + (np.arange(W) + .5) * T.a
        for r0 in range(0, H, block):
            r1 = min(H, r0 + block)
            a = src.read(1, window=rasterio.windows.Window(0, r0, W, r1 - r0)).astype("float64")
            lat_c = T.f + (np.arange(r0, r1) + .5) * T.e
            lon, lat = np.meshgrid(lon_c, lat_c)
            v = a.ravel(); ok = np.isfinite(v)
            if nod is not None and np.isfinite(nod):
                ok &= v != nod
            yield lon.ravel()[ok], lat.ravel()[ok], v[ok]


# ------------------------------------------------------------------ terrain (Copernicus GLO-90)
DEM_URL = "https://copernicus-dem-90m.s3.amazonaws.com/Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/Copernicus_DSM_COG_30_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"


def dem_tiles_for(gx, gy, buffer_km=5):
    """1°x1° tiles (lower-left lat, lon) needed to cover the cells + buffer."""
    tiles = set()
    for dx in (-buffer_km, 0, buffer_km):
        for dy in (-buffer_km, 0, buffer_km):
            lon, lat = cell_centroids_lonlat(np.asarray(gx) + dx, np.asarray(gy) + dy)
            tiles |= set(zip(np.floor(lat).astype(int), np.floor(lon).astype(int)))
    return sorted(tiles)


def dem_url(lat, lon):
    return DEM_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))


def terrain_grid(tile_paths) -> pd.DataFrame:
    """Per 1 km cell: mean, std (ruggedness), min, max elevation from all tiles."""
    acc = []
    for p in tile_paths:
        for lon, lat, v in _raster_pixels(p):
            gx, gy = _to_grid(lon, lat)
            d = pd.DataFrame({"gx": gx, "gy": gy, "v": v, "v2": v * v})
            acc.append(d.groupby(["gx", "gy"]).agg(n=("v", "size"), s=("v", "sum"), s2=("v2", "sum"),
                                                     mn=("v", "min"), mx=("v", "max")).reset_index())
    t = pd.concat(acc).groupby(["gx", "gy"]).agg(n=("n", "sum"), s=("s", "sum"), s2=("s2", "sum"),
                                                  mn=("mn", "min"), mx=("mx", "max")).reset_index()
    t["elev_mean"] = t.s / t.n
    t["elev_std"] = np.sqrt(np.maximum(t.s2 / t.n - t.elev_mean ** 2, 0))
    t["elev_range"] = t.mx - t.mn
    t["elev_max"] = t.mx
    return t[["gx", "gy", "elev_mean", "elev_std", "elev_range", "elev_max"]]

# ------------------------------------------------------------------ population (WorldPop 1 km)
WORLDPOP_URLS = [
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km/2020/GBR/gbr_ppp_2020_1km_Aggregated.tif",
    "https://data.worldpop.org/GIS/Population/Global_2000_2020_Constrained/2020/BSGM/GBR/gbr_ppp_2020_constrained.tif",
]


def population_grid(path) -> pd.DataFrame:
    acc = []
    for lon, lat, v in _raster_pixels(path):
        v = np.where(v < 0, 0, v)
        gx, gy = _to_grid(lon, lat)
        acc.append(pd.DataFrame({"gx": gx, "gy": gy, "pop": v}).groupby(["gx", "gy"], as_index=False)["pop"].sum())
    return pd.concat(acc).groupby(["gx", "gy"], as_index=False)["pop"].sum()

# ------------------------------------------------------------------ neighbourhood features
def neighbourhood(df: pd.DataFrame, col: str, radius: int, how="mean") -> pd.Series:
    """Aggregate `col` over the (2r+1)^2 window around each cell.
    how='sum'  : sum (missing cells count as 0)      e.g. population within ~5 km
    how='mean' : mean of the cells that exist (ignores sea / missing cells)"""
    from scipy.ndimage import uniform_filter
    x0, y0 = df.gx.min() - radius, df.gy.min() - radius
    W, H = df.gx.max() - x0 + radius + 1, df.gy.max() - y0 + radius + 1
    iy, ix = (df.gy - y0).to_numpy(), (df.gx - x0).to_numpy()
    A = np.zeros((H, W)); P = np.zeros((H, W))
    v = df[col].to_numpy(dtype=float); ok = np.isfinite(v)
    A[iy[ok], ix[ok]] = v[ok]; P[iy[ok], ix[ok]] = 1
    k = (2 * radius + 1) ** 2
    S = uniform_filter(A, size=2 * radius + 1, mode="constant") * k
    if how == "sum":
        return pd.Series(S[iy, ix], index=df.index)
    C = uniform_filter(P, size=2 * radius + 1, mode="constant") * k
    return pd.Series(S[iy, ix] / np.maximum(C[iy, ix], 1e-9), index=df.index)


def coast_distance_km(gx, gy, lad) -> np.ndarray:
    import geopandas as gpd
    land = lad.to_crs(27700).geometry.union_all() if hasattr(lad.geometry, "union_all") else lad.to_crs(27700).unary_union
    coast = land.boundary
    pts = gpd.GeoSeries(gpd.points_from_xy((np.asarray(gx) + .5) * GRID_M, (np.asarray(gy) + .5) * GRID_M), crs=27700)
    return pts.distance(coast).to_numpy() / 1000

# ------------------------------------------------------------------ spatial statistics
def morans_i(df, col, k=8, permutations=999, seed=0):
    from libpysal.weights import KNN
    from esda.moran import Moran
    W = KNN.from_array(np.c_[df.gx, df.gy], k=k); W.transform = "r"
    np.random.seed(seed)
    m = Moran(df[col].to_numpy(), W, permutations=permutations)
    return m, W


def lisa_clusters(df, col, W, permutations=999, alpha=0.05, seed=0):
    from esda.moran import Moran_Local
    np.random.seed(seed)
    lm = Moran_Local(df[col].to_numpy(), W, permutations=permutations)
    lab = np.array(["Not significant"] * len(df), dtype=object)
    sig = lm.p_sim < alpha
    names = {1: "High-High (hotspot)", 2: "Low-High", 3: "Low-Low (coldspot)", 4: "High-Low"}
    for q, nm in names.items():
        lab[sig & (lm.q == q)] = nm
    return pd.Series(lab, index=df.index), lm

# ------------------------------------------------------------------ modelling
def spatial_blocks(gx, gy, block_km=50):
    return (np.asarray(gx) // block_km).astype(int) * 10_000 + (np.asarray(gy) // block_km).astype(int)


def cv_evaluate(model_factory, X, y, groups=None, n_splits=5, seed=0, weight=None):
    """Out-of-fold predictions + metrics. groups=None -> random KFold (for comparison)."""
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    oof = np.full(len(y), np.nan)
    if groups is None:
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(X)
    else:
        splits = GroupKFold(n_splits).split(X, y, groups)
    fold_r2 = []
    for tr, te in splits:
        m = model_factory()
        if weight is not None:
            m.fit(X.iloc[tr], y.iloc[tr], sample_weight=weight.iloc[tr])
        else:
            m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X.iloc[te])
        fold_r2.append(r2_score(y.iloc[te], oof[te]))
    return {"R2": r2_score(y, oof), "MAE": mean_absolute_error(y, oof),
            "RMSE": math.sqrt(mean_squared_error(y, oof)), "R2_fold_sd": float(np.std(fold_r2))}, oof
