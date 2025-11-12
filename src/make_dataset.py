# src/make_dataset.py
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from math import radians, sin, cos, sqrt, asin
import os

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371.0 * c

def build(input_csv="data/sample_water_data.csv", output_csv="data/processed/dataset_ready.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    # parse time
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if time_cols:
        df['time_utc'] = pd.to_datetime(df[time_cols[0]], errors='coerce')
    else:
        df['time_utc'] = pd.NaT
    # rename minimal columns if needed
    if 'latitude' in df.columns and 'lat' not in df.columns:
        df = df.rename(columns={'latitude':'lat','longitude':'lon'})
    # ensure numeric
    for col in ['lat','lon','sst','sss']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['lat','lon']).reset_index(drop=True)
    # sss gradient
    if 'sss' in df.columns and len(df) >= 3:
        coords = df[['lat','lon']].values
        try:
            nbrs = NearestNeighbors(n_neighbors=min(6,len(df))).fit(coords)
            distances, indices = nbrs.kneighbors(coords)
            sss_vals = df['sss'].values
            mean_grad = []
            for i, inds in enumerate(indices):
                diffs = []
                for j in inds[1:]:
                    if np.isnan(sss_vals[i]) or np.isnan(sss_vals[j]):
                        continue
                    dkm = haversine(df.loc[i,'lon'], df.loc[i,'lat'], df.loc[j,'lon'], df.loc[j,'lat'])
                    if dkm == 0: continue
                    diffs.append(abs(sss_vals[i]-sss_vals[j]) / dkm)
                mean_grad.append(np.nanmean(diffs) if diffs else np.nan)
            df['sss_grad_per_km'] = mean_grad
        except Exception:
            df['sss_grad_per_km'] = np.nan
    else:
        df['sss_grad_per_km'] = np.nan
    # temporal features
    df['dayofyear'] = df['time_utc'].dt.dayofyear.fillna(0).astype(int)
    df['hour'] = df['time_utc'].dt.hour.fillna(0).astype(int)
    # fill small missings by median
    for c in ['sst','sss','sss_grad_per_km']:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
    df.to_csv(output_csv, index=False)
    print(f"Saved processed dataset to {output_csv} (n={len(df)})")

if __name__ == "__main__":
    build()
