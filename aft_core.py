# aft_core.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def calcola_drive_index(df: pd.DataFrame) -> pd.DataFrame:
    """ Calcola il 'Drive Index' (0-1). Scala rigidità corretta 5N-35N. """
    def score_plate(val):
        val = str(val).lower()
        if 'carbon' in val or 'carbitex' in val: return 1.0
        if 'fiberglass' in val: return 0.7
        if any(x in val for x in ['plastic', 'tpu', 'nylon']): return 0.5
        return 0.1 

    S_Plate = df['piastra'].apply(score_plate)

    def score_rocker(val):
        if pd.isna(val) or str(val) in ['nan', '#N/D']: return 0.0
        try:
            clean_val = str(val).replace(',', '.')
            parts = clean_val.split('x')
            if len(parts) >= 1:
                h = float(parts[0])
                return min(h / 10.0, 1.0)
            return 0.0
        except:
            return 0.0
            
    S_Rocker = df['rocker'].apply(score_rocker)
    S_Foam = df['EnergyIndex'] 
    
    flex_val = pd.to_numeric(df['rigidezza_flex'], errors='coerce').fillna(15)
    S_Stiffness = (flex_val - 5) / 30.0 
    S_Stiffness = S_Stiffness.clip(0, 1)

    Mechanical_Drive = S_Plate * S_Rocker * S_Stiffness
    df['DriveIndex'] = (0.6 * Mechanical_Drive) + (0.4 * S_Foam)
    
    return df

def calcola_indici(df: pd.DataFrame) -> pd.DataFrame:
    """ Calcola gli indici biomeccanici. """
    def safe_minmax(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        denom = max(xmax - xmin, np.finfo(float).eps)
        return (x - xmin) / denom

    # 1. Shock & Energy
    w_heel = 0.4
    w_mid = 0.6
    S_heel = safe_minmax(df["shock_abs_tallone"])
    S_mid  = safe_minmax(df["shock_abs_mesopiede"])
    ER_h   = safe_minmax(df["energy_ret_tallone"])
    ER_m   = safe_minmax(df["energy_ret_mesopiede"])

    df["ShockIndex"]  = (w_heel * S_heel + w_mid * S_mid) / (w_heel + w_mid)
    df["EnergyIndex"] = (w_heel * ER_h   + w_mid * ER_m)  / (w_heel + w_mid)

    # 2. Flex Index (Range 5-40N)
    Flex = pd.to_numeric(df["rigidezza_flex"], errors='coerce').fillna(15)
    FlexIndex = np.zeros(len(df))
    passi = df["passo"].astype(str).str.lower().to_list()

    for i, tipo in enumerate(passi):
        val_N = Flex[i]
        if "race" in tipo:
            FlexIndex[i] = 1 / (1 + np.exp(-(val_N - 18) / 4)) 
        else:
            FlexIndex[i] = np.exp(-((val_N - 12) ** 2) / (2 * 5 ** 2))
            
    df["FlexIndex"] = FlexIndex

    # 3. Weight Index
    W = df["peso"].astype(float).to_numpy()
    W_ref = 180.0 
    k = 0.005 
    WeightIndex = np.exp(-k * (W - W_ref))
    WeightIndex = np.clip(WeightIndex, 0, 1)
    df["WeightIndex"] = WeightIndex

    # 4. StackFactor
    stack = df["altezza_tallone"].astype(float).to_numpy()
    StabilityMod = np.ones(len(df))
    mask_hi = stack > 40
    if np.any(mask_hi):
        StabilityMod[mask_hi] = np.maximum(0.85, 1.0 - 0.015 * (stack[mask_hi] - 40.0))
    
    df["StackFactor"] = StabilityMod
    df["EnergyIndex"] = df["EnergyIndex"] * StabilityMod

    # 5. Drive Index
    df = calcola_drive_index(df)
    return df

def calcola_MPIB(df: pd.DataFrame) -> pd.DataFrame:
    w_shock  = 0.20
    w_energy = 0.30
    w_flex   = 0.20
    w_weight = 0.30

    df["MPI_B"] = (
        w_shock  * df["ShockIndex"] +
        w_energy * df["EnergyIndex"] +
        w_flex   * df["FlexIndex"] +
        w_weight * df["WeightIndex"]
    )
    df = df.sort_values(by="MPI_B", ascending=False).reset_index(drop=True)
    return df

def esegui_clustering(df: pd.DataFrame):
    def livello_index(val: float) -> str:
        if val < 0.33: return "Basso"
        elif val < 0.66: return "Medio"
        else: return "Alto"

    def descrizione_cluster_simplificata(row: pd.Series) -> str:
        shock  = livello_index(row["Shock"])
        energy = livello_index(row["Energy"])
        flex   = livello_index(row["Flex"])
        weight = livello_index(row["Weight"])
        drive  = livello_index(row["Drive"])
        return (f"Shock: {shock} | Drive: {drive} | "
                f"Flex: {flex} | Peso: {weight}")

    rng = 42
    np.random.seed(rng)
    X = df[["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "DriveIndex"]].to_numpy()
    labels_cols = ["Shock", "Energy", "Flex", "Weight", "Drive"]

    K_values = np.arange(2, 11)
    SSE = []; silh_mean = []
    for k in K_values:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=rng)
        idx_tmp = kmeans.fit_predict(X)
        SSE.append(kmeans.inertia_)
        silh_mean.append(np.mean(silhouette_samples(X, idx_tmp)))

    SSE = np.array(SSE); silh_mean = np.array(silh_mean)
    logSSE = np.log(SSE)
    if len(logSSE) > 2:
        d2 = np.gradient(np.gradient(logSSE))
        k_elbow = K_values[np.argmin(d2)]
    else: k_elbow = 3
    k_silh = K_values[np.argmax(silh_mean)]
    
    k_opt = int(round(0.7 * k_silh + 0.3 * k_elbow))
    k_opt = max(2, min(k_opt, 7))

    kmeans_final = KMeans(n_clusters=k_opt, n_init=50, random_state=rng)
    idx = kmeans_final.fit_predict(X)
    C = kmeans_final.cluster_centers_

    df["Cluster"] = idx + 1
    cluster_summary = pd.DataFrame(C, columns=labels_cols)
    cluster_summary["Cluster"] = np.arange(1, k_opt + 1)
    cluster_summary["Descrizione"] = cluster_summary.apply(descrizione_cluster_simplificata, axis=1)
    descr_map = dict(zip(cluster_summary["Cluster"], cluster_summary["Descrizione"]))
    df["ClusterDescrizione"] = df["Cluster"].map(descr_map)

    return df, cluster_summary

def trova_scarpe_simili(df, target_label, metrics_cols, weights=None, n_simili=3):
    """
    Trova le n scarpe più simili. Se weights è fornito, calcola la Distanza Euclidea Pesata.
    """
    try:
        target_vector = df.loc[df['label'] == target_label, metrics_cols].astype(float).values[0]
        df_calc = df.copy()
        
        vectors = df_calc[metrics_cols].astype(float).values
        
        if weights is not None:
            # Calcolo Distanza Euclidea PESATA
            # d(x,y) = sqrt( sum( w_i * (x_i - y_i)^2 ) )
            # Più alto è il peso w_i, più una differenza su quell'asse "allontana" la scarpa.
            w = np.array(weights)
            diff_sq = (vectors - target_vector) ** 2
            distances = np.sqrt((diff_sq * w).sum(axis=1))
        else:
            # Distanza Euclidea Standard
            distances = np.linalg.norm(vectors - target_vector, axis=1)
            
        df_calc['distanza_similitudine'] = distances
        
        simili = df_calc[df_calc['label'] != target_label].sort_values('distanza_similitudine').head(n_simili)
        return simili
    except Exception:
        return pd.DataFrame()
