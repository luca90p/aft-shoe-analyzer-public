# aft_core.py (Versione Corretta)
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# --- FUNZIONI DI SUPPORTO PER NUOVI INDICI ---

# aft_core.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# --- NUOVA FUNZIONE STABILITÀ ---
def calcola_grip_score(df: pd.DataFrame) -> pd.Series:
    """
    Calcola un punteggio di Grip normalizzato (0-1).
    Basato su 'test_trazione'. Range tipico DB: 0.25 - 0.85.
    """
    grip = pd.to_numeric(df['test_trazione'], errors='coerce')
    # Normalizziamo tra 0.3 (scivolosa) e 0.9 (colla)
    # Valori sotto 0.3 diventano 0, sopra 0.9 diventano 1
    S_Grip = (grip - 0.3) / (0.9 - 0.3)
    return S_Grip.clip(0, 1).fillna(0.5)
def calcola_stability_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola indice Stabilità (0-1).
    Include ora il GRIP come fattore di sicurezza attiva.
    """
    def safe_norm_fill(s):
        s = pd.to_numeric(s, errors='coerce')
        s = s.fillna(s.mean())
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    # 1. Rigidità Strutturale
    torsion = pd.to_numeric(df['rigidezza_torsionale'], errors='coerce').fillna(3)
    heel_stiff = pd.to_numeric(df['rigidezza_tallone'], errors='coerce').fillna(3)
    
    S_Torsion = ((torsion - 1) / 4).clip(0, 1)
    S_HeelStruct = ((heel_stiff - 1) / 4).clip(0, 1)

    # 2. Geometria Base
    S_Width_Mid = safe_norm_fill(df['larghezza_suola_mesop'])
    S_Width_Heel = safe_norm_fill(df['larghezza_suola_tallone'])

    # 3. Stack Height (Penalità)
    S_LowStack = 1.0 - safe_norm_fill(df['altezza_tallone'])
    
    # 4. Grip (Nuovo)
    S_Grip = calcola_grip_score(df)

    # 5. Calcolo Ponderato (Grip pesa il 10%)
    df['StabilityIndex'] = (
        0.25 * S_Torsion +
        0.20 * S_Width_Mid +
        0.15 * S_Width_Heel +
        0.15 * S_HeelStruct +
        0.15 * S_LowStack +
        0.10 * S_Grip 
    )
    return df

def calcola_durability_index(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Calcola un indice di durabilità (0-1).
    """
    # 1. Durabilità Suola (Wear Ratio)
    wear_depth = pd.to_numeric(df['resistenza_suola'], errors='coerce')
    thickness = pd.to_numeric(df['spessore_suola'], errors='coerce').replace(0, 0.1)
    
    wear_ratio = wear_depth / thickness
    
    # Normalizzazione Inversa (0 danno = 1.0 score)
    S_Suola = 1.0 - (wear_ratio / 0.8)
    S_Suola = S_Suola.clip(0, 1)

    # 2. Durabilità Tomaia e Tallone (Scala 1-5)
    def norm_resistance_score(col_name):
        val = pd.to_numeric(df[col_name], errors='coerce').fillna(3)
        # 1 -> 0.0, 5 -> 1.0
        norm = (val - 1) / 4.0
        return norm.clip(0, 1)
        
    S_Tomaia = norm_resistance_score('resistenza_tomaia')
    S_Tallone = norm_resistance_score('resistenza_tendach')

    # Peso ponderato
    df['DurabilityIndex'] = (0.60 * S_Suola) + (0.25 * S_Tomaia) + (0.15 * S_Tallone)
    
    return df

def calcola_fit_class(df: pd.DataFrame) -> pd.DataFrame:
    """ Classifica la calzata (Stretta/Standard/Ampia). """
    width = pd.to_numeric(df['larghezza_dita'], errors='coerce')
    mean_w = width.mean()
    std_w = width.std()
    
    def classify(w):
        if pd.isna(w): return "N/D"
        if w < (mean_w - 0.5 * std_w): return "Stretta 🤏"
        if w > (mean_w + 0.5 * std_w): return "Ampia ↔️"
        return "Standard 👌"
        
    df['FitClass'] = width.apply(classify)
    return df

def calcola_drive_index(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Calcola il 'Drive Index' (0-1). 
    Il GRIP agisce come moltiplicatore di efficienza.
    """
    # 1. Plate
    def score_plate(val):
        val = str(val).lower()
        if 'carbon' in val or 'carbitex' in val: return 1.0
        if 'fiberglass' in val: return 0.7
        if any(x in val for x in ['plastic', 'tpu', 'nylon']): return 0.5
        return 0.1 
    S_Plate = df['piastra'].apply(score_plate)

    # 2. Rocker
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

    # 3. Schiuma & Rigidità
    S_Foam = df['EnergyIndex'] 
    flex_val = pd.to_numeric(df['rigidezza_flex'], errors='coerce').fillna(15)
    S_Stiffness = ((flex_val - 5) / 30.0).clip(0, 1)
    
    # 4. Grip Efficiency Factor (0.8 a 1.1)
    # Un buon grip (1.0) dà un bonus del 10%. Un pessimo grip (0.0) dà un malus del 20%.
    S_Grip = calcola_grip_score(df)
    Grip_Factor = 0.8 + (0.3 * S_Grip) 

    # Formula Base Teeter-Totter
    Mechanical_Drive = S_Plate * S_Rocker * S_Stiffness
    Raw_Drive = (0.6 * Mechanical_Drive) + (0.4 * S_Foam)
    
    # Applicazione Efficienza Grip
    df['DriveIndex'] = (Raw_Drive * Grip_Factor).clip(0, 1)
    
    return df

# --- FUNZIONE PRINCIPALE ---

def calcola_indici(df: pd.DataFrame) -> pd.DataFrame:
    """ Calcola TUTTI gli indici biomeccanici e fisici. """
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

    # 2. Flex Index
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
    
    # 6. AGGIUNTA CRUCIALE: Chiamata alle funzioni Durabilità e Fit
    df = calcola_durability_index(df)
    df = calcola_fit_class(df)
    df = calcola_stability_index(df)
    df['GripIndex'] = calcola_grip_score(df).round(2)
    
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
        drive  = livello_index(row["Drive"])
        flex   = livello_index(row["Flex"])
        weight = livello_index(row["Weight"])
        return (f"Shock: {shock} | Drive: {drive} | Flex: {flex} | Peso: {weight}")

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
    try:
        target_vector = df.loc[df['label'] == target_label, metrics_cols].astype(float).values[0]
        df_calc = df.copy()
        
        vectors = df_calc[metrics_cols].astype(float).values
        
        if weights is not None:
            w = np.array(weights)
            w = w / w.sum()
            diff_sq = (vectors - target_vector) ** 2
            distances = np.sqrt((diff_sq * w).sum(axis=1))
        else:
            distances = np.linalg.norm(vectors - target_vector, axis=1)
            
        df_calc['distanza_similitudine'] = distances
        simili = df_calc[df_calc['label'] != target_label].sort_values('distanza_similitudine').head(n_simili)
        return simili
    except Exception:
        return pd.DataFrame()





