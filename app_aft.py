import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# =========================
#   FUNZIONI AFT (TRADUZIONE MATLAB)
# =========================

def calcola_indici(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica calcola_indici(T) in MATLAB.
    Richiede le colonne:
    shock_abs_tallone, shock_abs_mesopiede,
    energy_ret_tallone, energy_ret_mesopiede,
    rigidezza_flex, peso, altezza_tallone, passo
    """

    def safe_minmax(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        denom = max(xmax - xmin, np.finfo(float).eps)
        return (x - xmin) / denom

    # --- Shock & Energy ---
    w_heel = 0.4
    w_mid = 0.6

    S_heel = safe_minmax(df["shock_abs_tallone"])
    S_mid  = safe_minmax(df["shock_abs_mesopiede"])
    ER_h   = safe_minmax(df["energy_ret_tallone"])
    ER_m   = safe_minmax(df["energy_ret_mesopiede"])

    df["ShockIndex"]  = (w_heel * S_heel + w_mid * S_mid) / (w_heel + w_mid)
    df["EnergyIndex"] = (w_heel * ER_h   + w_mid * ER_m)  / (w_heel + w_mid)

    # --- Flex Index (campana adattiva per passo) ---
    Flex = df["rigidezza_flex"].astype(float).to_numpy()
    FlexIndex = np.zeros(len(df))
    passi = df["passo"].astype(str).str.lower().to_list()

    for i, tipo in enumerate(passi):
        if "race" in tipo:
            Flex_opt = 20.0
            sigma = 3.0
        elif "tempo" in tipo:
            Flex_opt = 17.0
            sigma = 2.5
        else:
            Flex_opt = 13.0
            sigma = 2.5

        FlexIndex[i] = np.exp(-((Flex[i] - Flex_opt) ** 2) / (2 * sigma ** 2))

    df["FlexIndex"] = FlexIndex

    # --- Weight Index ---
    W = df["peso"].astype(float).to_numpy()
    W_ref = np.nanmin(W)
    W_mean = np.nanmean(W)

    deltaW = W - W_ref
    alpha = 0.04
    beta = 0.4
    W_norm = deltaW / (W_mean - W_ref)
    WeightIndex = np.exp(-alpha * (W_norm * 100) ** beta)
    WeightIndex = WeightIndex / np.nanmax(WeightIndex)

    df["WeightIndex"] = WeightIndex

    # --- StackFactor (stabilità + energy mod) ---
    stack = df["altezza_tallone"].astype(float).to_numpy()

    EnergyMod = np.ones(len(df))
    StabilityMod = np.ones(len(df))

    # EnergyMod: stack > 45
    mask_hi = stack > 45
    if np.any(mask_hi):
        EnergyMod[mask_hi] = 1.0 + 0.0006 * (np.minimum(stack[mask_hi], 50.0) - 45.0)

    # EnergyMod: stack < 35
    mask_lo = stack < 35
    if np.any(mask_lo):
        EnergyMod[mask_lo] = 1.0 - 0.002 * (35.0 - np.maximum(stack[mask_lo], 30.0))

    # clamp [0.985, 1.006]
    EnergyMod = np.clip(EnergyMod, 0.985, 1.006)

    # StabilityMod: stack < 35
    if np.any(mask_lo):
        StabilityMod[mask_lo] = 1.0 / (1.0 + np.exp(-(stack[mask_lo] - 33.0) / 1.2))

    # StabilityMod: stack > 45
    if np.any(mask_hi):
        StabilityMod[mask_hi] = np.maximum(
            0.93,
            1.0 - 0.01 * (np.minimum(stack[mask_hi], 50.0) - 45.0)
        )

    df["StackFactor"] = StabilityMod
    df["EnergyIndex"] = df["EnergyIndex"] * EnergyMod * StabilityMod
    df["FlexIndex"]   = df["FlexIndex"]   * StabilityMod

    return df


def calcola_MPIB(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica calcola_MPIB(T).
    Usa:
    ShockIndex, EnergyIndex, FlexIndex, WeightIndex
    Crea:
    MPI_B e ordina in modo decrescente.
    """
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
    """
    Traduzione di esegui_clustering(T).
    Restituisce:
    - df con colonna 'Cluster'
    - cluster_summary (DataFrame con centroidi + descrizione)
    """
    rng = 42
    np.random.seed(rng)

    X = df[["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor"]].to_numpy()
    labels_cols = ["Shock", "Energy", "Flex", "Weight", "Stack"]

    K_values = np.arange(2, 11)  # 2:10
    SSE = []
    silh_mean = []

    for k in K_values:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=rng)
        idx_tmp = kmeans.fit_predict(X)
        SSE.append(kmeans.inertia_)
        s = silhouette_samples(X, idx_tmp)
        silh_mean.append(np.mean(s))

    SSE = np.array(SSE)
    silh_mean = np.array(silh_mean)

    # Metodo del gomito: curvatura di log(SSE)
    logSSE = np.log(SSE)
    d2 = np.gradient(np.gradient(logSSE))
    k_elbow = K_values[np.argmin(d2)]

    # Metodo silhouette
    k_silh = K_values[np.argmax(silh_mean)]

    # Decisione combinata
    k_elbow = max(2, min(int(k_elbow), int(K_values.max())))
    k_silh  = max(2, min(int(k_silh),  int(K_values.max())))
    k_opt   = int(round(0.7 * k_silh + 0.3 * k_elbow))

    k_opt = min(k_opt, 7)  # limite massimo pratico

    # K-means finale
    kmeans_final = KMeans(n_clusters=k_opt, n_init=50, random_state=rng)
    idx = kmeans_final.fit_predict(X)
    C = kmeans_final.cluster_centers_

    df["Cluster"] = idx + 1  # MATLAB usava 1..k

    cluster_summary = pd.DataFrame(C, columns=labels_cols)
    cluster_summary["Cluster"] = np.arange(1, k_opt + 1)

    # Descrizione automatica
    descrizioni = []
    for i in range(k_opt):
        c = cluster_summary.iloc[i][labels_cols].to_numpy()
        txt = ""

        # Shock
        if c[0] < 0.4:
            txt += "ammortizzazione elevata, "
        elif c[0] > 0.7:
            txt += "scarpe più rigide, "
        else:
            txt += "ammortizzazione bilanciata, "

        # Energy
        if c[1] > 0.7:
            txt += "elevato energy return, "
        elif c[1] < 0.4:
            txt += "ritorno di energia limitato, "
        else:
            txt += "energy return medio, "

        # Flex
        if c[2] > 0.7:
            txt += "molto flessibili, "
        elif c[2] < 0.4:
            txt += "più rigide, "
        else:
            txt += "flessibilità moderata, "

        # Weight
        if c[3] > 0.8:
            txt += "molto leggere, "
        elif c[3] < 0.6:
            txt += "più pesanti della media, "
        else:
            txt += "peso medio, "

        # Stack
        if c[4] > 0.8:
            txt += "stack elevato (geometria moderna)."
        elif c[4] < 0.5:
            txt += "stack ridotto o tradizionale."
        else:
            txt += "stack nella fascia sweet spot (35–45 mm)."

        descrizioni.append(f"Cluster {i+1}: {txt}")

    cluster_summary["Descrizione"] = descrizioni

    # Mappa descrizione nel df
    descr_map = dict(zip(cluster_summary["Cluster"], cluster_summary["Descrizione"]))
    df["ClusterDescrizione"] = df["Cluster"].map(descr_map)

    return df, cluster_summary

def plot_radar_indices(df_comp, metrics, label_col="label"):
    """
    Grafico radar con i singoli indici (una linea per scarpa).
    df_comp: dataframe con una riga per scarpa
    metrics: lista di colonne numeriche (es. ShockIndex, EnergyIndex, ...)
    label_col: colonna con il nome/etichetta della scarpa
    """
    import numpy as np

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    # chiudiamo il poligono aggiungendo il primo punto in fondo
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)      # parte da "su"
    ax.set_theta_direction(-1)          # gira in senso orario

    for _, row in df_comp.iterrows():
        values = [row[m] for m in metrics]
        # chiudiamo il poligono
        values = values + [values[0]]
        label = row[label_col]

        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, metrics)
    ax.set_ylim(0, 1)  # tutti gli indici sono normalizzati fra 0 e 1
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    return fig



# =========================
#   APP STREAMLIT
# =========================

st.set_page_config(page_title="AFT Explorer", layout="wide")

st.title("AFT Shoe Database – MPI & Clustering")
st.write("App basata sul CSV originale `database_AFT_20251124.csv`, con calcolo live di indici, MPI-B e cluster biomeccanici.")

file_name = "database_AFT_20251124.csv"

@st.cache_data
def load_and_process(path):
    df = pd.read_csv(path)

    # Calcola indici biomeccanici
    df = calcola_indici(df)

    # Calcola MPI-B
    df = calcola_MPIB(df)

    # Clustering
    df, cluster_summary = esegui_clustering(df)

    # ======================================
    # FORMATTING DECIMALS (INDICI & DROP)
    # ======================================
    
    # Indici biomeccanici → 3 decimali
    index_cols = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor", "MPI_B"]
    for col in index_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    # Drop → 1 decimale
    if "drop" in df.columns:
        df["drop"] = df["drop"].astype(float).round(1)

    # ======================================

    return df, cluster_summary

df, cluster_summary = load_and_process(file_name)

# ============================================
# IMPOSTAZIONI UTENTE: PESI & APPOGGIO
# ============================================

st.sidebar.header("Impostazioni personalizzate MPI")

# 1) Pesi tallone / avampiede
heel_pct = st.sidebar.slider(
    "Percentuale appoggio di tallone (%)",
    min_value=0,
    max_value=100,
    value=40,
    step=5
)
w_heel = heel_pct / 100.0
w_mid = 1.0 - w_heel
st.sidebar.write(f"Avampiede: {100 - heel_pct}%")

# 2) Importanza dei singoli indici (scala discreta 1–5)
w_shock  = st.sidebar.select_slider("Importanza Shock_abs",  options=[1,2,3,4,5], value=3)
w_energy = st.sidebar.select_slider("Importanza Energy_ret", options=[1,2,3,4,5], value=3)
w_flex   = st.sidebar.select_slider("Importanza Stiffness",  options=[1,2,3,4,5], value=3)
w_weight = st.sidebar.select_slider("Importanza Weight",     options=[1,2,3,4,5], value=3)

# Normalizza i pesi in modo che la SOMMA sia sempre = 1
raw_weights = np.array([w_shock, w_energy, w_flex, w_weight], dtype=float)
tot = raw_weights.sum()

if tot == 0:
    # se l'utente mette tutto a zero, uso i default originari
    norm_weights = np.array([0.20, 0.30, 0.20, 0.30])
else:
    norm_weights = raw_weights / tot

w_shock_eff, w_energy_eff, w_flex_eff, w_weight_eff = norm_weights

st.sidebar.markdown(
    f"""
**Pesi effettivi MPI (somma = 1):**  
- Shock_abs: `{w_shock_eff:.3f}`  
- Energy_ret: `{w_energy_eff:.3f}`  
- Stiffness: `{w_flex_eff:.3f}`  
- Weight: `{w_weight_eff:.3f}`
"""
)

# Normalizza i pesi degli indici (somma = 1)
tot = w_shock + w_energy + w_flex + w_weight
if tot == 0:
    tot = 1.0
w_shock  /= tot
w_energy /= tot
w_flex   /= tot
w_weight /= tot

# 3) Ricalcolo Shock/Energy in base a tallone/avampiede
def safe_minmax_series(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    xmin = x.min()
    xmax = x.max()
    denom = max(xmax - xmin, 1e-12)
    return (x - xmin) / denom

S_heel = safe_minmax_series(df["shock_abs_tallone"])
S_mid  = safe_minmax_series(df["shock_abs_mesopiede"])
ER_h   = safe_minmax_series(df["energy_ret_tallone"])
ER_m   = safe_minmax_series(df["energy_ret_mesopiede"])

df["ShockIndex"]  = (w_heel * S_heel + w_mid * S_mid)
df["EnergyIndex"] = (w_heel * ER_h   + w_mid * ER_m)

# Riscaliamo a [0,1]
for col in ["ShockIndex", "EnergyIndex"]:
    s = df[col]
    df[col] = (s - s.min()) / max(s.max() - s.min(), 1e-12)

# 4) Ricalcolo MPI_B con i pesi scelti dall’utente
df["MPI_B"] = (
    w_shock_eff  * df["ShockIndex"] +
    w_energy_eff * df["EnergyIndex"] +
    w_flex_eff   * df["FlexIndex"] +
    w_weight_eff * df["WeightIndex"]
)


# 5) Arrotondamenti (indici a 3 decimali, drop a 1)
index_cols = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor", "MPI_B"]
for col in index_cols:
    if col in df.columns:
        df[col] = df[col].astype(float).round(3)

if "drop" in df.columns:
    df["drop"] = df["drop"].astype(float).round(1)

# 6) (Opzionale) ricalcolo cluster con i nuovi indici
df, cluster_summary = esegui_clustering(df)

# ============================================

st.success(f"Database caricato e processato. Numero scarpe: {len(df)}")
st.write("Cluster trovati automaticamente:")
st.dataframe(cluster_summary, use_container_width=True)

# Label leggibile scarpa
def make_label(row):
    if "versione" in row and not pd.isna(row["versione"]):
        try:
            v = int(row["versione"])
            return f"{row['marca']} - {row['modello']} (v.{v})"
        except Exception:
            return f"{row['marca']} - {row['modello']}"
    else:
        return f"{row['marca']} - {row['modello']}"

df = df.copy()
df["label"] = df.apply(make_label, axis=1)

# =========================
#   FILTRI SIDEBAR
# =========================

st.sidebar.header("Filtri")

marche = ["Tutte"] + sorted(df["marca"].unique().tolist())
marca_sel = st.sidebar.selectbox("Marca", marche)

passi = ["Tutti"] + sorted(df["passo"].unique().tolist())
passo_sel = st.sidebar.selectbox("Categoria/Passo (AFT)", passi)

cluster_vals = ["Tutti"] + sorted(df["Cluster"].unique().tolist())
cluster_sel = st.sidebar.selectbox("Cluster biomeccanico", cluster_vals)

df_filt = df.copy()
if marca_sel != "Tutte":
    df_filt = df_filt[df_filt["marca"] == marca_sel]

if passo_sel != "Tutti":
    df_filt = df_filt[df_filt["passo"] == passo_sel]

if cluster_sel != "Tutti":
    df_filt = df_filt[df_filt["Cluster"] == cluster_sel]

st.subheader("Tabella filtrata")
st.write(f"Scarpe mostrate: {len(df_filt)}")

cols_tabella = [
    "marca", "modello", "versione", "passo",
    "MPI_B", "Cluster",
    "ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor",
    "peso", "drop", "altezza_tallone", "altezza_mesopiede"
]
cols_tabella = [c for c in cols_tabella if c in df_filt.columns]

st.dataframe(df_filt[cols_tabella], use_container_width=True)

# =========================
#   DETTAGLIO SCARPA
# =========================

st.subheader("Dettaglio scarpa")

if not df_filt.empty:
    scelta = st.selectbox(
        "Seleziona una scarpa",
        df_filt["label"].tolist()
    )
    scarpa = df_filt[df_filt["label"] == scelta].iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### {scarpa['marca']} {scarpa['modello']}")
        if "versione" in scarpa and not pd.isna(scarpa["versione"]):
            st.write(f"Versione: {int(scarpa['versione'])}")
        st.write(f"Passo / categoria (AFT): {scarpa['passo']}")
        st.write(f"Peso: {scarpa['peso']} g")
        st.write(f"Drop: {scarpa['drop']} mm")
        st.write(f"Stack (tallone): {scarpa['altezza_tallone']} mm")

    with col2:
        st.write("**Indici biomeccanici**")
        st.write(f"ShockIndex: {scarpa['ShockIndex']:.3f}")
        st.write(f"EnergyIndex: {scarpa['EnergyIndex']:.3f}")
        st.write(f"FlexIndex: {scarpa['FlexIndex']:.3f}")
        st.write(f"WeightIndex: {scarpa['WeightIndex']:.3f}")
        st.write(f"StackFactor: {scarpa['StackFactor']:.3f}")

    with col3:
        st.write("**Performance & cluster**")
        st.metric("MPI-B", f"{scarpa['MPI_B']:.3f}")
        st.write(f"Cluster: {scarpa['Cluster']}")
        st.write(scarpa["ClusterDescrizione"])
else:
    st.info("Nessuna scarpa corrisponde ai filtri selezionati.")

# =========================
#   CONFRONTO SCARPE
# =========================

st.subheader("Confronto scarpe")

if len(df_filt) >= 2:
    selezione_confronto = st.multiselect(
        "Seleziona fino a 3 scarpe da confrontare",
        df_filt["label"].tolist(),
        max_selections=3
    )

    if selezione_confronto:
        df_comp = df_filt[df_filt["label"].isin(selezione_confronto)].copy()

        # tabella numerica come prima (utile da leggere)
        colonne_confronto = [
            "label", "marca", "modello", "versione", "passo",
            "MPI_B", "Cluster",
            "ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor",
            "peso", "drop", "altezza_tallone", "altezza_mesopiede"
        ]
        colonne_confronto = [c for c in colonne_confronto if c in df_comp.columns]

        st.write("Tabella comparativa (MPI + indici)")
        st.dataframe(df_comp[colonne_confronto], use_container_width=True)

        # --- GRAFICO RADAR SUI 5 INDICI ---
        metrics = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex"]
        metrics = [m for m in metrics if m in df_comp.columns]

        if metrics:
            st.write("Profilo radar sugli indici biomeccanici")
            fig = plot_radar_indices(df_comp, metrics, label_col="label")
            st.pyplot(fig)
        else:
            st.info("Indici per il radar non disponibili.")
    else:
        st.info("Seleziona almeno una scarpa per il confronto.")
else:
    st.info("Servono almeno 2 scarpe nei filtri attuali per fare un confronto.")
