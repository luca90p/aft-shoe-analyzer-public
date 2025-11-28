import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# =========================
#   FUNZIONI AFT (CORE LOGIC)
# =========================

def calcola_indici(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola gli indici biomeccanici normalizzati.
    AGGIORNAMENTO V3 (Hybrid Weight):
    - WeightIndex: Mix ponderato tra Punteggio Globale (70%) e Relativo (30%).
      Garantisce che una scarpa oggettivamente leggera (es. Race) batta sempre 
      una scarpa più pesante (es. Daily), anche se quest'ultima è "ottima per la sua categoria".
    """

    def safe_minmax(x: pd.Series) -> pd.Series:
        x = x.astype(float)
        xmin = np.nanmin(x)
        xmax = np.nanmax(x)
        denom = max(xmax - xmin, np.finfo(float).eps)
        return (x - xmin) / denom

    # --- 1. Shock & Energy (Base) ---
    w_heel = 0.4
    w_mid = 0.6
    S_heel = safe_minmax(df["shock_abs_tallone"])
    S_mid  = safe_minmax(df["shock_abs_mesopiede"])
    ER_h   = safe_minmax(df["energy_ret_tallone"])
    ER_m   = safe_minmax(df["energy_ret_mesopiede"])

    df["ShockIndex"]  = (w_heel * S_heel + w_mid * S_mid) / (w_heel + w_mid)
    df["EnergyIndex"] = (w_heel * ER_h   + w_mid * ER_m)  / (w_heel + w_mid)

    # --- 2. Flex Index ---
    Flex = df["rigidezza_flex"].astype(float).to_numpy()
    FlexIndex = np.zeros(len(df))
    passi = df["passo"].astype(str).str.lower().to_list()

    for i, tipo in enumerate(passi):
        if "race" in tipo:
            Flex_opt = 20.0; sigma = 3.0
        elif "tempo" in tipo:
            Flex_opt = 17.0; sigma = 2.5
        else:
            Flex_opt = 13.0; sigma = 2.5
        FlexIndex[i] = np.exp(-((Flex[i] - Flex_opt) ** 2) / (2 * sigma ** 2))
    df["FlexIndex"] = FlexIndex

    # --- 3. Weight Index (Hybrid Approach) ---
    W = df["peso"].astype(float).to_numpy()
    
    # -- Configurazione Pesi --
    # MIX FACTOR: Quanto conta il peso assoluto vs quello relativo?
    # 0.70 significa che il 70% del voto dipende dai grammi effettivi (fisica),
    # il 30% dipende da quanto è brava rispetto alla categoria.
    ALPHA_GLOBAL = 0.75 
    ALPHA_LOCAL  = 1.0 - ALPHA_GLOBAL
    
    # Parametro curvatura (0.5 = radice quadrata, premia i medi)
    GAMMA = 0.5

    # A) CALCOLO SCORE GLOBALE (su tutto il database)
    w_min_g = np.nanmin(W)
    w_max_g = np.nanmax(W)
    denom_g = max(w_max_g - w_min_g, 1.0)
    
    # Più è basso il peso, più alto il punteggio (1 - norm)
    # Usiamo clip per sicurezza
    w_norm_g = np.clip((w_max_g - W) / denom_g, 0, 1)
    Score_Global = np.power(w_norm_g, GAMMA)

    # B) CALCOLO SCORE RELATIVO (per categoria)
    Score_Local = np.zeros(len(df))
    passi_series = df["passo"].astype(str).str.lower()
    
    mask_race  = passi_series.str.contains("race", na=False).to_numpy()
    mask_tempo = passi_series.str.contains("tempo", na=False).to_numpy()
    mask_daily = ~(mask_race | mask_tempo)
    masks = [mask_race, mask_tempo, mask_daily]

    for mask in masks:
        if np.any(mask):
            w_subset = W[mask]
            w_min_l = np.nanmin(w_subset)
            w_max_l = np.nanmax(w_subset)
            denom_l = max(w_max_l - w_min_l, 1.0)
            
            w_norm_l = np.clip((w_max_l - w_subset) / denom_l, 0, 1)
            Score_Local[mask] = np.power(w_norm_l, GAMMA)
            
    # C) MIX FINALE
    # Se una scarpa non ha categoria (improbabile), usiamo solo global
    # Score_Local è inizializzato a 0, ma le maschere coprono tutto.
    
    df["WeightIndex"] = (ALPHA_GLOBAL * Score_Global) + (ALPHA_LOCAL * Score_Local)

    # --- 4. StackFactor ---
    stack = df["altezza_tallone"].astype(float).to_numpy()
    EnergyMod = np.ones(len(df))
    StabilityMod = np.ones(len(df))

    mask_hi = stack > 45
    if np.any(mask_hi):
        EnergyMod[mask_hi] = 1.0 + 0.0006 * (np.minimum(stack[mask_hi], 50.0) - 45.0)

    mask_lo = stack < 35
    if np.any(mask_lo):
        EnergyMod[mask_lo] = 1.0 - 0.002 * (35.0 - np.maximum(stack[mask_lo], 30.0))

    EnergyMod = np.clip(EnergyMod, 0.985, 1.006)

    if np.any(mask_lo):
        StabilityMod[mask_lo] = 1.0 / (1.0 + np.exp(-(stack[mask_lo] - 33.0) / 1.2))

    if np.any(mask_hi):
        StabilityMod[mask_hi] = np.maximum(0.93, 1.0 - 0.01 * (np.minimum(stack[mask_hi], 50.0) - 45.0))

    df["StackFactor"] = StabilityMod
    df["EnergyIndex"] = df["EnergyIndex"] * EnergyMod * StabilityMod
    df["FlexIndex"]   = df["FlexIndex"]   * StabilityMod

    return df

def calcola_MPIB(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola l'MPI_B base. I pesi verranno poi sovrascritti dall'utente
    nella UI, ma serve una base per il primo load.
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
    Esegue clustering automatico (Elbow + Silhouette) usando gli indici biomeccanici.
    """

    # Helper: Livelli descrittivi
    def livello_index(val: float) -> str:
        if val < 0.33: return "Basso"
        elif val < 0.66: return "Medio"
        else: return "Alto"

    def descrizione_cluster_simplificata(row: pd.Series) -> str:
        shock  = livello_index(row["Shock"])
        energy = livello_index(row["Energy"])
        flex   = livello_index(row["Flex"])
        weight = livello_index(row["Weight"])
        stack  = livello_index(row["Stack"])
        return (f"Ammortizz.: {shock} | Energy: {energy} | "
                f"Flex: {flex} | Peso: {weight} | Stack: {stack}")

    rng = 42
    np.random.seed(rng)

    # Usiamo i 5 indici chiave
    X = df[["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor"]].to_numpy()
    labels_cols = ["Shock", "Energy", "Flex", "Weight", "Stack"]

    # Scelta K ottimale
    K_values = np.arange(2, 11)
    SSE = []
    silh_mean = []

    for k in K_values:
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=rng)
        idx_tmp = kmeans.fit_predict(X)
        SSE.append(kmeans.inertia_)
        silh_mean.append(np.mean(silhouette_samples(X, idx_tmp)))

    SSE = np.array(SSE)
    silh_mean = np.array(silh_mean)

    # Logica combinata Elbow/Silhouette
    # Elbow (derivata seconda)
    logSSE = np.log(SSE)
    # Gestione array piccoli per gradient
    if len(logSSE) > 2:
        d2 = np.gradient(np.gradient(logSSE))
        k_elbow = K_values[np.argmin(d2)]
    else:
        k_elbow = 3

    k_silh = K_values[np.argmax(silh_mean)]

    k_elbow = max(2, min(int(k_elbow), int(K_values.max())))
    k_silh  = max(2, min(int(k_silh),  int(K_values.max())))
    
    # Mix: 70% silhouette, 30% elbow
    k_opt = int(round(0.7 * k_silh + 0.3 * k_elbow))
    k_opt = max(2, min(k_opt, 7)) # Limitiamo a max 7 cluster per leggibilità

    # Clustering Finale
    kmeans_final = KMeans(n_clusters=k_opt, n_init=50, random_state=rng)
    idx = kmeans_final.fit_predict(X)
    C = kmeans_final.cluster_centers_

    df["Cluster"] = idx + 1

    # Summary
    cluster_summary = pd.DataFrame(C, columns=labels_cols)
    cluster_summary["Cluster"] = np.arange(1, k_opt + 1)
    cluster_summary["Descrizione"] = cluster_summary.apply(descrizione_cluster_simplificata, axis=1)

    descr_map = dict(zip(cluster_summary["Cluster"], cluster_summary["Descrizione"]))
    df["ClusterDescrizione"] = df["Cluster"].map(descr_map)

    return df, cluster_summary


def plot_radar_indices(df_comp, metrics, label_col="label"):
    """ Grafico Radar Matplotlib """
    import numpy as np
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in df_comp.iterrows():
        values = [row[m] for m in metrics]
        values = values + [values[0]]
        label = row[label_col]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, metrics)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig


# =========================
#   APP STREAMLIT
# =========================

st.set_page_config(page_title="AFT Explorer V2", layout="wide")

st.title("AFT Shoe Database – MPI & Clustering V2")
st.markdown("Basata su `database_completo_AFT_20251124_clean.csv`. Nuova logica Weight Index (Categorical Relative + Convex).")

# --- EXPANDER SPIEGAZIONI ---
with st.expander("ℹ️ Dettagli sugli Indici (Aggiornato V2)"):
    st.markdown("""
**1. ShockIndex:** Ammortizzazione (40% Tallone, 60% Avampiede).  
**2. EnergyIndex:** Ritorno energia (modulato dallo stack).  
**3. FlexIndex:** Rigidità ottimale basata sul passo (Race vs Tempo vs Daily).  
**4. WeightIndex (NUOVO):** - Confronta la scarpa **solo con la sua categoria** (Race con Race, Daily con Daily).
   - Usa una curva "convessa" ($\gamma=0.5$): le scarpe di peso medio non vengono penalizzate troppo. Solo quelle molto pesanti (per la loro categoria) ricevono voti bassi.
   - Premia al massimo solo la più leggera in assoluto della categoria.
**5. StackFactor:** Penalità/Bonus stabilità in base all'altezza.
    """)

file_name = "database_completo_AFT_20251124_clean.csv"

@st.cache_data
def load_and_process(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File {path} non trovato. Assicurati che sia nella stessa cartella.")
        return pd.DataFrame(), pd.DataFrame()

    # Pipeline
    df = calcola_indici(df)
    df = calcola_MPIB(df) # Calcolo base
    df, cluster_summary = esegui_clustering(df)

    # Formatting
    index_cols = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor", "MPI_B"]
    for col in index_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    if "drop" in df.columns:
        df["drop"] = df["drop"].astype(float).round(1)

    price_cols = [c for c in df.columns if "prezzo" in c.lower()]
    if price_cols:
        df[price_cols[0]] = df[price_cols[0]].astype(float).round(0)

    return df, cluster_summary

df_raw, cluster_summary_raw = load_and_process(file_name)

if df_raw.empty:
    st.stop()

# Lavoriamo su una copia per i filtri dinamici
df = df_raw.copy()
PRICE_COLS = [c for c in df.columns if "prezzo" in c.lower()]
PRICE_COL = PRICE_COLS[0] if PRICE_COLS else None

# =========================
#   SIDEBAR: CONTROLLI
# =========================
st.sidebar.header("Impostazioni MPI Personalizzato")

# 1. Slider Pesi
heel_pct = st.sidebar.slider("Appoggio Tallone (%)", 0, 100, 40, 5)
w_heel = heel_pct / 100.0
w_mid = 1.0 - w_heel

st.sidebar.markdown("---")
st.sidebar.write("**Importanza Indici (1-5)**")
w_shock  = st.sidebar.slider("Shock (Ammortizz.)", 1, 5, 3)
w_energy = st.sidebar.slider("Energy (Ritorno)", 1, 5, 3)
w_flex   = st.sidebar.slider("Stiffness (Flex)", 1, 5, 3)
w_weight = st.sidebar.slider("Weight (Leggerezza)", 1, 5, 3)

# Normalizzazione pesi
raw_weights = np.array([w_shock, w_energy, w_flex, w_weight], dtype=float)
tot = raw_weights.sum() if raw_weights.sum() > 0 else 1.0
norm_weights = raw_weights / tot
w_shock_eff, w_energy_eff, w_flex_eff, w_weight_eff = norm_weights

st.sidebar.info(
    f"Pesi Reali: Shock {w_shock_eff:.2f}, Energy {w_energy_eff:.2f}, "
    f"Flex {w_flex_eff:.2f}, Weight {w_weight_eff:.2f}"
)

# =========================
#   RICALCOLO LIVE (MPI)
# =========================
# Nota: WeightIndex e FlexIndex sono statici (o meglio, calcolati intelligentemente in load_and_process),
# mentre Shock e Energy dipendono dallo slider tallone/avampiede.

def safe_minmax_series(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)

# Ricalcolo Shock/Energy in base a tallone/avampiede
S_heel = safe_minmax_series(df["shock_abs_tallone"])
S_mid  = safe_minmax_series(df["shock_abs_mesopiede"])
ER_h   = safe_minmax_series(df["energy_ret_tallone"])
ER_m   = safe_minmax_series(df["energy_ret_mesopiede"])

df["ShockIndex"]  = (w_heel * S_heel + w_mid * S_mid)
df["EnergyIndex"] = (w_heel * ER_h   + w_mid * ER_m)
# Riscaliamo 0-1
df["ShockIndex"] = safe_minmax_series(df["ShockIndex"])
df["EnergyIndex"] = safe_minmax_series(df["EnergyIndex"])

# Ricalcolo MPI_B
df["MPI_B"] = (
    w_shock_eff  * df["ShockIndex"] +
    w_energy_eff * df["EnergyIndex"] +
    w_flex_eff   * df["FlexIndex"] +
    w_weight_eff * df["WeightIndex"]
).round(3)

# Value Index
if PRICE_COL:
    # Calcoliamo Value Index (MPI / Prezzo)
    # Filtriamo prezzi nulli o zero per evitare crash
    valid_p = (df[PRICE_COL] > 10) # almeno 10 euro
    vals = df.loc[valid_p, "MPI_B"] / df.loc[valid_p, PRICE_COL]
    
    # Normalizziamo 0-1
    if not vals.empty:
        v_min, v_max = vals.min(), vals.max()
        df.loc[valid_p, "ValueIndex"] = ((vals - v_min) / (v_max - v_min)).round(3)
    else:
        df["ValueIndex"] = 0.0
else:
    df["ValueIndex"] = 0.0

# Label
def make_label(row):
    lbl = f"{row['marca']} {row['modello']}"
    if pd.notna(row.get("versione")):
        lbl += f" v{int(row['versione'])}"
    return lbl

df["label"] = df.apply(make_label, axis=1)

# =========================
#   INTERFACCIA MAIN
# =========================

st.subheader("Cluster Biomeccanici (Profilo Medio)")
st.dataframe(cluster_summary_raw, use_container_width=True)

# FILTRI
st.sidebar.markdown("---")
st.sidebar.header("Filtra Database")
all_brands = ["Tutte"] + sorted(df["marca"].unique())
sel_brand = st.sidebar.selectbox("Marca", all_brands)

all_passi = ["Tutti"] + sorted(df["passo"].unique())
sel_passo = st.sidebar.selectbox("Categoria", all_passi)

df_filt = df.copy()
if sel_brand != "Tutte":
    df_filt = df_filt[df_filt["marca"] == sel_brand]
if sel_passo != "Tutti":
    df_filt = df_filt[df_filt["passo"] == sel_passo]

st.subheader(f"Database ({len(df_filt)} scarpe)")

cols_view = ["label", "passo", "peso", "MPI_B", "ValueIndex", 
             "ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "ClusterDescrizione"]
if PRICE_COL: cols_view.insert(4, PRICE_COL)

st.dataframe(
    df_filt[cols_view].sort_values("MPI_B", ascending=False), 
    use_container_width=True
)



# ============================================
# UTILITY: SCATTER PLOT MPI vs PREZZO (GLOBALE + CONFRONTO)
# ============================================

def plot_mpi_vs_price(df_val, df_comp_labels, price_col):
    """
    Scatter plot MPI-B vs Prezzo su tutto il database filtrato (df_val), 
    evidenziando i modelli selezionati (df_comp_labels).
    """
    
    # Prepara il grafico
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1. Plot di tutte le scarpe filtrate (sfondo)
    ax.scatter(
        df_val[price_col],
        df_val["MPI_B"],
        color='gray',
        alpha=0.4,
        s=30,
        label="Tutte le scarpe filtrate"
    )

    # 2. Evidenziazione dei modelli in confronto
    df_comp = df_val[df_val['label'].isin(df_comp_labels)].copy()
    
    if not df_comp.empty:
        # Plot dei punti evidenziati
        ax.scatter(
            df_comp[price_col],
            df_comp["MPI_B"],
            color='red',
            edgecolors='black',
            linewidths=1.5,
            alpha=0.8,
            s=120, # Punti più grandi e marcati
            label="Modelli selezionati"
        )
        
        # 3. Aggiunta delle etichette dettagliate ai punti evidenziati
        for i, row in df_comp.iterrows():
            
            # Testo dettagliato per l'annotazione
            text_label = (
                f"{row['label']}\n"
                f"MPI: {row['MPI_B']:.3f}\n"
                f"Costo: {row[price_col]:.0f}€\n"
                f"Value: {row['ValueIndex']:.3f}"
            )
            
            ax.annotate(
                text_label,
                (row[price_col], row["MPI_B"]),
                textcoords="offset points",
                xytext=(10, 5),  # Sposta l'etichetta a destra e leggermente in alto
                ha='left',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none") # Sfondo per leggibilità
            )

    ax.set_title("MPI-B Score vs. Prezzo (Performance vs. Costo)")
    ax.set_xlabel(f"{price_col} (€)")
    ax.set_ylabel("MPI-B Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig


# ============================================
#   ANALISI E CONFRONTO (RIORGANIZZATO)
# ============================================

# Selezioni da usare in entrambi i grafici (Dettaglio -> Confronto)
selected_for_detail = None
if not df_filt.empty:
    selected_for_detail = st.selectbox(
        "Seleziona una scarpa per il Dettaglio",
        df_filt["label"].tolist(),
        index=0 
    )

# --- BLOCCO DETTAGLIO (usa selected_for_detail) ---
st.subheader("Dettaglio scarpa")

if selected_for_detail:
    scarpa = df_filt[df_filt["label"] == selected_for_detail].iloc[0]
    # ... [MANTIENI QUI TUTTO IL TUO CODICE DI VISUALIZZAZIONE COL1, COL2, COL3] ...
    # ... (omesso per brevità nella risposta, ma devi lasciarlo) ...

    # --- BLOCCO DETTAGLIO (usa selected_for_detail) ---
st.subheader("Dettaglio scarpa")

if selected_for_detail:
    scarpa = df_filt[df_filt["label"] == selected_for_detail].iloc[0]
    
    # Inizializzo 'scelta' per il multiselect del confronto
    scelta = selected_for_detail

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### {scarpa['marca']} {scarpa['modello']}")
        if "versione" in scarpa and pd.notna(scarpa["versione"]):
            st.write(f"Versione: {int(scarpa['versione'])}")
        st.write(f"Passo / categoria (AFT): {scarpa['passo']}")
        st.write(f"Peso: {scarpa['peso']} g")

        # Prezzo (se presente nel dataframe)
        if PRICE_COL is not None and pd.notna(scarpa[PRICE_COL]):
            st.write(f"Prezzo: {scarpa[PRICE_COL]:.0f} €")

        st.write(f"Drop: {scarpa['drop']} mm")
        st.write(f"Stack (tallone): {scarpa['altezza_tallone']} mm")

    with col2:
        st.write("**Indici biomeccanici**")
        # Uso st.progress per una visualizzazione immediata
        st.progress(float(scarpa['ShockIndex']), text=f"ShockIndex: {scarpa['ShockIndex']:.3f}")
        st.progress(float(scarpa['EnergyIndex']), text=f"EnergyIndex: {scarpa['EnergyIndex']:.3f}")
        st.progress(float(scarpa['FlexIndex']), text=f"FlexIndex: {scarpa['FlexIndex']:.3f}")
        st.progress(float(scarpa['WeightIndex']), text=f"WeightIndex: {scarpa['WeightIndex']:.3f}")
        st.write(f"StackFactor: {scarpa['StackFactor']:.3f}")

    with col3:
        st.write("**Performance & cluster**")
        st.metric("MPI-B", f"{scarpa['MPI_B']:.3f}")

        # Value Index normalizzato 0–1 (qualità/prezzo)
        if "ValueIndex" in scarpa.index and pd.notna(scarpa["ValueIndex"]):
            st.write(f"Value index (0–1): {scarpa['ValueIndex']:.3f}")

        # Nome cluster
        st.write(f"Cluster: {int(scarpa['Cluster'])}")

        # Descrizione estesa del cluster
        if "ClusterDescrizione" in scarpa.index:
            st.write(scarpa["ClusterDescrizione"])

else:
    st.info("Nessuna scarpa corrisponde ai filtri selezionati.")
    scelta = None
# --- FINE BLOCCO DETTAGLIO ---
    
    # Riporto qui solo il codice per popolare 'scelta' per il multiselect
    scelta = selected_for_detail

else:
    st.info("Nessuna scarpa corrisponde ai filtri selezionati.")
    scelta = None
# --- FINE BLOCCO DETTAGLIO ---


# =========================
#   CONFRONTO SCARPE E GRAFICO GLOBALE
# =========================

st.subheader("Confronto e Analisi Globale")

# Pre-seleziona la scarpa dal Dettaglio
initial_selection = [scelta] if scelta and scelta in df_filt["label"].tolist() else []

selezione_confronto = st.multiselect(
    "Seleziona fino a 5 scarpe da confrontare (per evidenziarle nei grafici)",
    df_filt["label"].tolist(),
    max_selections=5,
    default=initial_selection
)

if PRICE_COL is not None and PRICE_COL in df_filt.columns:
    
    # 1. Prepara i dati validi (MPI e Prezzo validi)
    df_val = df_filt.dropna(subset=[PRICE_COL, "MPI_B"]).copy()
    df_val = df_val[df_val[PRICE_COL] > 0]
    
    if not df_val.empty:
        # --- SCATTER PLOT MPI vs PREZZO ---
        st.write("### 📊 MPI-B vs Prezzo: Posizionamento sul Mercato")
        fig_scatter = plot_mpi_vs_price(df_val, selezione_confronto, PRICE_COL)
        st.pyplot(fig_scatter)
        
        # --- CLASSIFICA VALUE INDEX ---
        st.write("### 🏆 Classifica Qualità/Prezzo (MPI-B / Costo)")
        df_val_sorted = df_val.sort_values(by="ValueIndex", ascending=False)
        cols_show = ["label", "passo", "MPI_B", PRICE_COL, "ValueIndex"]
        st.dataframe(df_val_sorted[[c for c in cols_show if c in df_val_sorted.columns]], use_container_width=True)
    else:
        st.info("Nessuna scarpa con prezzo e MPI-B validi nei filtri attuali.")
else:
    st.warning("Colonna prezzo non disponibile nel dataset per l'analisi.")


# --- GRAFICO RADAR (SOLO I MODELLI CONFRONTATI) ---
if selezione_confronto:
    st.write("---")
    st.write("### Profilo Biomeccanico Dettagliato (Radar)")
    df_comp = df_filt[df_filt["label"].isin(selezione_confronto)].copy()

    metrics = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex"]
    metrics = [m for m in metrics if m in df_comp.columns]

    if metrics:
        fig = plot_radar_indices(df_comp, metrics, label_col="label")
        st.pyplot(fig)
    else:
        st.info("Indici per il radar non disponibili.")




