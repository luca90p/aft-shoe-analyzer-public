import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# =========================
#   FUNZIONI AFT (CORE LOGIC)
# =========================

def calcola_indici(df: pd.DataFrame) -> pd.DataFrame:
    """ Calcola gli indici biomeccanici normalizzati. """

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
    
    ALPHA_GLOBAL = 0.75 
    ALPHA_LOCAL  = 1.0 - ALPHA_GLOBAL
    GAMMA = 0.5 

    # A) CALCOLO SCORE GLOBALE
    w_min_g = np.nanmin(W)
    w_max_g = np.nanmax(W)
    denom_g = max(w_max_g - w_min_g, 1.0)
    w_norm_g = np.clip((w_max_g - W) / denom_g, 0, 1)
    Score_Global = np.power(w_norm_g, GAMMA)

    # B) CALCOLO SCORE RELATIVO
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
    """ Calcola l'MPI_B base. """
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
    """ Esegue clustering automatico (Elbow + Silhouette). """

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

    X = df[["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor"]].to_numpy()
    labels_cols = ["Shock", "Energy", "Flex", "Weight", "Stack"]

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

    k_elbow = max(2, min(int(k_elbow), int(K_values.max())))
    k_silh  = max(2, min(int(k_silh),  int(K_values.max())))
    
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


def plot_mpi_vs_price_plotly(df_val, price_col, selected_points_labels):
    """ Scatter plot MPI-B vs Prezzo usando Plotly (solo hover). """
    
    # Crea la colonna per l'evidenziazione
    df_val['Colore_Evidenziazione'] = df_val['label'].apply(
        lambda x: 'Selezionato' if x in selected_points_labels else 'Mercato'
    )
    
    # Ordina per portare i punti selezionati in primo piano nel grafico
    df_val = df_val.sort_values(by='Colore_Evidenziazione', ascending=True).reset_index(drop=True)
    
    df_val['hover_text'] = df_val.apply(
        lambda row: f"<b>{row['label']}</b><br>"
                    f"MPI-B: {row['MPI_B']:.3f}<br>"
                    f"Costo: {row[price_col]:.0f}€<br>"
                    f"Value Index: {row['ValueIndex']:.3f}", axis=1
    )

    fig = px.scatter(
        df_val,
        x=price_col,
        y="MPI_B",
        color='Colore_Evidenziazione',
        size='ValueIndex',
        size_max=25,
        hover_name='hover_text',
        color_discrete_map={
            'Selezionato': 'red',
            'Mercato': 'gray'
        },
        custom_data=['label'],
        labels={price_col: f'{price_col} (€)', "MPI_B": "MPI-B Score"},
        title="MPI-B Score vs. Prezzo (Dimensione = Value Index)"
    )

    fig.update_traces(
        marker=dict(
            opacity=0.7,
            line=dict(width=1, color='black')
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        hovermode="closest",
        yaxis=dict(range=[0, 1.05]),
        legend_title_text='Legenda'
    )
    
    return fig

def trova_scarpe_simili(df, target_label, metrics_cols, n_simili=3):
    """
    Trova le n scarpe più simili basandosi sulla distanza euclidea 
    degli indici biomeccanici (escluso Value Index).
    """
    try:
        target_vector = df.loc[df['label'] == target_label, metrics_cols].astype(float).values[0]
        df_calc = df.copy()
        
        # Calcolo distanza euclidea vettoriale
        vectors = df_calc[metrics_cols].astype(float).values
        distances = np.linalg.norm(vectors - target_vector, axis=1)
        
        df_calc['distanza_similitudine'] = distances
        
        # Escludi se stesso
        simili = df_calc[df_calc['label'] != target_label].sort_values('distanza_similitudine').head(n_simili)
        
        return simili
    except Exception as e:
        st.error(f"Errore nel calcolo similitudine: {e}")
        return pd.DataFrame()


# =========================
#   HELPER DI VISUALIZZAZIONE
# =========================
def render_stars(value):
    """
    Converte un valore 0-1 in una stringa di 5 stelle.
    Es: 0.8 -> ★★★★☆
    """
    if pd.isna(value):
        return ""
    score = int(round(value * 5))
    score = max(0, min(5, score))
    full_star = "★"
    empty_star = "☆"
    return (full_star * score) + (empty_star * (5 - score))


# =========================
#   APP STREAMLIT
# =========================

st.set_page_config(page_title="AFT Explorer V2", layout="wide")

st.title("AFT Shoe Database – MPI & Clustering V2")

file_name = "database_completo_AFT_20251124_clean.csv"

# =========================
#   PRE-PROCESSING (Cached)
# =========================

@st.cache_data
def load_and_process(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File {path} non trovato. Assicurati che sia nella stessa cartella.")
        return pd.DataFrame(), pd.DataFrame()

    df = calcola_indici(df)
    df = calcola_MPIB(df)
    df, cluster_summary = esegui_clustering(df)

    index_cols = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor", "MPI_B"]
    for col in index_cols:
        if col in df.columns:
            df[col] = df[col].astype(float).round(3)

    if "drop" in df.columns:
        df["drop"] = df["drop"].astype(float).round(1)

    price_cols = [c for c in df.columns if "prezzo" in c.lower()]
    if price_cols:
        df[price_cols[0]] = df[price_cols[0]].astype(float).round(0)

    def make_label(row):
        lbl = f"{row['marca']} {row['modello']}"
        if pd.notna(row.get("versione")):
            lbl += f" v{int(row['versione'])}"
        return lbl
    df["label"] = df.apply(make_label, axis=1)

    return df, cluster_summary

df_raw, cluster_summary_raw = load_and_process(file_name)

if df_raw.empty:
    st.stop()

df = df_raw.copy()
PRICE_COLS = [c for c in df.columns if "prezzo" in c.lower()]
PRICE_COL = PRICE_COLS[0] if PRICE_COLS else None

# =========================
#   SIDEBAR: INFO E FILTRI BASE
# =========================

with st.sidebar:
    st.header("Filtri per il Database")
    st.subheader("Cluster Biomeccanici")
    st.dataframe(cluster_summary_raw, use_container_width=True)

    st.markdown("---")

    # FILTRI BASE
    all_brands = ["Tutte"] + sorted(df["marca"].unique())
    sel_brand = st.selectbox("Marca", all_brands)

    all_passi = ["Tutti"] + sorted(df["passo"].unique())
    sel_passo = st.selectbox("Categoria", all_passi)
    
    df_filt = df.copy()
    if sel_brand != "Tutte":
        df_filt = df_filt[df_filt["marca"] == sel_brand]
    if sel_passo != "Tutti":
        df_filt = df_filt[df_filt["passo"] == sel_passo]

# ============================================
# 1. INPUT GUIDATO (MPI WIZARD)
# ============================================

st.header("Step 1: Personalizza i tuoi Criteri di Performance (MPI)")
st.info("Regola i parametri qui sotto per calcolare l'MPI Score in base alle **tue esigenze di corsa**.")

col_appoggio, col_pesi = st.columns(2)

# --- Appoggio ---
with col_appoggio:
    st.subheader("1A. Appoggio Piede (Shock/Energy Index)")
    heel_pct = st.slider(
        "Percentuale appoggio di tallone (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5
    )
    w_heel = heel_pct / 100.0
    w_mid = 1.0 - w_heel
    st.write(f"Avampiede: {100 - heel_pct}%")

# --- Importanza Indici ---
with col_pesi:
    st.subheader("1B. Importanza dei Parametri (MPI Score)")
    st.markdown("Peso da 1 (Basso) a 5 (Alto)")
    w_shock  = st.slider("Shock (Ammortizz.)", 1, 5, 3)
    w_energy = st.slider("Energy (Ritorno)", 1, 5, 3)
    w_flex   = st.slider("Stiffness (Flex)", 1, 5, 3)
    w_weight = st.slider("Weight (Leggerezza)", 1, 5, 3)

# Ricalcolo MPI dinamico
raw_weights = np.array([w_shock, w_energy, w_flex, w_weight], dtype=float)
tot = raw_weights.sum() if raw_weights.sum() > 0 else 1.0
norm_weights = raw_weights / tot
w_shock_eff, w_energy_eff, w_flex_eff, w_weight_eff = norm_weights

# Ricalcolo Shock/Energy
def safe_minmax_series(x):
    return (x - x.min()) / max(x.max() - x.min(), 1e-12)

S_heel = safe_minmax_series(df_filt["shock_abs_tallone"])
S_mid  = safe_minmax_series(df_filt["shock_abs_mesopiede"])
ER_h   = safe_minmax_series(df_filt["energy_ret_tallone"])
ER_m   = safe_minmax_series(df_filt["energy_ret_mesopiede"])

df_filt.loc[:, "ShockIndex_calc"] = (w_heel * S_heel + w_mid * S_mid)
df_filt.loc[:, "EnergyIndex_calc"] = (w_heel * ER_h   + w_mid * ER_m)
df_filt["ShockIndex_calc"] = safe_minmax_series(df_filt["ShockIndex_calc"])
df_filt["EnergyIndex_calc"] = safe_minmax_series(df_filt["EnergyIndex_calc"])

# Calcolo MPI Finale
df_filt.loc[:, "MPI_B"] = (
    w_shock_eff  * df_filt["ShockIndex_calc"] +
    w_energy_eff * df_filt["EnergyIndex_calc"] +
    df_filt["FlexIndex"] * w_flex_eff +
    df_filt["WeightIndex"] * w_weight_eff
).round(3)

# Ricalcolo Value Index
if PRICE_COL:
    valid_p = (df_filt[PRICE_COL] > 10)
    vals = df_filt.loc[valid_p, "MPI_B"] / df_filt.loc[valid_p, PRICE_COL]
    
    if not vals.empty:
        v_min, v_max = vals.min(), vals.max()
        df_filt.loc[valid_p, "ValueIndex"] = ((vals - v_min) / (v_max - v_min)).round(3)
    else:
        df_filt["ValueIndex"] = 0.0
else:
    df_filt["ValueIndex"] = 0.0

st.success("MPI Score ricalcolato in base ai tuoi criteri!")
st.markdown("---")

# ============================================
# 2. RISULTATI GLOBALI INTERATTIVI
# ============================================

st.header("Step 2: Analisi di Mercato (MPI vs Prezzo)")

if PRICE_COL is not None and PRICE_COL in df_filt.columns:
    
    df_val = df_filt.dropna(subset=[PRICE_COL, "MPI_B", "ValueIndex"]).copy()
    df_val = df_val[df_val[PRICE_COL] > 0]
    
    if not df_val.empty:
        
        # 2A. INIZIALIZZAZIONE E ORDINAMENTO
        df_val_sorted = df_val.sort_values(by="ValueIndex", ascending=False)
        default_label_on_load = df_val_sorted.iloc[0]['label']
        
        # Inizializzazione Session State
        if 'selected_point_key' not in st.session_state:
            st.session_state['selected_point_key'] = default_label_on_load
        
        # FIX: Controllo consistenza selezione post-filtro
        current_model_list = df_val_sorted['label'].tolist()
        if st.session_state['selected_point_key'] not in current_model_list:
             st.session_state['selected_point_key'] = current_model_list[0]
        
        selected_label = st.session_state['selected_point_key']
        
        # 2B. SELEZIONE UNIFICATA (Selectbox)
        st.write("### 📊 Posizionamento MPI vs Prezzo")
        
        selected_index = current_model_list.index(selected_label)

        selected_label_input = st.selectbox(
            "🔎 **Trova ed evidenzia un modello nel grafico e nella scheda dettagli:**",
            current_model_list,
            index=selected_index,
            key='main_selectbox'
        )
        
        # Aggiornamento stato
        if selected_label_input != st.session_state['selected_point_key']:
             st.session_state['selected_point_key'] = selected_label_input
             st.rerun() 

        selected_points_labels = [st.session_state['selected_point_key']]
        
        # Mostriamo il grafico Plotly
        fig_scatter = plot_mpi_vs_price_plotly(df_val, PRICE_COL, selected_points_labels)
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 2C. CLASSIFICA VALUE INDEX
        st.write("### 🏆 Classifica Qualità/Prezzo (MPI-B / Costo)")
        cols_show = ["label", "passo", "MPI_B", PRICE_COL, "ValueIndex"]
        st.dataframe(df_val_sorted[[c for c in cols_show if c in df_val_sorted.columns]], use_container_width=True)
        
    else:
        st.info("Nessuna scarpa con prezzo e MPI-B validi nei filtri attuali per l'analisi.")
        st.stop()
else:
    st.warning("Colonna prezzo non disponibile nel dataset per l'analisi. Impossibile procedere.")
    st.stop()


# ============================================
# 3. SCHEDA DETTAGLIO (con Stelle e Valori)
# ============================================

st.markdown("---")
st.header("Step 3: Scheda Dettaglio Modello Selezionato")

selected_for_detail = st.session_state['selected_point_key']

if selected_for_detail:
    # Estraiamo i dati per la card
    try:
        row = df_filt[df_filt["label"] == selected_for_detail].iloc[0]
    except IndexError:
        st.warning("Il modello selezionato non è presente nei filtri correnti.")
        st.stop()
    
    with st.container():
        col_sx, col_dx = st.columns([1, 2])
        
        with col_sx:
            st.subheader(f"{row['marca']}")
            st.markdown(f"### {row['modello']}")
            if pd.notna(row.get('versione')):
                st.caption(f"Versione: {int(row['versione'])}")
            
            st.metric("MPI-B Score", f"{row['MPI_B']:.3f}")
            st.metric("Prezzo", f"{row[PRICE_COL]:.0f} €" if PRICE_COL else "N/A")
            
            if pd.notna(row.get('ValueIndex')):
                val_idx = float(row['ValueIndex'])
                stars = render_stars(val_idx)
                st.write("**Value Index:**")
                st.markdown(f"### {val_idx:.3f} {stars}")

        with col_dx:
            st.write("#### Caratteristiche Biomeccaniche")
            st.write(f"**Categoria:** {row['passo']} | **Peso:** {row['peso']}g")
            st.write(f"**Cluster:** {int(row['Cluster'])} – {row['ClusterDescrizione']}")
            
            st.markdown("---")
            
            val_shock = float(row['ShockIndex_calc'])
            st.write(f"**Shock Absorption:** {val_shock:.3f} / 1.0")
            st.progress(val_shock)
            
            val_energy = float(row['EnergyIndex_calc'])
            st.write(f"**Energy Return:** {val_energy:.3f} / 1.0")
            st.progress(val_energy)
            
            val_flex = float(row['FlexIndex'])
            st.write(f"**Flexibility (Score):** {val_flex:.3f} / 1.0")
            st.progress(val_flex)
            
            val_weight = float(row['WeightIndex'])
            st.write(f"**Weight Efficiency:** {val_weight:.3f} / 1.0")
            st.progress(val_weight)

# ============================================
# 4. MODELLI SIMILI (BIOMECCANICA)
# ============================================

st.markdown("---")
st.header("🔎 Modelli Alternativi Simili")
st.info("Questi modelli hanno un profilo biomeccanico (Shock, Energy, Flex, Weight) molto simile a quello selezionato, indipendentemente dal prezzo.")

# Calcoliamo le similarità usando gli indici calcolati dinamicamente
cols_simil = ["ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex"]
df_simili = trova_scarpe_simili(df_filt, selected_for_detail, cols_simil, n_simili=3)

if not df_simili.empty:
    cols = st.columns(3)
    for i, (idx, row_sim) in enumerate(df_simili.iterrows()):
        with cols[i]:
            st.markdown(f"#### {row_sim['marca']}")
            
            # FIX VERSIONE: Mostriamo modello + versione in modo chiaro
            nome_modello = f"{row_sim['modello']}"
            if pd.notna(row_sim.get('versione')):
                nome_modello += f" v{int(row_sim['versione'])}"
            
            st.write(f"**{nome_modello}**")
            
            # Mostriamo la differenza di prezzo
            diff_prezzo = row_sim[PRICE_COL] - row[PRICE_COL]
            icon_prezzo = "🔴" if diff_prezzo > 0 else "🟢"
            st.write(f"Prezzo: {row_sim[PRICE_COL]:.0f} € ({icon_prezzo} {diff_prezzo:+.0f} €)")
            
            st.write(f"MPI-B: **{row_sim['MPI_B']:.3f}**")
            st.caption(f"Distanza Biomec.: {row_sim['distanza_similitudine']:.4f}")
            
else:
    st.write("Nessun modello simile trovato nei filtri correnti.")
