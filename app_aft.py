import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
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


def plot_radar_indices(df_comp: pd.DataFrame, metrics: list, label_col="label"):
    """ 
    Grafico Radar Matplotlib.
    FIX: Legge e casta i dati a float scalarmente per prevenire l'errore strutturale.
    """
    import numpy as np
    
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    for _, row in df_comp.iterrows():
        # --- CORREZIONE: Esegui il casting a float scalarmente ---
        try:
            # L'uso di float(row[m]) è il metodo più sicuro per estrarre il valore scalare
            values = [float(row[m]) for m in metrics]
        except (ValueError, TypeError) as e:
            # Se ci sono dati non numerici, salta questa riga
            print(f"Errore di conversione nel Radar Chart per riga {row[label_col]}: {e}")
            continue 
        # --------------------------------------------------------

        values = values + [values[0]]
        label = row[label_col]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.15)

    ax.set_thetagrids(angles[:-1] * 180 / np.pi, metrics)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig


def plot_mpi_vs_price_plotly(df_val, price_col, selected_points_labels):
    """ Scatter plot MPI-B vs Prezzo usando Plotly per l'interattività (hover e click). """
    
    # Crea la colonna per l'evidenziazione
    df_val['Colore_Evidenziazione'] = df_val['label'].apply(
        lambda x: 'Selezionato' if x in selected_points_labels else 'Mercato'
    )
    
    # Ordina per portare i punti selezionati in primo piano nel grafico
    df_val = df_val.sort_values(by='Colore_Evidenziazione', ascending=False).reset_index(drop=True)
    
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
        hover_name='hover_text', # Usiamo il testo personalizzato nell'hover
        color_discrete_map={
            'Selezionato': 'red',
            'Mercato': 'gray'
        },
        custom_data=['label'],
        labels={price_col: f'{price_col} (€)', "MPI_B": "MPI-B Score"},
        title="MPI-B Score vs. Prezzo (Performance vs. Costo)"
    )

    fig.update_traces(
        marker=dict(
            size=10,
            opacity=0.7,
            line=dict(width=1, color='black')
        ),
        selector=dict(mode='markers')
    )
    
    fig.update_layout(
        hovermode="closest",
        yaxis=dict(range=[0, 1.05]),
        legend_title_text='Punti Dati'
    )
    
    fig.add_annotation(
        x=df_val[price_col].mean(), 
        y=1.05,
        text="Clicca su un punto per selezionarlo per il Dettaglio/Confronto!",
        showarrow=False,
        font=dict(size=12, color="blue")
    )

    return fig


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

# Ricalcolo MPI dinamico (basato sugli input del wizard)
raw_weights = np.array([w_shock, w_energy, w_flex, w_weight], dtype=float)
tot = raw_weights.sum() if raw_weights.sum() > 0 else 1.0
norm_weights = raw_weights / tot
w_shock_eff, w_energy_eff, w_flex_eff, w_weight_eff = norm_weights

# Ricalcolo Shock/Energy (basato sull'appoggio)
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
# 2. RISULTATI GLOBALI INTERATTIVI (Grafico e Classifica)
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
        
        # FIX: Se il modello precedentemente selezionato non è nel nuovo filtro, resettiamo il default.
        if st.session_state['selected_point_key'] not in df_val_sorted['label'].tolist():
             st.session_state['selected_point_key'] = default_label_on_load
        
        selected_label = st.session_state['selected_point_key']
        
        # 2B. SELEZIONE UNIFICATA (Selectbox e Grafico)
        st.write("### 📊 Posizionamento MPI vs Prezzo")
        
        # Trova l'indice del modello corrente per preimpostare correttamente la selectbox
        # FIX: Usiamo l'approccio condizionale corretto per l'indice
        model_list = df_val_sorted['label'].tolist()
        
        if selected_label in model_list:
            selected_index = model_list.index(selected_label)
        else:
            selected_index = 0
            # Se siamo qui, selected_label non era valido, ma lo stato è stato resettato sopra,
            # quindi selected_label dovrebbe essere model_list[0] e l'indice è 0.

        selected_label_input = st.selectbox(
            "Seleziona un modello per il Dettaglio (o clicca sul grafico per cambiarlo):",
            model_list,
            index=selected_index # Usa l'indice trovato
        )
        
        # Aggiorna lo stato se l'utente cambia la selectbox
        if selected_label_input != st.session_state['selected_point_key']:
             st.session_state['selected_point_key'] = selected_label_input
             st.rerun() # Ricarica per aggiornare Step 3

        selected_points_labels = [st.session_state['selected_point_key']]
        
        fig_scatter = plot_mpi_vs_price_plotly(df_val, PRICE_COL, selected_points_labels)
        
        # Utilizzo dell'API standard di Streamlit per catturare la selezione al click
        plotly_event = st.plotly_chart(
            fig_scatter, 
            use_container_width=True, 
            selection_mode="single",
            key='mpi_scatter_chart' 
        )

        # CATTURA DELL'EVENTO DI SELEZIONE TRAMITE SESSION_STATE
        
        selection_data_state = st.session_state.get('mpi_scatter_chart')

        # Analizza se sono stati selezionati punti
        if selection_data_state and selection_data_state.get('selection'):
            selection_points = selection_data_state['selection'].get('points')
            
            if selection_points and selection_points[0].get('customdata'):
                new_selection = selection_points[0]['customdata'][0]
                
                # Aggiorno lo stato se la selezione è cambiata
                if new_selection != st.session_state['selected_point_key']:
                    st.session_state['selected_point_key'] = new_selection
                    st.rerun() # Ricarica per aggiornare Step 3

        # 2C. CLASSIFICA VALUE INDEX
        st.write("### 🏆 Classifica Qualità/Prezzo (MPI-B / Costo)")
        cols_show = ["label", "passo", "MPI_B", PRICE_COL, "ValueIndex"]
        st.dataframe(df_val_sorted[[c for c in cols_show if c in df_val_sorted.columns]], use_container_width=True)

        st.markdown("---")
        
    else:
        st.info("Nessuna scarpa con prezzo e MPI-B validi nei filtri attuali per l'analisi.")
        st.stop()
else:
    st.warning("Colonna prezzo non disponibile nel dataset per l'analisi. Impossibile procedere.")
    st.stop()


# ============================================
# 3. ANALISI DI DETTAGLIO E CONFRONTO
# ============================================

st.header("Step 3: Analisi di Dettaglio e Confronto")

selected_for_detail = st.session_state['selected_point_key']
    
st.info(f"Stai analizzando il modello: **{selected_for_detail if selected_for_detail else 'Nessun modello selezionato'}**")


if selected_for_detail:
    scarpa = df_filt[df_filt["label"] == selected_for_detail].iloc[0]
    
    st.subheader(f"🔬 Dettaglio: {selected_for_detail}")

    # --- INPUT CONFRONTO (Multi-select) ---
    default_comparison = [selected_for_detail]
    
    # ⚠️ FIX: Uso una chiave unica per la multiselect e riuso il valore dello stato 
    # per garantire che la selezione non si 'perda' dopo il rerun.
    if 'comparison_models' not in st.session_state:
        st.session_state['comparison_models'] = default_comparison

    selezione_confronto = st.multiselect(
        "Seleziona altri modelli per il Radar Chart",
        df_filt["label"].tolist(),
        max_selections=5,
        default=default_comparison,
        key='radar_multiselect'
    )
    
    # In questo punto, 'selezione_confronto' è il valore aggiornato.

    col_dettaglio, col_confronto_radar = st.columns([1, 2])

    # --- 3A. DETTAGLIO SCARPA (Visualizzazione) ---
    with col_dettaglio:
        st.subheader("Informazioni Base")
        st.markdown(f"**Marca:** {scarpa['marca']}")
        st.markdown(f"**Modello:** {scarpa['modello']}")
        if "versione" in scarpa and pd.notna(scarpa["versione"]):
            st.write(f"Versione: {int(scarpa['versione'])}")
        st.write(f"Passo / categoria (AFT): {scarpa['passo']}")
        st.write(f"Peso: {scarpa['peso']} g")

        if PRICE_COL is not None and pd.notna(scarpa[PRICE_COL]):
            st.write(f"Prezzo: {scarpa[PRICE_COL]:.0f} €")
        
        st.markdown("---")
        st.metric("MPI-B Score", f"{scarpa['MPI_B']:.3f}")
        if "ValueIndex" in scarpa.index and pd.notna(scarpa["ValueIndex"]):
            st.write(f"Value index (0–1): {scarpa['ValueIndex']:.3f}")
        
        st.markdown("---")
        st.write("**Cluster:**")
        st.write(f"Cl. {int(scarpa['Cluster'])}: {scarpa['ClusterDescrizione']}")

    # --- 3B. RADAR CHART (Visualizzazione) ---
with col_confronto_radar:
    st.subheader("Analisi Biomeccanica (Indici 0-1)")
    
    # ⚠️ FIX LOGICO: Controlliamo se la lista selezionata ha almeno un elemento
    if selezione_confronto:
        
        df_comp = df_filt[df_filt["label"].isin(selezione_confronto)].copy()
        df_comp = df_comp.reset_index(drop=True) 

        # Rinominiamo le colonne calcolate nel DataFrame TEMPORANEO per il plot
        df_comp = df_comp.rename(columns={
            "ShockIndex_calc": "ShockIndex",
            "EnergyIndex_calc": "EnergyIndex"
        })
        
        metrics_plot = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex"]
        
        # Verifichiamo che tutte le colonne siano presenti e che il DataFrame non sia vuoto
        if all(m in df_comp.columns for m in metrics_plot) and not df_comp.empty:
            
            fig = plot_radar_indices(df_comp, metrics_plot, label_col="label")
            st.pyplot(fig)
        else:
            # Caso: Selezione presente, ma i dati filtrati sono incompleti o non numerici
            st.info("Dati per il Radar Chart incompleti o non numerici.")
    else:
        # Caso: La lista di selezione è vuota
        st.warning("Seleziona almeno un modello per visualizzare il Radar Chart.")


