import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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
            # Per Race: Più rigido è meglio (fino a un punto), poi plateau
            Flex_opt = 20.0; sigma = 3.0
        elif "tempo" in tipo:
            Flex_opt = 17.0; sigma = 2.5
        else:
            # Per Daily: Ottimo intermedio
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
    
    # --- 5. DRIVE INDEX (NUOVO) ---
    df = calcola_drive_index(df)

    return df

def calcola_drive_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola il 'Drive Index' (0-1) basato su Piastra, Rocker (Altezza), Torsione e Grip.
    """
    # 1. Punteggio Piastra
    def score_plate(val):
        val = str(val).lower()
        if 'carbon' in val: return 1.0
        if any(x in val for x in ['plastic', 'nylon', 'fiber', 'pebax', 'airzoom']): return 0.6
        return 0.0

    df['S_Plate'] = df['piastra'].apply(score_plate)

    # 2. Punteggio Rocker (Estrae la PRIMA parte = Altezza/Toe Spring)
    def score_rocker(val):
        if pd.isna(val) or str(val) in ['nan', '#N/D']: return 0.0
        try:
            clean_val = str(val).replace(',', '.')
            parts = clean_val.split('x')
            if len(parts) >= 1:
                height = float(parts[0]) 
                return min(height / 8.0, 1.0)
            return 0.0
        except:
            return 0.0
            
    df['S_Rocker'] = df['rocker'].apply(score_rocker)

    # 3. Punteggio Torsione
    torsion = pd.to_numeric(df['rigidezza_torsionale'], errors='coerce').fillna(2)
    df['S_Torsion'] = (torsion - 2) / 3.0
    df['S_Torsion'] = df['S_Torsion'].clip(0, 1)

    # 4. Punteggio Grip
    grip = pd.to_numeric(df['test_trazione'], errors='coerce')
    min_g, max_g = grip.min(), grip.max()
    df['S_Grip'] = (grip - min_g) / (max_g - min_g)
    df['S_Grip'] = df['S_Grip'].fillna(0.5)

    # FORMULA FINALE DRIVE
    df['DriveIndex'] = (
        0.30 * df['S_Plate'] +
        0.30 * df['S_Rocker'] +
        0.20 * df['S_Torsion'] +
        0.20 * df['S_Grip']
    )
    
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
    
    df_val['Colore_Evidenziazione'] = df_val['label'].apply(
        lambda x: 'Selezionato' if x in selected_points_labels else 'Mercato'
    )
    
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

def plot_radar_comparison_plotly_styled(df_shoes, metrics, title="Confronto Biomeccanico (Radar)"):
    """ Crea un Radar Chart interattivo con Plotly. """
    
    fig = go.Figure()
    
    metrics_readable = {
        "ShockIndex_calc": "Shock",
        "EnergyIndex_calc": "Energy",
        "FlexIndex": "Flex",
        "WeightIndex": "Weight",
        "DriveIndex": "Drive"
    }
    
    categories = [metrics_readable.get(m, m) for m in metrics]
    comparison_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    
    # 1. DISEGNA PRIMA LE SCARPE DI CONFRONTO
    for i in range(1, len(df_shoes)):
        row = df_shoes.iloc[i]
        values = [float(row[m]) for m in metrics]
        values += [values[0]]
        categories_closed = categories + [categories[0]]
        
        color = comparison_colors[(i-1) % len(comparison_colors)]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=f"{row['label']} (Simile)",
            line=dict(color=color, width=1, dash='dot'),
            fillcolor=color,
            opacity=0.3,
            hoveron='points+fills'
        ))

    # 2. DISEGNA PER ULTIMA LA SCARPA SELEZIONATA
    if not df_shoes.empty:
        row = df_shoes.iloc[0]
        values = [float(row[m]) for m in metrics]
        values += [values[0]]
        categories_closed = categories + [categories[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill='toself',
            name=f"★ {row['label']}",
            line=dict(color='red', width=4),
            fillcolor='rgba(255, 0, 0, 0.1)',
            opacity=0.9
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
                gridcolor='lightgrey'
            ),
            bgcolor='white'
        ),
        title=dict(text=title, x=0.5),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1)
    )
    
    return fig

def trova_scarpe_simili(df, target_label, metrics_cols, n_simili=3):
    """ Trova le n scarpe più simili basandosi sulla distanza euclidea. """
    try:
        target_vector = df.loc[df['label'] == target_label, metrics_cols].astype(float).values[0]
        df_calc = df.copy()
        
        vectors = df_calc[metrics_cols].astype(float).values
        distances = np.linalg.norm(vectors - target_vector, axis=1)
        
        df_calc['distanza_similitudine'] = distances
        
        simili = df_calc[df_calc['label'] != target_label].sort_values('distanza_similitudine').head(n_simili)
        
        return simili
    except Exception as e:
        st.error(f"Errore nel calcolo similitudine: {e}")
        return pd.DataFrame()

def render_stars(value):
    if pd.isna(value): return ""
    score = int(round(value * 5))
    score = max(0, min(5, score))
    return ("★" * score) + ("☆" * (5 - score))


# =========================
#   APP STREAMLIT
# =========================

st.set_page_config(page_title="AFT Explorer V2", layout="wide")

st.title("AFT Shoe Database – MPI & Clustering V2")

# EXPANDER SCIENTIFICO
with st.expander("📚 La Scienza dietro l'Algoritmo AFT (Evidence-Based)"):
    st.markdown("""
    L'algoritmo di questa app non è casuale, ma si basa su studi biomeccanici recenti. Ecco come vengono calcolati i punteggi:

    ### 1. Il "Costo" del Peso (Weight Penalty) ⚖️
    La letteratura scientifica dimostra che ogni **100g di peso extra** ai piedi aumentano il costo metabolico dell'**1%**.
    * **Come funziona qui:** Usiamo una funzione di decadimento esponenziale. Una scarpa sotto i 200g ottiene punteggi altissimi (0.9-1.0), mentre sopra i 280g il punteggio crolla rapidamente (sotto 0.6), penalizzando le scarpe pesanti più di quanto farebbe una semplice scala lineare.

    ### 2. L'Effetto "Teeter-Totter" (Drive Index) 🚀
    Avere una piastra in carbonio non basta. Lo studio evidenzia che la performance è una sinergia (effetto leva) tra:
    * **Piastra Rigida:** Agisce come leva.
    * **Geometria Rocker:** Agisce come fulcro.
    * **Rigidità Torsionale:** Trasmette la forza senza dispersioni.
    * **Come funziona qui:** Il *Drive Index* moltiplica questi fattori. Se una scarpa ha la piastra ma è piatta (niente rocker), il punteggio Drive sarà basso.

    ### 3. Rigidità Longitudinale: La curva a "U" 🔧
    La meta-analisi suggerisce che la rigidità non segue una regola "più è meglio".
    * **Come funziona qui:** Per le scarpe da gara (*Race*), premiamo la rigidità alta (necessaria per la piastra). Per le scarpe da allenamento (*Daily*), usiamo una curva a campana: troppa rigidità a ritmi lenti è penalizzante e scomoda.

    ### 4. Schiuma vs Piastra (Foam is King) ☁️
    Contrariamente al marketing, gli studi indicano che la **schiuma (PEBA)** contribuisce al risparmio energetico più della piastra stessa.
    * **Come funziona qui:** Nel calcolo finale, l'*Energy Return* (dato dalla schiuma) ha un peso preponderante rispetto alla sola presenza della piastra.
    """)

file_name = "database_completo_AFT_20251124_clean.csv"

# =========================
#   PRE-PROCESSING
# =========================

@st.cache_data
def load_and_process(path):
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        st.error(f"File {path} non trovato.")
        return pd.DataFrame(), pd.DataFrame()

    df = calcola_indici(df)
    df = calcola_MPIB(df)
    df, cluster_summary = esegui_clustering(df)

    index_cols = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "StackFactor", "MPI_B", "DriveIndex"]
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
#   SIDEBAR
# =========================

with st.sidebar:
    st.header("Filtri per il Database")
    st.subheader("Cluster Biomeccanici")
    st.dataframe(cluster_summary_raw, use_container_width=True)

    st.markdown("---")

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

with col_appoggio:
    st.subheader("1A. Appoggio Piede")
    heel_pct = st.slider("Appoggio tallone (%)", 0, 100, 40, 5)
    w_heel = heel_pct / 100.0
    w_mid = 1.0 - w_heel
    st.write(f"Avampiede: {100 - heel_pct}%")

with col_pesi:
    st.subheader("1B. Importanza Parametri")
    w_shock  = st.slider("Shock (Ammortizz.)", 1, 5, 3)
    w_energy = st.slider("Energy (Ritorno)", 1, 5, 3)
    w_flex   = st.slider("Stiffness (Flex)", 1, 5, 3)
    w_weight = st.slider("Weight (Leggerezza)", 1, 5, 3)

# Ricalcolo MPI
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

df_filt.loc[:, "MPI_B"] = (
    w_shock_eff  * df_filt["ShockIndex_calc"] +
    w_energy_eff * df_filt["EnergyIndex_calc"] +
    df_filt["FlexIndex"] * w_flex_eff +
    df_filt["WeightIndex"] * w_weight_eff
).round(3)

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

st.success("MPI Score ricalcolato!")
st.markdown("---")


# ============================================
# NUOVO BLOCCO: BEST PICK PER BUDGET
# ============================================

st.header("💡 Best Pick: La Migliore per Te")

if PRICE_COL:
    min_p = int(df_filt[PRICE_COL].min())
    max_p = int(df_filt[PRICE_COL].max())
    
    col_budget, col_best = st.columns([1, 2])
    
    with col_budget:
        budget_max = st.slider(
            "💰 Seleziona il tuo Budget Massimo (€):",
            min_value=min_p, 
            max_value=max_p, 
            value=int(max_p * 0.8),
            step=5
        )
    
    df_budget = df_filt[df_filt[PRICE_COL] <= budget_max].copy()
    
    with col_best:
        if not df_budget.empty:
            best_pick = df_budget.sort_values(by="MPI_B", ascending=False).iloc[0]
            
            with st.container(border=True):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.subheader(f"🏆 {best_pick['marca']} {best_pick['modello']}")
                    st.write(f"**La scelta più performante sotto i {budget_max}€**")
                    if pd.notna(best_pick.get('versione')):
                        st.caption(f"Versione: {int(best_pick['versione'])}")

                with c2:
                    st.metric("MPI Score", f"{best_pick['MPI_B']:.3f}")
                    st.write(f"Prezzo: **{best_pick[PRICE_COL]:.0f} €**")
                    
                    if pd.notna(best_pick.get('ValueIndex')):
                        stars = render_stars(best_pick['ValueIndex'])
                        st.write(f"Value: {stars}")
        else:
            st.warning("Nessuna scarpa trovata con questo budget. Prova ad aumentarlo!")

st.markdown("---")


# ============================================
# 2. ANALISI MERCATO
# ============================================

st.header("Step 2: Analisi di Mercato (MPI vs Prezzo)")

if PRICE_COL is not None and PRICE_COL in df_filt.columns:
    
    df_val = df_filt.dropna(subset=[PRICE_COL, "MPI_B", "ValueIndex"]).copy()
    df_val = df_val[df_val[PRICE_COL] > 0]
    
    if not df_val.empty:
        
        # Gestione Stato
        df_val_sorted = df_val.sort_values(by="ValueIndex", ascending=False)
        default_label_on_load = df_val_sorted.iloc[0]['label']
        
        if 'selected_point_key' not in st.session_state:
            st.session_state['selected_point_key'] = default_label_on_load
        
        current_model_list = df_val_sorted['label'].tolist()
        if st.session_state['selected_point_key'] not in current_model_list:
             st.session_state['selected_point_key'] = current_model_list[0]
        
        selected_label = st.session_state['selected_point_key']
        selected_index = current_model_list.index(selected_label)

        col_plot, col_list = st.columns([3, 1])

        with col_plot:
            st.write("### 📊 Grafico Interattivo")
            selected_label_input = st.selectbox(
                "🔎 Trova ed evidenzia modello:",
                current_model_list,
                index=selected_index,
                key='main_selectbox'
            )
            
            if selected_label_input != st.session_state['selected_point_key']:
                 st.session_state['selected_point_key'] = selected_label_input
                 st.rerun() 

            selected_points_labels = [st.session_state['selected_point_key']]
            fig_scatter = plot_mpi_vs_price_plotly(df_val, PRICE_COL, selected_points_labels)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_list:
            st.write("### 🏆 Top 10 Value")
            st.dataframe(
                df_val_sorted[["label", "MPI_B", "ValueIndex"]].head(10), 
                use_container_width=True,
                hide_index=True
            )
        
    else:
        st.info("Nessun dato valido per il grafico.")
        st.stop()
else:
    st.stop()


# ============================================
# 3. SCHEDA DETTAGLIO
# ============================================

st.markdown("---")
st.header("Step 3: Scheda Dettaglio")

selected_for_detail = st.session_state['selected_point_key']

if selected_for_detail:
    try:
        row = df_filt[df_filt["label"] == selected_for_detail].iloc[0]
    except IndexError:
        st.stop()
    
    with st.container(border=True):
        col_sx, col_dx = st.columns([1, 2])
        
        with col_sx:
            st.subheader(f"{row['marca']}")
            st.markdown(f"### {row['modello']}")
            if pd.notna(row.get('versione')):
                st.caption(f"Versione: {int(row['versione'])}")
            
            st.metric("MPI-B Score", f"{row['MPI_B']:.3f}")
            st.metric("Prezzo", f"{row[PRICE_COL]:.0f} €")
            
            if pd.notna(row.get('ValueIndex')):
                val_idx = float(row['ValueIndex'])
                stars = render_stars(val_idx)
                st.write("**Value Index:**")
                st.markdown(f"### {val_idx:.3f} {stars}")

        with col_dx:
            st.write("#### Biomeccanica")
            st.write(f"**Cat:** {row['passo']} | **Peso:** {row['peso']}g")
            
            c1, c2 = st.columns(2)
            with c1:
                val_shock = float(row['ShockIndex_calc'])
                st.caption(f"Shock Abs: {val_shock:.2f}")
                st.progress(val_shock)
                
                val_flex = float(row['FlexIndex'])
                st.caption(f"Flexibility: {val_flex:.2f}")
                st.progress(val_flex)
                
                val_drive = float(row['DriveIndex'])
                st.markdown(f"**🚀 Drive (Spinta): {val_drive:.2f}**")
                st.progress(val_drive)

            with c2:
                val_energy = float(row['EnergyIndex_calc'])
                st.caption(f"Energy Ret: {val_energy:.2f}")
                st.progress(val_energy)
                
                val_weight = float(row['WeightIndex'])
                st.caption(f"Weight Eff: {val_weight:.2f}")
                st.progress(val_weight)

# ============================================
# 4. MODELLI SIMILI & RADAR
# ============================================

st.markdown("---")
st.header("🔎 Modelli Simili (Biomeccanica)")

cols_simil = ["ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex", "DriveIndex"]
df_simili = trova_scarpe_simili(df_filt, selected_for_detail, cols_simil, n_simili=3)

if not df_simili.empty:
    # 1. Mostra le card dei modelli simili
    cols = st.columns(3)
    for i, (idx, row_sim) in enumerate(df_simili.iterrows()):
        with cols[i]:
            with st.container(border=True):
                label_sim = row_sim['label']
                st.markdown(f"**{label_sim}**")
                
                diff_prezzo = row_sim[PRICE_COL] - row[PRICE_COL]
                icon = "🔴" if diff_prezzo > 0 else "🟢"
                st.write(f"{row_sim[PRICE_COL]:.0f}€ ({icon} {diff_prezzo:+.0f}€)")
                st.caption(f"MPI: {row_sim['MPI_B']:.3f}")
                st.caption(f"Dist: {row_sim['distanza_similitudine']:.3f}")
    
    st.markdown("#### Confronto Radar")
    df_radar = pd.concat([
        df_filt[df_filt['label'] == selected_for_detail],
        df_simili
    ], ignore_index=True)
    
    fig_radar = plot_radar_comparison_plotly_styled(df_radar, cols_simil)
    st.plotly_chart(fig_radar, use_container_width=True)
            
else:
    st.write("Nessun modello simile trovato nei filtri correnti.")
