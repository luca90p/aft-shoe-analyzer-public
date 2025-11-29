import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# =========================
#   CONFIGURAZIONE PAGINA
# =========================
st.set_page_config(page_title="AFT Analyst", layout="wide")

# =========================
#   SISTEMA DI LOGIN
# =========================
def check_password():
    """Ritorna `True` se l'utente ha inserito la password corretta."""
    ACTUAL_PASSWORD = "aft" 

    def password_entered():
        if st.session_state["password"] == ACTUAL_PASSWORD:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "🔒 Inserisci la Password di accesso:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        st.text_input(
            "🔒 Inserisci la Password di accesso:", 
            type="password", 
            on_change=password_entered, 
            key="password"
        )
        st.error("😕 Password errata. Riprova.")
        return False
    else:
        return True

if check_password():

    # =========================
    #   FUNZIONI AFT
    # =========================

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
        df_val['Status'] = df_val['label'].apply(
            lambda x: 'Selezionato' if x in selected_points_labels else 'Database'
        )
        df_val = df_val.sort_values(by='Status', ascending=True).reset_index(drop=True)
        
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
            color='Status',
            size='ValueIndex',
            size_max=20,
            hover_name='hover_text',
            color_discrete_map={'Selezionato': '#FF4B4B', 'Database': '#A9A9A9'},
            custom_data=['label'],
            labels={price_col: f'{price_col} [€]', "MPI_B": "Indice MPI-B"},
            title="Analisi Costo-Efficienza (MPI vs Prezzo)"
        )

        fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='#333333')))
        fig.update_layout(
            hovermode="closest",
            yaxis=dict(title="Performance Index (MPI)", range=[0, 1.05]),
            xaxis=dict(title="Prezzo di Listino (€)"),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    def plot_radar_comparison_plotly_styled(df_shoes, metrics, title="Analisi Comparativa Radar"):
        fig = go.Figure()
        metrics_readable = {
            "ShockIndex_calc": "Shock Abs.",
            "EnergyIndex_calc": "Energy Ret.",
            "FlexIndex": "Flexibility",
            "WeightIndex": "Weight Eff.",
            "DriveIndex": "Drive Mech."
        }
        categories = [metrics_readable.get(m, m) for m in metrics]
        comparison_colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd'] 
        
        for i in range(1, len(df_shoes)):
            row = df_shoes.iloc[i]
            values = [float(row[m]) for m in metrics]
            values += [values[0]]
            categories_closed = categories + [categories[0]]
            color = comparison_colors[(i-1) % len(comparison_colors)]
            
            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories_closed, fill='toself',
                name=f"{row['label']} (Simile)",
                line=dict(color=color, width=1.5, dash='dot'),
                fillcolor=color, opacity=0.15
            ))

        if not df_shoes.empty:
            row = df_shoes.iloc[0]
            values = [float(row[m]) for m in metrics]
            values += [values[0]]
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values, theta=categories_closed, fill='toself',
                name=f"★ {row['label']} (Target)",
                line=dict(color='#FF4B4B', width=3),
                fillcolor='rgba(255, 75, 75, 0.1)', opacity=0.9
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title=dict(text=title, x=0.5),
            showlegend=True,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        return fig

    def trova_scarpe_simili(df, target_label, metrics_cols, n_simili=3):
        try:
            target_vector = df.loc[df['label'] == target_label, metrics_cols].astype(float).values[0]
            df_calc = df.copy()
            vectors = df_calc[metrics_cols].astype(float).values
            distances = np.linalg.norm(vectors - target_vector, axis=1)
            df_calc['distanza_similitudine'] = distances
            simili = df_calc[df_calc['label'] != target_label].sort_values('distanza_similitudine').head(n_simili)
            return simili
        except Exception:
            return pd.DataFrame()

    def render_stars(value):
        if pd.isna(value): return ""
        score = int(round(value * 5))
        score = max(0, min(5, score))
        return ("★" * score) + ("☆" * (5 - score))

    # =========================
    #   UI PRINCIPALE
    # =========================
    st.title("Database AFT: Analisi Biomeccanica e Clustering")
    st.markdown("**Advanced Footwear Technology Analysis Tool**")

    with st.expander("📘 Metodologia e Riferimenti Bibliografici"):
        st.markdown("""
        **1. Costo Metabolico del Peso**
        Ogni 100g extra aumentano il costo energetico dell'1%.
        *Fonte:* [metabolic cost of running, body weight influence.pdf]
        
        **2. Indice di Spinta Meccanica (Drive Index)**
        Sinergia tra Piastra, Rocker e Rigidità.
        *Fonte:* [Effects of the curved carbon fibre plate...pdf]
        
        **3. Rigidità Longitudinale (Flex Index)**
        Range misurato: 5N (Soft) - 40N (Stiff).
        *Fonte:* [The eﬀects of footwear midsole longitudinal bending...pdf]
        """)

    with st.expander("📐 Formule Matematiche del Modello AFT"):
        st.markdown(r"""
        ### 1. Flex Index ($I_{Flex}$) - Range 5-40 N
        * **Race:** Sigmoide centrata su 18N.
        * **Daily:** Gaussiana centrata su 12N.

        ### 2. Drive Index ($I_{Drive}$)
        $$ I_{Drive} = 0.6 \cdot (S_{Plate} \cdot S_{Rocker} \cdot S_{Stiffness}) + 0.4 \cdot S_{Foam} $$
        Dove $S_{Stiffness}$ è normalizzato su 30N.
        """)

    file_name = "database_completo_AFT_20251124_clean.csv"

    @st.cache_data
    def load_and_process(path):
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            st.error("File database non trovato.")
            return pd.DataFrame(), pd.DataFrame()

        df = calcola_indici(df)
        df = calcola_MPIB(df)
        df, cluster_summary = esegui_clustering(df)
        
        # Arrotondamento
        cols_to_round = ["ShockIndex", "EnergyIndex", "FlexIndex", "WeightIndex", "MPI_B", "DriveIndex"]
        for c in cols_to_round: 
            if c in df.columns: df[c] = df[c].round(3)

        price_cols = [c for c in df.columns if "prezzo" in c.lower()]
        if price_cols:
            df[price_cols[0]] = df[price_cols[0]].astype(float).round(0)
            
        df["label"] = df.apply(lambda r: f"{r['marca']} {r['modello']} v{int(r['versione'])}" if pd.notna(r.get('versione')) else f"{r['marca']} {r['modello']}", axis=1)
        return df, cluster_summary

    df_raw, cluster_summary_raw = load_and_process(file_name)
    if df_raw.empty: st.stop()

    df = df_raw.copy()
    PRICE_COL = [c for c in df.columns if "prezzo" in c.lower()]
    PRICE_COL = PRICE_COL[0] if PRICE_COLS else None

    with st.sidebar:
        st.header("Filtri Database")
        st.dataframe(cluster_summary_raw, use_container_width=True)
        st.markdown("---")
        
        all_brands = ["Tutte"] + sorted(df["marca"].unique())
        sel_brand = st.selectbox("Marca", all_brands)
        all_passi = ["Tutti"] + sorted(df["passo"].unique())
        sel_passo = st.selectbox("Categoria", all_passi)
        
        df_filt = df.copy()
        if sel_brand != "Tutte": df_filt = df_filt[df_filt["marca"] == sel_brand]
        if sel_passo != "Tutti": df_filt = df_filt[df_filt["passo"] == sel_passo]

    st.header("1. Parametrizzazione Performance (MPI)")
    c1, c2 = st.columns(2)
    with c1:
        heel_pct = st.slider("Appoggio Tallone (%)", 0, 100, 40, 5)
        w_heel = heel_pct/100
        w_mid = 1.0 - w_heel
    with c2:
        w_shock = st.slider("Shock", 1, 5, 3)
        w_energy = st.slider("Energy", 1, 5, 3)
        w_flex = st.slider("Stiffness", 1, 5, 3)
        w_weight = st.slider("Weight", 1, 5, 3)

    # Ricalcolo dinamico
    def safe_norm(s): return (s - s.min()) / max(s.max() - s.min(), 1e-9)
    
    df_filt["ShockIndex_calc"] = safe_norm(w_heel * df_filt["shock_abs_tallone"] + w_mid * df_filt["shock_abs_mesopiede"])
    df_filt["EnergyIndex_calc"] = safe_norm(w_heel * df_filt["energy_ret_tallone"] + w_mid * df_filt["energy_ret_mesopiede"])
    
    tot_w = w_shock + w_energy + w_flex + w_weight
    df_filt["MPI_B"] = (
        (w_shock * df_filt["ShockIndex_calc"] + 
         w_energy * df_filt["EnergyIndex_calc"] + 
         w_flex * df_filt["FlexIndex"] + 
         w_weight * df_filt["WeightIndex"]) / tot_w
    ).round(3)

    if PRICE_COL:
        df_filt = df_filt[df_filt[PRICE_COL] > 0].copy()
        v_raw = df_filt["MPI_B"] / df_filt[PRICE_COL]
        df_filt["ValueIndex"] = ((v_raw - v_raw.min()) / (v_raw.max() - v_raw.min())).round(3)
    else:
        df_filt["ValueIndex"] = 0.0

    # Best Pick
    st.markdown("---")
    st.header("💡 Best Pick")
    if PRICE_COL:
        b_max = st.slider("Budget Max (€)", int(df_filt[PRICE_COL].min()), int(df_filt[PRICE_COL].max()), 200, 5)
        picks = df_filt[df_filt[PRICE_COL] <= b_max].sort_values("MPI_B", ascending=False)
        if not picks.empty:
            bp = picks.iloc[0]
            with st.container(border=True):
                k1, k2 = st.columns([3, 1])
                k1.subheader(f"🏆 {bp['label']}")
                k2.metric("MPI", f"{bp['MPI_B']}")
                k2.write(f"{bp[PRICE_COL]:.0f} €")
    
    # Analisi Mercato
    st.markdown("---")
    st.header("2. Analisi Mercato")
    
    if not df_filt.empty:
        df_val_sorted = df_filt.sort_values("ValueIndex", ascending=False)
        models = df_val_sorted['label'].tolist()
        
        if 'selected_point_key' not in st.session_state or st.session_state['selected_point_key'] not in models:
            st.session_state['selected_point_key'] = models[0]
            
        curr_sel = st.session_state['selected_point_key']
        idx_sel = models.index(curr_sel)
        
        sel_input = st.selectbox("Seleziona Modello:", models, index=idx_sel, key='main_sb')
        if sel_input != curr_sel:
            st.session_state['selected_point_key'] = sel_input
            st.rerun()
            
        c_plot, c_list = st.columns([3, 1])
        with c_plot:
            st.plotly_chart(plot_mpi_vs_price_plotly(df_filt, PRICE_COL, [sel_input]), use_container_width=True)
        with c_list:
            st.dataframe(df_val_sorted[["label", "MPI_B", "ValueIndex"]].head(10), use_container_width=True, hide_index=True)

        # Dettaglio
        st.markdown("---")
        st.header("3. Scheda Tecnica")
        row = df_filt[df_filt["label"] == sel_input].iloc[0]
        
        with st.container(border=True):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.subheader(row['marca'])
                st.markdown(f"**{row['modello']}**")
                st.metric("MPI", row['MPI_B'])
                st.write(f"**Value:** {row['ValueIndex']} {render_stars(row['ValueIndex'])}")
            with c2:
                st.write(f"Peso: {row['peso']}g | Cluster: {row['ClusterDescrizione']}")
                colA, colB = st.columns(2)
                colA.progress(row['ShockIndex_calc'], text=f"Shock: {row['ShockIndex_calc']:.2f}")
                colB.progress(row['EnergyIndex_calc'], text=f"Energy: {row['EnergyIndex_calc']:.2f}")
                colA.progress(row['FlexIndex'], text=f"Flex: {row['FlexIndex']:.2f}")
                colB.progress(row['DriveIndex'], text=f"Drive: {row['DriveIndex']:.2f}")

        # Simili
        st.markdown("---")
        st.header("4. Similitudine & Radar")
        cols_sim = ["ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex", "DriveIndex"]
        simili = trova_scarpe_simili(df_filt, sel_input, cols_sim)
        
        if not simili.empty:
            cc = st.columns(3)
            for i, (_, s_row) in enumerate(simili.iterrows()):
                with cc[i]:
                    with st.container(border=True):
                        st.markdown(f"**{s_row['label']}**")
                        st.caption(f"Dist: {s_row['distanza_similitudine']:.3f}")
            
            df_rad = pd.concat([df_filt[df_filt['label']==sel_input], simili], ignore_index=True)
            st.plotly_chart(plot_radar_comparison_plotly_styled(df_rad, cols_sim), use_container_width=True)
    else:
        st.warning("Nessun dato con i filtri attuali.")
        
    # ============================================
    # 5. TABELLA DI CONTROLLO (NUOVA SEZIONE)
    # ============================================
    st.markdown("---")
    with st.expander("📊 Tabella di Controllo Completa (Tutti gli Indici)"):
        st.info("Questa tabella mostra i valori esatti di tutti gli indici calcolati per ogni scarpa nel filtro corrente.")
        
        # Selezioniamo le colonne di interesse
        # Nota: 'ShockIndex_calc' e 'EnergyIndex_calc' sono quelli dinamici usati per MPI
        cols_ctrl_base = ["label", "MPI_B", "ValueIndex", "DriveIndex", "StackFactor"]
        cols_indices = ["ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex"]
        
        # Uniamo e verifichiamo esistenza
        all_cols = cols_ctrl_base + cols_indices
        if PRICE_COL: all_cols.append(PRICE_COL)
        
        existing_cols = [c for c in all_cols if c in df_filt.columns]
        
        df_ctrl = df_filt[existing_cols].copy()
        
        # Rinomina per leggibilità
        rename_map = {
            "ShockIndex_calc": "Shock",
            "EnergyIndex_calc": "Energy",
            "FlexIndex": "Flex",
            "WeightIndex": "Weight",
            "DriveIndex": "Drive",
            "StackFactor": "StackFact",
            "label": "Modello"
        }
        df_ctrl = df_ctrl.rename(columns=rename_map)
        
        st.dataframe(df_ctrl, use_container_width=True)
