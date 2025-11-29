# main.py
import streamlit as st
import pandas as pd
import numpy as np

# Import dai moduli personalizzati
from aft_core import trova_scarpe_simili
from aft_plots import render_stars, plot_mpi_vs_price_plotly, plot_radar_comparison_plotly_styled
from aft_utils import check_password, load_and_process

# Configurazione Pagina
st.set_page_config(page_title="AFT Analyst", layout="wide")

if check_password():
    st.title("Database AFT: Analisi Biomeccanica e Clustering")
    st.markdown("**Advanced Footwear Technology Analysis Tool**")

    # --- Expanders ---
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
        
        ### 3. Weight Efficiency ($I_{Weight}$)
        Decadimento esponenziale dal riferimento 180g.
        $$ I_{Weight} = e^{-0.005 \cdot (Peso_{g} - 180)} $$
        """)

    # Caricamento Dati
    file_name = "database_completo_AFT_20251124_clean.csv"
    # Nota: se il file è nella stessa cartella, il path è vuoto o "./"
    df_raw, cluster_summary_raw = load_and_process("./", file_name)
    
    if df_raw.empty: st.stop()
    
    df = df_raw.copy()
    PRICE_COL = [c for c in df.columns if "prezzo" in c.lower()]
    PRICE_COL = PRICE_COL[0] if PRICE_COL else None

    # --- SIDEBAR ---
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

    # --- STEP 1: MPI ---
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

    # Ricalcolo Dinamico MPI
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

    # --- BEST PICK ---
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
    
    # --- STEP 2: ANALISI MERCATO ---
    st.markdown("---")
    st.header("2. Analisi Mercato")
    
    if not df_filt.empty:
        df_val_sorted = df_filt.sort_values("ValueIndex", ascending=False)
        models = df_val_sorted['label'].tolist()
        
        # Gestione Stato Selezione
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
            st.write("Top Value")
            st.dataframe(df_val_sorted[["label", "MPI_B", "ValueIndex"]].head(10), use_container_width=True, hide_index=True)

        # --- STEP 3: DETTAGLIO ---
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

        # --- STEP 4: SIMILITUDINE ---
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
        
    # --- TABELLA CONTROLLO ---
    st.markdown("---")
    with st.expander("📊 Tabella di Controllo Completa"):
        cols_ctrl = ["label", "MPI_B", "ValueIndex", "DriveIndex", "StackFactor", "ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex"]
        if PRICE_COL: cols_ctrl.append(PRICE_COL)
        st.dataframe(df_filt[cols_ctrl], use_container_width=True)