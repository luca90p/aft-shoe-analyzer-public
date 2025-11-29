# main.py
import streamlit as st
import pandas as pd
import numpy as np

# Import dai moduli personalizzati
from aft_core import trova_scarpe_simili
from aft_plots import plot_mpi_vs_price_plotly, plot_radar_comparison_plotly_styled, render_stars
from aft_utils import check_password, load_and_process, safe_norm

# =========================
#   CONFIGURAZIONE E LOGIN
# =========================
st.set_page_config(page_title="AFT Analyst", layout="wide")

if check_password():
    st.title("Database AFT: Analisi Biomeccanica e Clustering")
    st.markdown("**Advanced Footwear Technology Analysis Tool**")

    # --- EXPANDERs (Documentazione) ---
    with st.expander("📘 Metodologia e Riferimenti Bibliografici"):
        st.markdown("""
        **1. Costo Metabolico del Peso**
        Ogni 100g extra aumentano il costo energetico dell'1%.
        *Fonte:* [cite: metabolic cost of running, body weight influence.pdf]
        
        **2. Indice di Spinta Meccanica (Drive Index)**
        Sinergia tra Piastra, Rocker e Rigidità.
        *Fonte:* [cite: Effects of the curved carbon fibre plate and PEBA foam on the energycost of running and muscle activation.pdf]
        
        **3. Rigidità Longitudinale (Flex Index)**
        Range misurato: 5N (Soft) - 40N (Stiff).
        *Fonte:* [cite: The eﬀects of footwear midsole longitudinal bending stiﬀness on runningeconomy and ground contact biomechanics A systematic review and meta-analysis.pdf]
        """)

    with st.expander("📐 Formule Matematiche del Modello AFT"):
        st.markdown(r"""
        ### 1. Flex Index ($I_{Flex}$) - Range 5-40 N
        * **Race:** Sigmoide centrata su 18N.
        * **Daily:** Gaussiana centrata su 12N.

        ### 2. Drive Index ($I_{Drive}$)
        Modella l'effetto leva ("Teeter-Totter"). La componente meccanica è una moltiplicazione (interazione), non una somma.
        $$ I_{Drive} = 0.6 \cdot (S_{Plate} \cdot S_{Rocker} \cdot S_{Stiffness}) + 0.4 \cdot S_{Foam} $$
        """)

    # --- CARICAMENTO DATI ---
    file_name = "database_completo_AFT_20251124_clean.csv"
    df_raw, cluster_summary_raw = load_and_process("./", file_name)
    
    if df_raw.empty: st.stop()
    
    df = df_raw.copy()
    PRICE_COL = [c for c in df.columns if "prezzo" in c.lower()]
    PRICE_COL = PRICE_COL[0] if PRICE_COL else None

    # --- SIDEBAR (Filtri) ---
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

    # ============================================
    # 1. WIZARD GUIDATO
    # ============================================

    st.header("1. Parametrizzazione Performance (MPI)")
    st.info("Definisci i criteri per calcolare l'indice MPI personalizzato in base alle tue esigenze.")

    # --- INTERFACCIA UTENTE SEMPLIFICATA ---
    col_obiettivi, col_preferenze = st.columns(2)

    with col_obiettivi:
        st.subheader("🎯 Obiettivo")
        run_type = st.select_slider(
            "Per cosa userai queste scarpe?",
            options=["Recupero / Easy", "Lungo Lento", "Allenamento Quotidiano", "Tempo / Ripetute", "Gara / PB"],
            value="Allenamento Quotidiano"
        )
        weight_priority = st.slider(
            "Quanto è importante che la scarpa sia leggera?",
            min_value=0, max_value=100, value=50, step=10
        )

    with col_preferenze:
        st.subheader("❤️ Sensazioni")
        feel_preference = st.select_slider(
            "Che sensazione cerchi sotto il piede?",
            options=["Nuvola (Max Morbidezza)", "Bilanciata", "Secca / Reattiva"],
            value="Bilanciata"
        )
        heel_pct = st.slider(
            "Come appoggi il piede?",
            min_value=0, max_value=100, value=40, step=10
        )

    # --- MOTORE DI TRADUZIONE (USER -> TECH) ---
    map_run_type = {"Recupero / Easy": 0, "Lungo Lento": 1, "Allenamento Quotidiano": 2, "Tempo / Ripetute": 3, "Gara / PB": 4}
    score_goal = map_run_type[run_type]
    map_feel = {"Nuvola (Max Morbidezza)": 0, "Bilanciata": 1, "Secca / Reattiva": 2}
    score_feel = map_feel[feel_preference]

    # Calcolo pesi euristici
    w_shock, w_energy, w_flex, w_weight = 2.0, 2.0, 1.0, 1.0
    w_shock -= score_goal * 0.3; w_energy += score_goal * 0.8; w_flex += score_goal * 0.6
    if score_feel == 0: w_shock += 2.0; w_flex -= 0.5
    elif score_feel == 2: w_shock -= 0.5; w_flex += 1.0; w_energy += 0.5
    w_weight = 0.5 + (weight_priority / 100.0) * 3.5
    w_shock, w_energy, w_flex, w_weight = max(0.1, w_shock), max(0.1, w_energy), max(0.1, w_flex), max(0.1, w_weight)
    total_w = w_shock + w_energy + w_flex + w_weight
    pct_shock, pct_energy, pct_flex, pct_weight = (w_shock / total_w) * 100, (w_energy / total_w) * 100, (w_flex / total_w) * 100, (w_weight / total_w) * 100

    with st.expander(f"⚙️ Pesi Tecnici Applicati"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ammortizz.", f"{pct_shock:.0f}%")
        c2.metric("Ritorno Energia", f"{pct_energy:.0f}%")
        c3.metric("Spinta/Rigidità", f"{pct_flex:.0f}%")
        c4.metric("Leggerezza", f"{pct_weight:.0f}%")

    # --- CALCOLO MPI REALE ---
    w_mid = 1.0 - (heel_pct / 100.0)
    w_heel_val = heel_pct / 100.0
    
    # Ricalcolo dinamico indici parziali (Shock/Energy)
    # FIX: safe_norm deve essere chiamata su colonne valide. 
    # Le variabili w_heel_val e w_mid sono ora definite e utilizzabili.
    df_filt.loc[:, "ShockIndex_calc"] = safe_norm(w_heel_val * df_filt["shock_abs_tallone"] + w_mid * df_filt["shock_abs_mesopiede"])
    df_filt.loc[:, "EnergyIndex_calc"] = safe_norm(w_heel_val * df_filt["energy_ret_tallone"] + w_mid * df_filt["energy_ret_mesopiede"])

    df_filt.loc[:, "MPI_B"] = (
        (w_shock * df_filt["ShockIndex_calc"] + 
         w_energy * df_filt["EnergyIndex_calc"] + 
         w_flex * df_filt["FlexIndex"] + 
         w_weight * df_filt["WeightIndex"]) / total_w
    ).round(3)

    if PRICE_COL:
        df_filt = df_filt[df_filt[PRICE_COL] > 0].copy()
        v_raw = df_filt["MPI_B"] / df_filt[PRICE_COL]
        df_filt["ValueIndex"] = safe_norm(v_raw)
    else:
        df_filt["ValueIndex"] = 0.0

    # --- BEST PICK ---
    st.markdown("---")
    st.header("💡 Best Pick")
    if PRICE_COL:
        min_price = df_filt[PRICE_COL].min()
        max_price = df_filt[PRICE_COL].max()
        
        b_max = st.slider("Budget Max (€)", int(min_price) if not np.isnan(min_price) else 50, int(max_price) if not np.isnan(max_price) else 300, 200, 5)
        
        picks = df_filt[df_filt[PRICE_COL] <= b_max].sort_values("MPI_B", ascending=False)
        if not picks.empty:
            bp = picks.iloc[0]
            with st.container(border=True):
                k1, k2 = st.columns([3, 1])
                k1.subheader(f"🏆 {bp['label']}")
                k2.metric("MPI", f"{bp['MPI_B']}")
                k2.write(f"{bp[PRICE_COL]:.0f} €")
        else:
            st.warning("Nessun risultato nel range di budget.")
    
    # --- STEP 2: ANALISI MERCATO ---
    st.markdown("---")
    st.header("2. Analisi Mercato")
    
    if not df_filt.empty:
        df_val_sorted = df_filt.sort_values("ValueIndex", ascending=False)
        models = df_val_sorted['label'].tolist()
        
        if 'selected_point_key' not in st.session_state or st.session_state['selected_point_key'] not in models:
            st.session_state['selected_point_key'] = models[0]
            
        curr_sel = st.session_state['selected_point_key']
        idx_sel = models.index(curr_sel)
        
        sel_input = st.selectbox("Selezione Modello Target:", models, index=idx_sel, key='main_sb')
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
            st.dataframe(df_filt[[c for c in cols_ctrl if c in df_filt.columns]], use_container_width=True)
