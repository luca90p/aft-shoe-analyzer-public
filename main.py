# main.py
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

    # --- EXPANDERs (Documentazione) - CONTENUTO RICHIESTO DALL'UTENTE ---
    with st.expander("📘 Metodologia e Riferimenti Bibliografici"):
        st.markdown("""
        **1. Rigidità Longitudinale (Flex Index)**
        Il database misura la **Forza (N)** necessaria per flettere la suola di 30°. Il punteggio segue una modellazione non lineare.
        * **Logica:** Le scarpe da gara premiano la rigidità elevata; quelle da allenamento cercano un valore moderato per comfort.
        * *Fonte:* **Rodrigo-Carranza et al. (2022).** *The effects of footwear midsole longitudinal bending stiffness on running economy...*
        
        **2. Costo Metabolico del Peso (Weight Efficiency)**
        *Ogni 100g di massa aggiuntiva aumentano il costo energetico dell'1%.* La funzione di penalità del peso segue un decadimento esponenziale.
        * *Fonte:* **Teunissen, Grabowski & Kram (2007).** *Effects of independently altering body weight and body mass on the metabolic cost of running.*
        
        **3. Indice di Spinta Meccanica (Drive Index)**
        La performance deriva dall'interazione ("Teeter-Totter effect") tra la piastra, la geometria Rocker e la rigidità.
        * *Fonte:* **Ghanbari et al. (2025).** *Effects of the curved carbon fibre plate and PEBA foam on the energy cost of running...*
        
        **4. Stack Height e Stabilità**
        Lo stack alto (>40mm) può compromettere la stabilità biomeccanica se non adeguatamente compensato.
        * *Fonte:* **Kettner et al. (2025).** *The effects of running shoe stack height on running style and stability...*

        **5. Indice di Durabilità (Durability Index)**
        Questo indice sintetico (0-1) stima la longevità della scarpa combinando tre fattori di resistenza all'abrasione:
        * **Suola:** Calcolato come rapporto tra la profondità del danno abrasivo e lo spessore totale della gomma (Wear Ratio).
        * **Tomaia e Tallone:** Basato su test qualitativi di resistenza all'usura (scala standardizzata).

        **6. Analisi Volumetrica della Calzata (Fit Class)**
        La classificazione (Stretta, Standard, Ampia) è determinata statisticamente analizzando la distribuzione normale delle larghezze del toe-box (zona dita) dell'intero database. I modelli che deviano di oltre 0.5 deviazioni standard dalla media vengono classificati come Stretti o Ampi.
        """)
        

    with st.expander("📐 Formule Matematiche del Modello AFT"):
        st.markdown(r"""
        Il calcolo del punteggio totale **MPI-B** è una somma pesata di 5 indici normalizzati $[0, 1]$.
        
        ### 1. Flex Index ($I_{Flex}$)
        Basato sulla **Forza di Flessione ($F_N$)** in Newton (Range 5N - 40N).
        * **Race (Modello Sigmoide):** Premia la rigidità alta (> 18N).
          $$ I_{Flex} = \frac{1}{1 + e^{-(F_N - 18)/2.5}} $$
        * **Daily (Modello Gaussiano):** Premia il comfort (~12N).
          $$ I_{Flex} = e^{-\frac{(F_N - 12)^2}{2 \cdot 5^2}} $$

        ### 2. Drive Index ($I_{Drive}$) - "Teeter-Totter Effect"
        Modella la spinta come **interazione moltiplicativa** (effetto leva) tra i componenti meccanici, sommata al contributo del materiale.
        $$ I_{Drive} = 0.6 \cdot (S_{Plate} \cdot S_{Rocker} \cdot S_{Stiff}) + 0.4 \cdot I_{Energy} $$
        * $S_{Plate}$: 1.0 (Carbonio), 0.7 (Vetro), 0.5 (Plastica).
        * $S_{Rocker}$: Altezza punta normalizzata su 10mm.
        * $S_{Stiff}$: Rigidità normalizzata su 35N ($F_N / 35$).

        ### 3. Weight Efficiency ($I_{Weight}$)
        Decadimento esponenziale basato sul costo metabolico (+1% per +100g).
        $$ I_{Weight} = e^{-0.005 \cdot (Peso_{g} - 180)} $$
        *(Penalizza progressivamente i pesi superiori a 180g)*.
        
        ### 4. Durability Index ($I_{Dur}$)
        Combinazione pesata dei tassi di usura.
        $$ I_{Dur} = 0.6 \cdot (1 - \frac{Danno_{suola}}{Spessore_{suola}}) + 0.25 \cdot Res_{Tomaia} + 0.15 \cdot Res_{Tallone} $$
        """)
    # --- FINE EXPANDERs RICHIESTI ---

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
    # 1. WIZARD GUIDATO (PARAMETRIZZAZIONE MPI)
    # ============================================

    st.header("1. Parametrizzazione Performance (MPI)")
    st.info("Definisci i criteri per calcolare l'indice di performance biomeccanica personalizzato.")

    col_obiettivi, col_preferenze = st.columns(2)

    with col_obiettivi:
        st.subheader("🎯 Obiettivo e Fisico")
        
        # Input Fisici
        user_weight_kg = st.slider("Massa Corporea (kg):", 50, 120, 75, 5)
        
        # Input Passo (Select Slider per UX)
        pace_options = [f"{m}:{s:02d}" for m in range(3, 8) for s in range(0, 60, 15)]
        pace_options = [p for p in pace_options if p <= "7:00"]
        target_pace_str = st.select_slider("Passo Medio Target (min/km):", options=pace_options, value="5:00")
        
        # Conversione passo
        m, s = map(int, target_pace_str.split(':'))
        target_pace_sec_km = m * 60 + s

        weight_priority = st.slider("Importanza Leggerezza:", 0, 100, 50, 10)

    with col_preferenze:
        st.subheader("❤️ Sensazioni Richieste")
        
        shock_preference = st.select_slider(
            "Ammortizzazione (Shock):",
            options=["Minima", "Moderata", "Bilanciata", "Elevata", "Massima"],
            value="Bilanciata",
            help="Quanto vuoi che la scarpa assorba l'impatto (Dumping)."
        )
        
        drive_preference = st.select_slider(
            "Reattività (Drive/Energy):",
            options=["Minima", "Moderata", "Bilanciata", "Elevata", "Massima"],
            value="Bilanciata",
            help="Quanto vuoi che la scarpa restituisca energia e spinga."
        )
        
        heel_pct = st.slider("Appoggio Tallone (%):", 0, 100, 40, 10)

    # --- MOTORE DI TRADUZIONE (User -> Pesi Tecnici) ---
    
    # 1. Fattore Performance (Pace) & Peso
    P_min, P_max = 180, 420
    perf_factor = np.clip(1.0 - (target_pace_sec_km - P_min)/(P_max - P_min), 0, 1)
    weight_sens = np.clip((user_weight_kg - 60) / 40, 0, 1)
    amp_factor = perf_factor * weight_sens

    # 2. Mappatura Sensazioni
    map_pref = {"Minima": 0, "Moderata": 1, "Bilanciata": 2, "Elevata": 3, "Massima": 4}
    s_shock = map_pref[shock_preference]
    s_drive = map_pref[drive_preference]

    # 3. Calcolo Pesi MPI
    w_shock = (0.5 + s_shock * 1.0) + (1.5 * weight_sens)
    w_energy = (0.5 + s_drive * 1.0) + (1.5 * perf_factor) + (1.0 * amp_factor)
    w_flex = (0.5 + s_drive * 1.0) + (1.0 * perf_factor) + (0.5 * amp_factor)
    w_weight = 0.5 + (weight_priority / 100.0) * 3.5
    
    # Peso implicito per Drive (usato solo per similarità vettoriale)
    w_drive = (w_energy + w_flex) / 1.5

    # Clamp e Normalizzazione visuale
    w_shock, w_energy, w_flex, w_weight = max(0.1, w_shock), max(0.1, w_energy), max(0.1, w_flex), max(0.1, w_weight)
    tot_w = w_shock + w_energy + w_flex + w_weight

    with st.expander(f"⚙️ Pesi Tecnici Applicati"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ammortizz.", f"{(w_shock/tot_w)*100:.0f}%")
        c2.metric("Ritorno Energia", f"{(w_energy/tot_w)*100:.0f}%")
        c3.metric("Spinta/Rigidità", f"{(w_flex/tot_w)*100:.0f}%")
        c4.metric("Leggerezza", f"{(w_weight/tot_w)*100:.0f}%")

    # --- CALCOLO MPI REALE ---
    w_mid = 1.0 - (heel_pct / 100.0); w_heel_val = heel_pct / 100.0
    
    # Ricalcolo dinamico indici parziali
    df_filt.loc[:, "ShockIndex_calc"] = safe_norm(w_heel_val * df_filt["shock_abs_tallone"] + w_mid * df_filt["shock_abs_mesopiede"])
    df_filt.loc[:, "EnergyIndex_calc"] = safe_norm(w_heel_val * df_filt["energy_ret_tallone"] + w_mid * df_filt["energy_ret_mesopiede"])

    # MPI include SOLO metriche di performance (come richiesto)
    df_filt.loc[:, "MPI_B"] = (
        (w_shock * df_filt["ShockIndex_calc"] + 
         w_energy * df_filt["EnergyIndex_calc"] + 
         w_flex * df_filt["FlexIndex"] + 
         w_weight * df_filt["WeightIndex"]) / tot_w
    ).round(2)

    if PRICE_COL:
        df_filt = df_filt[df_filt[PRICE_COL] > 0].copy()
        v_raw = df_filt["MPI_B"] / df_filt[PRICE_COL]
        df_filt["ValueIndex"] = safe_norm(v_raw).round(2)
    else:
        df_filt["ValueIndex"] = 0.0

    # ============================================
    # 1.5 BEST PICK (PODIO)
    # ============================================

    st.markdown("---")
    st.header("💡 Best Pick: Il Podio per il tuo Budget")
    
    best_pick_label = None

    if PRICE_COL:
        min_price = df_filt[PRICE_COL].min()
        max_price = df_filt[PRICE_COL].max()
        
        min_p = int(min_price) if not np.isnan(min_price) else 50
        max_p = int(max_price) if not np.isnan(max_price) else 300
        default_p = int(min_p + (max_p - min_p) * 0.75)

        col_budget, col_best = st.columns([1, 2])

        with col_budget:
            budget_max = st.slider("Budget Max (€):", min_p, max_p, default_p, 5)
        
        df_budget = df_filt[df_filt[PRICE_COL] <= budget_max].copy()
        
        with col_best:
            if not df_budget.empty:
                
                # Trova i Top 3 basati SOLO sull'MPI (Logica Podio Pura)
                top_picks = df_budget.sort_values(by="MPI_B", ascending=False).head(3)
                
                if not top_picks.empty:
                    best_pick_label = top_picks.iloc[0]['label']
                    rank_labels = ["🥇 1° Posto", "🥈 2° Posto", "🥉 3° Posto"]
                    
                    cols_podium = st.columns(3)
                    
                    for i, (idx, bp) in enumerate(top_picks.iterrows()):
                        with cols_podium[i]:
                            with st.container(border=True):
                                st.markdown(f"#### {rank_labels[i]}")
                                st.subheader(f"{bp['marca']} {bp['modello']}")
                                
                                if pd.notna(bp.get('versione')):
                                    st.caption(f"Versione: {int(bp['versione'])}")
                                
                                st.write(f"MPI Score: **{bp['MPI_B']:.2f}**")
                                st.write(f"Prezzo: **{bp[PRICE_COL]:.0f} €**")
                                
                                if pd.notna(bp.get('ValueIndex')):
                                    stars = render_stars(bp['ValueIndex'])
                                    st.caption(f"Value: {stars}")
            else:
                st.warning("Nessun risultato nel range di budget.")

    # ============================================
    # 2. ANALISI MERCATO
    # ============================================

    st.markdown("---")
    st.header("2. Analisi Comparativa di Mercato")
    
    if not df_filt.empty:
        df_val_sorted = df_filt.sort_values("ValueIndex", ascending=False)
        models = df_val_sorted['label'].tolist()
        
        # Gestione selezione
        best_pick_label_check = best_pick_label if 'best_pick_label' in locals() and best_pick_label and best_pick_label in models else models[0]

        if 'selected_point_key' not in st.session_state:
            st.session_state['selected_point_key'] = best_pick_label_check
        
        if best_pick_label_check != st.session_state['selected_point_key']:
             st.session_state['selected_point_key'] = best_pick_label_check
             st.rerun() 

        curr_sel = st.session_state['selected_point_key']
        if curr_sel not in models:
            curr_sel = models[0]
            st.session_state['selected_point_key'] = curr_sel
            
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

        # ============================================
    # 3. SCHEDA DETTAGLIO
    # ============================================

    st.markdown("---")
    st.header("3. Scheda Tecnica")
    
    # Recupera la riga selezionata in modo sicuro
    try:
        row = df_filt[df_filt["label"] == sel_input].iloc[0]
    except IndexError:
        st.warning("Errore nel recupero del modello selezionato.")
        st.stop()
    
    with st.container(border=True):
        c1, c2 = st.columns([1, 2])
        
        # Colonna Sinistra: Info Generali e Valore
        with c1:
            st.subheader(f"{row['marca']}")
            st.markdown(f"**{row['modello']}**")
            if pd.notna(row.get('versione')):
                st.caption(f"Versione: {int(row['versione'])}")
            
            st.metric("MPI Score", f"{row['MPI_B']:.2f}")
            st.write(f"Prezzo: **{row[PRICE_COL]:.0f} €**")
            
            if pd.notna(row.get('ValueIndex')):
                val_idx = float(row['ValueIndex'])
                stars = render_stars(val_idx)
                st.write(f"**Value:** {val_idx:.2f} {stars}")

            # --- NUOVO: Visualizzazione Fit ---
            if 'FitClass' in row:
                st.info(f"👟 **Calzata:** {row['FitClass']}")

        # Colonna Destra: Dettagli Tecnici
        with c2:
            st.write(f"**Peso:** {row['peso']}g | **Cluster:** {row['ClusterDescrizione']}")
            
            colA, colB = st.columns(2)
            
            # Indici Performance (MPI)
            colA.progress(row['ShockIndex_calc'], text=f"Shock: {row['ShockIndex_calc']:.2f}")
            colB.progress(row['EnergyIndex_calc'], text=f"Energy: {row['EnergyIndex_calc']:.2f}")
            colA.progress(row['FlexIndex'], text=f"Flex: {row['FlexIndex']:.2f}")
            colB.progress(row['DriveIndex'], text=f"Drive: {row['DriveIndex']:.2f}")
            
            # --- NUOVO: Visualizzazione Durabilità ---
            st.markdown("---")
            if 'StabilityIndex' in row:
                    stab = float(row['StabilityIndex'])
                    stab_desc = "Alta" if stab > 0.7 else "Media" if stab > 0.4 else "Bassa"
                    st.write(f"⚖️ **Stabilità Strutturale:** {stab_desc} ({stab:.2f})")
                    st.progress(stab)
                
            if 'DurabilityIndex' in row:
                dur = float(row['DurabilityIndex'])
                dur_label = "Alta" if dur > 0.7 else "Media" if dur > 0.4 else "Bassa"
                st.write(f"🛡️ **Durabilità Stimata:** {dur_label} ({dur:.2f})")
                st.progress(dur)
                
        # --- STEP 4: SIMILITUDINE (MOSTRA ANCHE EXTRA) ---
        st.markdown("---")
        st.header("4. Similitudine & Radar")
        cols_sim = ["ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex", "DriveIndex"]
        sim_weights = [w_shock, w_energy, w_flex, w_weight, w_drive]
        
        # Trova 2 scarpe simili basate sui pesi MPI
        simili = trova_scarpe_simili(df_filt, sel_input, cols_sim, weights=sim_weights, n_simili=2)
        
        if not simili.empty:
            cc = st.columns(2)
            for i, (_, s_row) in enumerate(simili.iterrows()):
                with cc[i]:
                    with st.container(border=True):
                        st.markdown(f"**Alternativa {i+1}: {s_row['label']}**")
                        st.caption(f"Distanza Biomeccanica: {s_row['distanza_similitudine']:.2f}")
                        
                        # Info extra per confronto rapido
                        fit_info = s_row.get('FitClass', 'N/D')
                        dur_val_sim = s_row.get('DurabilityIndex', 0)
                        st.caption(f"Fit: {fit_info} | Durata: {dur_val_sim:.2f}")
            
            df_rad = pd.concat([df_filt[df_filt['label']==sel_input], simili], ignore_index=True)
            st.plotly_chart(plot_radar_comparison_plotly_styled(df_rad, cols_sim), use_container_width=True)
        else:
            st.warning("Nessun modello simile trovato con i filtri attuali.")
        
        # --- TABELLA CONTROLLO ---
        st.markdown("---")
        with st.expander("Tabella Dati Completa (Tutti gli Indici)"):
            cols_ctrl = ["label", "MPI_B", "ValueIndex", "DriveIndex", "StabilityIndex", "DurabilityIndex", "FitClass", "ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex"]
            if PRICE_COL: cols_ctrl.append(PRICE_COL)
            st.dataframe(df_filt[[c for c in cols_ctrl if c in df_filt.columns]], use_container_width=True)



