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

    # --- EXPANDERs (Documentazione) ---
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

    col_obiettivi, col_preferenze = st.columns(2)

    with col_obiettivi:
        st.subheader("🎯 Fisico e Obiettivo")
        
        # INPUT 1: Peso Corporeo
        user_weight_kg = st.slider(
            "Massa Corporea Attuale (kg):",
            min_value=50, max_value=120, value=75, step=5,
            help="Un peso maggiore richiede più ammortizzazione e stabilità."
        )
        
        # INPUT 2: Passo al Km (Select Slider per UX migliore)
        # Generiamo opzioni di passo da 3:00 a 7:00
        pace_options = [f"{m}:{s:02d}" for m in range(3, 8) for s in range(0, 60, 15)] # Ogni 15 sec
        # Rimuoviamo gli ultimi eccessivi oltre 7:00 se ce ne sono
        pace_options = [p for p in pace_options if p <= "7:00"]
        
        target_pace_str = st.select_slider(
            "Passo Medio Target (min/km):",
            options=pace_options,
            value="5:00",
            help="Il passo influenza quanto conta la reattività rispetto al comfort."
        )
        
        # Conversione Passo (str) -> Secondi (int) per il calcolo
        m, s = map(int, target_pace_str.split(':'))
        target_pace_sec_km = m * 60 + s

        weight_priority = st.slider(
            "Importanza della leggerezza:",
            min_value=0, max_value=100, value=50, step=10,
            help="0% = Priorità protezione. 100% = Priorità efficienza."
        )

    with col_preferenze:
        st.subheader("❤️ Sensazioni Richieste")
        
        shock_preference = st.select_slider(
            "Ammortizzazione e Protezione (Shock):",
            options=["Minima", "Moderata", "Bilanciata", "Elevata", "Massima"],
            value="Bilanciata",
            help="Quanto vuoi che la scarpa assorba l'impatto (Dumping)."
        )
        
        drive_preference = st.select_slider(
            "Reattività e Spinta (Drive/Energy):",
            options=["Minima", "Moderata", "Bilanciata", "Elevata", "Massima"],
            value="Bilanciata",
            help="Quanto vuoi che la scarpa restituisca energia e spinga in avanti."
        )
        
        heel_pct = st.slider(
            "Percentuale di appoggio del tallone:",
            min_value=0, max_value=100, value=40, step=10,
            help="0% = Avampiede puro. 100% = Tallone puro."
        )

    # --- MOTORE DI TRADUZIONE (PESO/PACE -> TECH) ---
    
    # 1. Fattore di Performance (Pace)
    # 3:00/km (180s) = Alta Performance (1.0) | 7:00/km (420s) = Bassa Performance (0.0)
    P_min_s = 180; P_max_s = 420
    performance_factor = 1.0 - (target_pace_sec_km - P_min_s) / (P_max_s - P_min_s)
    performance_factor = np.clip(performance_factor, 0, 1)

    # 2. Sensibilità al Peso Corporeo (Range 60kg -> 100kg)
    # > 90kg richiede molta più protezione e stabilità
    W_min_sens = 60; W_max_sens = 100
    weight_sensitivity = (user_weight_kg - W_min_sens) / (W_max_sens - W_min_sens)
    weight_sensitivity = np.clip(weight_sensitivity, 0, 1) 

    # 3. Fattore di Amplificazione Biomeccanica (MAX se Massivo E Veloce)
    # Un runner pesante che corre veloce ha bisogno di una scarpa molto strutturata e reattiva.
    amplification_factor = performance_factor * weight_sensitivity 

    # Mappatura Selettori Sensazioni (0-4)
    map_pref = {"Minima": 0, "Moderata": 1, "Bilanciata": 2, "Elevata": 3, "Massima": 4}
    score_shock = map_pref[shock_preference]
    score_drive = map_pref[drive_preference]

    # --- CALCOLO PESI MPI FINALI ---
    
    # 1. Shock: Base + Preferenza + Sensibilità al Peso Corporeo
    w_shock = (1.0 + score_shock * 1.0) + (1.5 * weight_sensitivity)
    
    # 2. Energy/Flex: Base + Preferenza + Pace + Amplificazione Biomeccanica (Effetto Carico Veloce)
    w_energy = (1.0 + score_drive * 1.0) + (1.5 * performance_factor) + (1.0 * amplification_factor)
    w_flex = (1.0 + score_drive * 1.0) + (1.0 * performance_factor) + (0.5 * amplification_factor)
    
    # 3. Weight: Base + Priorità Utente (Diretta)
    # Se l'utente è pesante, la leggerezza è meno critica della struttura (o viceversa, dipende dalla filosofia)
    # Qui assumiamo che se chiedi leggerezza, la vuoi.
    w_weight = 0.5 + (weight_priority / 100.0) * 3.5

    # Calcolo peso per Drive (per la similitudine)
    w_drive = w_energy + w_flex 

    # Clamp e Normalizzazione
    w_shock, w_energy, w_flex, w_weight = max(0.1, w_shock), max(0.1, w_energy), max(0.1, w_flex), max(0.1, w_weight)
    total_w = w_shock + w_energy + w_flex + w_weight
    
    # Pesi Percentuali per Display
    pct_shock = (w_shock / total_w) * 100
    pct_energy = (w_energy / total_w) * 100
    pct_flex = (w_flex / total_w) * 100
    pct_weight = (w_weight / total_w) * 100

    with st.expander(f"⚙️ Pesi Tecnici Applicati"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ammortizz.", f"{pct_shock:.0f} %")
        c2.metric("Ritorno Energia", f"{pct_energy:.0f} %")
        c3.metric("Spinta/Rigidità", f"{pct_flex:.0f} %")
        c4.metric("Leggerezza", f"{pct_weight:.0f} %")

    # --- CALCOLO MPI REALE ---
    w_mid = 1.0 - (heel_pct / 100.0); w_heel_val = heel_pct / 100.0
    
    df_filt.loc[:, "ShockIndex_calc"] = safe_norm(w_heel_val * df_filt["shock_abs_tallone"] + w_mid * df_filt["shock_abs_mesopiede"])
    df_filt.loc[:, "EnergyIndex_calc"] = safe_norm(w_heel_val * df_filt["energy_ret_tallone"] + w_mid * df_filt["energy_ret_mesopiede"])

    df_filt.loc[:, "MPI_B"] = (
        (w_shock * df_filt["ShockIndex_calc"] + 
         w_energy * df_filt["EnergyIndex_calc"] + 
         w_flex * df_filt["FlexIndex"] + 
         w_weight * df_filt["WeightIndex"]) / total_w
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
                
                # 1. Trova i Top 3 basati SOLTANTO sull'MPI (Logica Podio Pura)
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
    # 2. ANALISI MERCATO (UTILIZZA BEST PICK COME DEFAULT)
    # ============================================

    st.markdown("---")
    st.header("2. Analisi Comparativa di Mercato")
    
    if not df_filt.empty:
        df_val_sorted = df_filt.sort_values("ValueIndex", ascending=False)
        models = df_val_sorted['label'].tolist()
        
        # 1. Aggiorna lo stato se il Best Pick è stato trovato
        best_pick_label_check = best_pick_label if 'best_pick_label' in locals() and best_pick_label and best_pick_label in models else models[0]

        if 'selected_point_key' not in st.session_state:
            st.session_state['selected_point_key'] = best_pick_label_check
        
        # 2. Aggiorna lo stato se il best pick è diverso dal target attuale
        if best_pick_label_check != st.session_state['selected_point_key']:
             st.session_state['selected_point_key'] = best_pick_label_check
             st.rerun() 

        # 3. Controllo consistenza (fallback)
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
            # Prepara l'elenco per evidenziare il podio nel grafico (Opzionale, qui evidenziamo solo la selezione)
            # Se volessi evidenziare tutto il podio: podium_labels = top_picks['label'].tolist() se esiste
            
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
                st.subheader(f"{row['marca']}")
                st.markdown(f"**{row['modello']}**")
                st.metric("MPI", f"{row['MPI_B']:.2f}")
                st.write(f"**Value:** {row['ValueIndex']:.2f} {render_stars(row['ValueIndex'])}")
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
        
        # Qui usiamo il DriveIndex per la similarità, che è importante
        cols_sim = ["ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex", "DriveIndex"]
        
        # Pesi per la similarità (opzionale, ma coerente con le scelte utente)
        sim_weights = [w_shock, w_energy, w_flex, w_weight, w_energy] # Drive pesa come Energy approx

        simili = trova_scarpe_simili(df_filt, sel_input, cols_sim, weights=sim_weights, n_simili=3)
        
        if not simili.empty:
            cc = st.columns(3)
            for i, (_, s_row) in enumerate(simili.iterrows()):
                with cc[i]:
                    with st.container(border=True):
                        st.markdown(f"**{s_row['label']}**")
                        st.caption(f"Dist: {s_row['distanza_similitudine']:.2f}")
            
            df_rad = pd.concat([df_filt[df_filt['label']==sel_input], simili], ignore_index=True)
            st.plotly_chart(plot_radar_comparison_plotly_styled(df_rad, cols_sim), use_container_width=True)
        else:
            st.warning("Nessun dato con i filtri attuali.")
        
        # Tabella Controllo
        st.markdown("---")
        with st.expander("📊 Tabella di Controllo Completa"):
            cols_ctrl = ["label", "MPI_B", "ValueIndex", "DriveIndex", "StackFactor", "ShockIndex_calc", "EnergyIndex_calc", "FlexIndex", "WeightIndex"]
            if PRICE_COL: cols_ctrl.append(PRICE_COL)
            st.dataframe(df_filt[[c for c in cols_ctrl if c in df_filt.columns]], use_container_width=True)

