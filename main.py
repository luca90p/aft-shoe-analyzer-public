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

    # ============================================
# 1. WIZARD GUIDATO (SEMPIFICATO)
# ============================================

st.header("1. Il tuo Profilo di Corsa")
st.info("Rispondi a queste domande per trovare la scarpa perfetta per le tue esigenze.")

# --- INTERFACCIA UTENTE SEMPLIFICATA ---
col_obiettivi, col_preferenze = st.columns(2)

with col_obiettivi:
    st.subheader("🎯 Obiettivo")
    
    # Slider: Tipo di Corsa
    run_type = st.select_slider(
        "Per cosa userai queste scarpe?",
        options=["Recupero / Easy", "Lungo Lento", "Allenamento Quotidiano", "Tempo / Ripetute", "Gara / PB"],
        value="Allenamento Quotidiano"
    )
    
    # Slider: Peso Corporeo / Importanza Leggerezza
    weight_priority = st.slider(
        "Quanto è importante che la scarpa sia leggera?",
        min_value=0, max_value=100, value=50, step=10,
        help="0 = Non mi interessa il peso, voglio protezione. 100 = Voglio la scarpa più leggera possibile."
    )

with col_preferenze:
    st.subheader("❤️ Sensazioni")
    
    # Slider: Feeling (Soft vs Responsive)
    feel_preference = st.select_slider(
        "Che sensazione cerchi sotto il piede?",
        options=["Nuvola (Max Morbidezza)", "Bilanciata", "Secca / Reattiva"],
        value="Bilanciata"
    )
    
    # Slider: Appoggio (Resta tecnico ma semplificato)
    heel_pct = st.slider(
        "Come appoggi il piede?",
        min_value=0, max_value=100, value=40, step=10,
        help="0% = Tutto sull'avampiede (Punta). 100% = Tutto sul tallone."
    )

# --- MOTORE DI TRADUZIONE (USER -> TECH) ---
# Traduciamo le scelte dell'utente in pesi (w) per l'algoritmo MPI

# 1. Mappatura Obiettivo (0-4)
map_run_type = {
    "Recupero / Easy": 0,
    "Lungo Lento": 1,
    "Allenamento Quotidiano": 2,
    "Tempo / Ripetute": 3,
    "Gara / PB": 4
}
score_goal = map_run_type[run_type] # 0 a 4

# 2. Mappatura Feeling (0-2)
map_feel = {
    "Nuvola (Max Morbidezza)": 0,
    "Bilanciata": 1,
    "Secca / Reattiva": 2
}
score_feel = map_feel[feel_preference] # 0 a 2

# --- CALCOLO PESI AUTOMATICO ---
# Base di partenza
w_shock = 2.0
w_energy = 2.0
w_flex = 1.0
w_weight = 1.0

# Aggiustamento basato su OBIETTIVO
# Più si va verso Gara, meno conta Shock, più contano Energy e Flex
w_shock  -= score_goal * 0.3  # Toglie importanza all'ammortizzazione pura
w_energy += score_goal * 0.8  # Aumenta drasticamente ritorno energia
w_flex   += score_goal * 0.6  # Aumenta importanza rigidità/spinta

# Aggiustamento basato su FEELING
if score_feel == 0: # Nuvola (Max Morbidezza)
    w_shock += 2.0
    w_flex  -= 0.5
elif score_feel == 2: # Secca / Reattiva
    w_shock -= 0.5
    w_flex  += 1.0
    w_energy += 0.5

# Aggiustamento basato su IMPORTANZA PESO
# Scala da 0.5 (poco importante) a 4.0 (fondamentale)
w_weight = 0.5 + (weight_priority / 100.0) * 3.5

# Assicuriamoci che nessun peso sia troppo basso
w_shock = max(0.1, w_shock)
w_energy = max(0.1, w_energy)
w_flex = max(0.1, w_flex)
w_weight = max(0.1, w_weight)

# Normalizzazione per visualizzazione (somma 100%)
total_w = w_shock + w_energy + w_flex + w_weight
pct_shock = (w_shock / total_w) * 100
pct_energy = (w_energy / total_w) * 100
pct_flex = (w_flex / total_w) * 100
pct_weight = (w_weight / total_w) * 100

# --- FEEDBACK VISIVO UTENTE ---
with st.expander(f"⚙️ Vedi come l'algoritmo ha interpretato le tue scelte (Pesi Tecnici)"):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ammortizz.", f"{pct_shock:.0f}%")
    c2.metric("Ritorno Energia", f"{pct_energy:.0f}%")
    c3.metric("Spinta/Rigidità", f"{pct_flex:.0f}%")
    c4.metric("Leggerezza", f"{pct_weight:.0f}%")


# --- CALCOLO MPI REALE ---
w_mid = 1.0 - (heel_pct / 100.0)
w_heel_val = heel_pct / 100.0

# Ricalcolo dinamico indici parziali (usa la logica Shock/Energy modificata)
def safe_norm(s): 
    # Usiamo una funzione interna per la normalizzazione del df filtrato
    s = pd.to_numeric(s, errors='coerce').fillna(s.mean())
    return (s - s.min()) / max(s.max() - s.min(), 1e-9)

# Qui usiamo la logica di ammortizzazione tallone/mesopiede
df_filt["ShockIndex_calc"] = safe_norm(w_heel_val * df_filt["shock_abs_tallone"] + w_mid * df_filt["shock_abs_mesopiede"])
df_filt["EnergyIndex_calc"] = safe_norm(w_heel_val * df_filt["energy_ret_tallone"] + w_mid * df_filt["energy_ret_mesopiede"])

# Calcolo MPI Finale Ponderato
df_filt["MPI_B"] = (
    (w_shock * df_filt["ShockIndex_calc"] + 
     w_energy * df_filt["EnergyIndex_calc"] + 
     w_flex * df_filt["FlexIndex"] + 
     w_weight * df_filt["WeightIndex"]) / total_w
).round(3)

# Calcolo Value Index
if PRICE_COL and not df_filt.empty:
    # Filtra prezzi validi e ValueIndex
    df_filt = df_filt[df_filt[PRICE_COL] > 0].copy()
    v_raw = df_filt["MPI_B"] / df_filt[PRICE_COL]
    
    # Normalizza ValueIndex
    v_min = v_raw.min()
    v_max = v_raw.max()
    denom = max(v_max - v_min, 1e-9) # Evita divisione per zero
    
    df_filt["ValueIndex"] = ((v_raw - v_min) / denom).round(3)
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
