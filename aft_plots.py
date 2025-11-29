# aft_plots.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def render_stars(value):
    if pd.isna(value): return ""
    score = int(round(value * 5))
    score = max(0, min(5, score))
    return ("★" * score) + ("☆" * (5 - score))

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
        legend_title_text='Legenda',
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
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9, color="gray"), gridcolor='#e6e6e6'), bgcolor='white'),
        title=dict(text=title, x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig