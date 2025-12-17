# app_sonic_ravensgate.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import chardet
import os
import tempfile
import io
import plotly.graph_objects as go
from datetime import datetime

# --- Table de réglage température (Ts) ---
temperature_control_table = {
    36: {0:31, 5:31, 10:32, 15:33, 20:34, 25:34, 30:35, 35:36, 40:37, 45:37, 50:38, 55:39, 60:40, 65:40, 70:41, 75:42, 80:43, 85:43, 90:44, 95:45},
    38: {0:32, 5:33, 10:34, 15:35, 20:35, 25:36, 30:37, 35:38, 40:39, 45:39, 50:40, 55:41, 60:41, 65:42, 70:43, 75:44, 80:44, 85:45, 90:46, 95:47},
    40: {0:34, 5:35, 10:36, 15:36, 20:37, 25:38, 30:39, 35:39, 40:40, 45:41, 50:42, 55:42, 60:43, 65:44, 70:45, 75:45, 80:46, 85:47, 90:48, 95:48},
    42: {0:36, 5:36, 10:37, 15:38, 20:39, 25:39, 30:40, 35:41, 40:42, 45:42, 50:43, 55:44, 60:45, 65:45, 70:46, 75:47, 80:48, 85:48, 90:49, 95:50},
    44: {0:37, 5:38, 10:39, 15:40, 20:40, 25:41, 30:42, 35:43, 40:43, 45:44, 50:45, 55:46, 60:46, 65:47, 70:48, 75:49, 80:49, 85:50, 90:51, 95:52},
    46: {0:39, 5:40, 10:41, 15:41, 20:42, 25:43, 30:44, 35:44, 40:45, 45:46, 50:47, 55:47, 60:48, 65:49, 70:50, 75:50, 80:51, 85:52, 90:53, 95:53},
    48: {0:41, 5:42, 10:42, 15:43, 20:44, 25:45, 30:45, 35:46, 40:47, 45:48, 50:48, 55:49, 60:50, 65:51, 70:51, 75:52, 80:53, 85:54, 90:54, 95:55},
    50: {0:43, 5:43, 10:44, 15:45, 20:45, 25:46, 30:47, 35:48, 40:49, 45:49, 50:50, 55:51, 60:52, 65:52, 70:53, 75:54, 80:55, 85:55, 90:56, 95:57},
    52: {0:44, 5:45, 10:46, 15:46, 20:47, 25:48, 30:49, 35:49, 40:50, 45:51, 50:52, 55:52, 60:53, 65:54, 70:55, 75:55, 80:56, 85:57, 90:58, 95:58},
    54: {0:46, 5:47, 10:47, 15:48, 20:49, 25:50, 30:50, 35:51, 40:52, 45:53, 50:53, 55:54, 60:55, 65:56, 70:55, 75:57, 80:58, 85:59, 90:59, 95:60},
    56: {0:48, 5:48, 10:49, 15:50, 20:51, 25:51, 30:52, 35:53, 40:54, 45:54, 50:55, 55:56, 60:57, 65:57, 70:58, 75:59, 80:60, 85:60, 90:61, 95:62},
    58: {0:49, 5:50, 10:51, 15:52, 20:52, 25:53, 30:54, 35:55, 40:55, 45:56, 50:57, 55:58, 60:58, 65:59, 70:60, 75:61, 80:61, 85:62, 90:63, 95:64},
    60: {0:51, 5:52, 10:53, 15:53, 20:54, 25:55, 30:56, 35:56, 40:57, 45:58, 50:59, 55:59, 60:60, 65:61, 70:62, 75:62, 80:63, 85:64, 90:65, 95:65},
    62: {0:53, 5:53, 10:54, 15:55, 20:56, 25:56, 30:57, 35:58, 40:59, 45:59, 50:60, 55:61, 60:62, 65:62, 70:63, 75:64, 80:65, 85:65, 90:66, 95:67},
    64: {0:54, 5:55, 10:56, 15:57, 20:57, 25:58, 30:59, 35:60, 40:60, 45:61, 50:62, 55:63, 60:63, 65:64, 70:65, 75:66, 80:66, 85:67, 90:68, 95:69},
    66: {0:56, 5:57, 10:58, 15:58, 20:59, 25:60, 30:61, 35:61, 40:62, 45:63, 50:64, 55:64, 60:65, 65:66, 70:67, 75:67, 80:68, 85:69, 90:70, 95:70},
    68: {0:58, 5:59, 10:59, 15:60, 20:61, 25:62, 30:62, 35:63, 40:64, 45:65, 50:65, 55:66, 60:67, 65:68, 70:68, 75:69, 80:70, 85:71, 90:71, 95:72},
    70: {0:60, 5:60, 10:61, 15:62, 20:63, 25:63, 30:64, 35:65, 40:66, 45:66, 50:67, 55:68, 60:69, 65:69, 70:70, 75:71, 80:72, 85:72, 90:73, 95:74},
    72: {0:61, 5:62, 10:63, 15:63, 20:64, 25:65, 30:66, 35:66, 40:67, 45:68, 50:69, 55:70, 60:71, 65:72, 70:72, 75:73, 80:74, 85:75, 90:75, 95:75},
    74: {0:63, 5:64, 10:64, 15:65, 20:66, 25:67, 30:67, 35:68, 40:69, 45:70, 50:70, 55:71, 60:72, 65:73, 70:73, 75:74, 80:75, 85:76, 90:76, 95:77},
    76: {0:65, 5:65, 10:66, 15:67, 20:68, 25:68, 30:69, 35:70, 40:71, 45:71, 50:72, 55:73, 60:74, 65:74, 70:75, 75:76, 80:77, 85:77, 90:78, 95:79},
    78: {0:66, 5:67, 10:68, 15:69, 20:69, 25:70, 30:71, 35:72, 40:72, 45:73, 50:74, 55:75, 60:75, 65:76, 70:77, 75:78, 80:78, 85:79, 90:80, 95:81},
    80: {0:68, 5:69, 10:70, 15:70, 20:71, 25:72, 30:73, 35:73, 40:74, 45:75, 50:76, 55:76, 60:77, 65:78, 70:79, 75:79, 80:80, 85:81, 90:82, 95:82},
    82: {0:70, 5:70, 10:71, 15:72, 20:73, 25:73, 30:74, 35:75, 40:76, 45:76, 50:77, 55:78, 60:79, 65:79, 70:80, 75:81, 80:82, 85:82, 90:83, 95:84},
    84: {0:71, 5:72, 10:73, 15:74, 20:74, 25:75, 30:76, 35:77, 40:77, 45:78, 50:79, 55:80, 60:80, 65:81, 70:82, 75:83, 80:83, 85:84, 90:85, 95:86},
    86: {0:73, 5:74, 10:75, 15:75, 20:76, 25:77, 30:78, 35:78, 40:79, 45:80, 50:81, 55:81, 60:82, 65:83, 70:84, 75:84, 80:85, 85:86, 90:87, 95:87},
    88: {0:75, 5:76, 10:76, 15:77, 20:78, 25:79, 30:79, 35:80, 40:81, 45:82, 50:82, 55:83, 60:84, 65:85, 70:85, 75:86, 80:87, 85:88, 90:88, 95:89},
    90: {0:77, 5:77, 10:78, 15:79, 20:80, 25:80, 30:81, 35:82, 40:83, 45:83, 50:84, 55:85, 60:86, 65:86, 70:87, 75:88, 80:89, 85:89, 90:90, 95:91}
}

def get_ts(tw_f: float, tg_f: float) -> int:
    tw = int(tw_f / 2 + 0.5) * 2
    tg = int(tg_f / 5 + 0.5) * 5
    tw = max(36, min(90, tw))
    tg = max(0, min(95, tg))
    return temperature_control_table[tw][tg]

# --- Fonction pour créer un rapport PDF complet ---
def create_pdf_report(df, unit, figures_dict):
    """
    Crée un rapport PDF complet avec tous les tableaux et graphiques
    
    Args:
        df: DataFrame avec les données
        unit: Unité de mesure
        figures_dict: Dictionnaire contenant toutes les figures matplotlib
        
    Returns:
        Bytes du fichier PDF
    """
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Page de titre
        fig_title = plt.figure(figsize=(8.5, 11))
        fig_title.text(0.5, 0.7, 'Rapport d\'Analyse ERT', 
                      ha='center', va='center', fontsize=24, fontweight='bold')
        fig_title.text(0.5, 0.6, 'Ravensgate Sonic Water Level Meter', 
                      ha='center', va='center', fontsize=16)
        fig_title.text(0.5, 0.5, f'Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                      ha='center', va='center', fontsize=12)
        fig_title.text(0.5, 0.4, f'Total mesures: {len(df)}', 
                      ha='center', va='center', fontsize=12)
        fig_title.text(0.5, 0.35, f'Points de sondage: {df["survey_point"].nunique()}', 
                      ha='center', va='center', fontsize=12)
        fig_title.text(0.5, 0.3, f'Unité: {unit}', 
                      ha='center', va='center', fontsize=12)
        plt.axis('off')
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)
        
        # Page 2: Statistiques descriptives
        fig_stats = plt.figure(figsize=(8.5, 11))
        ax_stats = fig_stats.add_subplot(111)
        
        stats_data = [
            ['Total mesures', len(df)],
            ['Points de sondage', df['survey_point'].nunique()],
            ['Profondeurs uniques', df['depth'].nunique()],
            [f'DTW moyen ({unit})', f"{df['data'].mean():.2f}"],
            [f'DTW min ({unit})', f"{df['data'].min():.2f}"],
            [f'DTW max ({unit})', f"{df['data'].max():.2f}"],
            [f'Écart-type ({unit})', f"{df['data'].std():.2f}"],
        ]
        
        table_stats = ax_stats.table(cellText=stats_data, 
                                     colLabels=['Statistique', 'Valeur'],
                                     cellLoc='left', loc='center',
                                     colWidths=[0.6, 0.4])
        table_stats.auto_set_font_size(False)
        table_stats.set_fontsize(10)
        table_stats.scale(1, 2)
        ax_stats.axis('off')
        ax_stats.set_title('Statistiques descriptives', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig_stats, bbox_inches='tight')
        plt.close(fig_stats)
        
        # Page 3+: Statistiques par profondeur
        depth_stats = df.groupby('depth')['data'].agg(['mean', 'min', 'max', 'std']).round(2)
        
        fig_depth = plt.figure(figsize=(8.5, 11))
        ax_depth = fig_depth.add_subplot(111)
        
        depth_data = [[f"{idx:.1f}", f"{row['mean']:.2f}", f"{row['min']:.2f}", 
                      f"{row['max']:.2f}", f"{row['std']:.2f}"] 
                     for idx, row in depth_stats.iterrows()]
        
        table_depth = ax_depth.table(cellText=depth_data,
                                    colLabels=['Profondeur', 'Moyenne DTW', 'Min DTW', 'Max DTW', 'Écart-type'],
                                    cellLoc='center', loc='center',
                                    colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
        table_depth.auto_set_font_size(False)
        table_depth.set_fontsize(9)
        table_depth.scale(1, 1.5)
        ax_depth.axis('off')
        ax_depth.set_title(f'Statistiques par profondeur ({unit})', fontsize=16, fontweight='bold', pad=20)
        pdf.savefig(fig_depth, bbox_inches='tight')
        plt.close(fig_depth)
        
        # Ajouter toutes les figures fournies
        for fig_name, fig in figures_dict.items():
            if fig is not None:
                pdf.savefig(fig, bbox_inches='tight')
        
        # Métadonnées du PDF
        d = pdf.infodict()
        d['Title'] = 'Rapport Analyse ERT - Ravensgate Sonic'
        d['Author'] = 'ERTest Application'
        d['Subject'] = 'Analyse des niveaux d\'eau souterraine'
        d['Keywords'] = 'ERT, Ravensgate, Water Level, DTW'
        d['CreationDate'] = datetime.now()
    
    buffer.seek(0)
    return buffer.getvalue()

def create_stratigraphy_pdf_report(df, figures_strat_dict):
    """
    Crée un rapport PDF complet pour l'analyse stratigraphique
    
    Args:
        df: DataFrame avec les données de résistivité
        figures_strat_dict: Dictionnaire contenant toutes les figures stratigraphiques
        
    Returns:
        Bytes du fichier PDF
    """
    buffer = io.BytesIO()
    
    with PdfPages(buffer) as pdf:
        # Page 1: Page de titre
        fig_title = plt.figure(figsize=(8.5, 11), dpi=150)
        fig_title.text(0.5, 0.75, '🪨 RAPPORT STRATIGRAPHIQUE COMPLET', 
                      ha='center', va='center', fontsize=22, fontweight='bold')
        fig_title.text(0.5, 0.68, 'Classification Géologique avec Résistivités', 
                      ha='center', va='center', fontsize=16, style='italic')
        fig_title.text(0.5, 0.6, f'📅 Date: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 
                      ha='center', va='center', fontsize=12)
        
        # Statistiques du sondage
        rho_data = pd.to_numeric(df['data'], errors='coerce').dropna()
        depth_data = np.abs(pd.to_numeric(df['depth'], errors='coerce').dropna())
        
        fig_title.text(0.5, 0.5, '📊 RÉSUMÉ DES DONNÉES', 
                      ha='center', va='center', fontsize=14, fontweight='bold')
        fig_title.text(0.5, 0.44, f'Nombre total de mesures: {len(df)}', 
                      ha='center', va='center', fontsize=11)
        fig_title.text(0.5, 0.40, f'Profondeur maximale: {depth_data.max():.3f} m (≈{depth_data.max()*1000:.0f} mm)', 
                      ha='center', va='center', fontsize=11)
        fig_title.text(0.5, 0.36, f'Résistivité min: {rho_data.min():.3f} Ω·m', 
                      ha='center', va='center', fontsize=11)
        fig_title.text(0.5, 0.32, f'Résistivité max: {rho_data.max():.0f} Ω·m', 
                      ha='center', va='center', fontsize=11)
        fig_title.text(0.5, 0.28, f'Résistivité moyenne: {rho_data.mean():.2f} Ω·m', 
                      ha='center', va='center', fontsize=11)
        
        # Catégories identifiées
        fig_title.text(0.5, 0.18, '🎯 CATÉGORIES GÉOLOGIQUES IDENTIFIÉES', 
                      ha='center', va='center', fontsize=12, fontweight='bold')
        
        categories = [
            ('💧 Eaux', (0.1, 1000)),
            ('🧱 Argiles & Sols saturés', (1, 100)),
            ('🏖️ Sables & Graviers', (50, 1000)),
            ('🪨 Roches sédimentaires', (100, 5000)),
            ('🌋 Roches ignées', (1000, 100000)),
            ('💎 Minéraux & Minerais', (0.001, 1000000))
        ]
        
        y_pos = 0.12
        for cat_name, (rho_min, rho_max) in categories:
            mask = (rho_data >= rho_min) & (rho_data <= rho_max)
            count = mask.sum()
            if count > 0:
                fig_title.text(0.5, y_pos, f'{cat_name}: {count} mesures', 
                              ha='center', va='center', fontsize=9)
                y_pos -= 0.03
        
        fig_title.text(0.5, 0.02, '© Belikan M. - Analyse ERT - Novembre 2025', 
                      ha='center', va='center', fontsize=8, style='italic', color='gray')
        plt.axis('off')
        pdf.savefig(fig_title, bbox_inches='tight')
        plt.close(fig_title)
        
        # Ajouter toutes les figures du dictionnaire
        for fig_name, fig in figures_strat_dict.items():
            pdf.savefig(fig, bbox_inches='tight', dpi=150)
            plt.close(fig)
        
        # Métadonnées du PDF
        d = pdf.infodict()
        d['Title'] = 'Rapport Stratigraphique Complet'
        d['Author'] = 'Belikan M. - ERTest Application'
        d['Subject'] = 'Classification géologique par résistivité électrique'
        d['Keywords'] = 'ERT, Stratigraphie, Résistivité, Géologie, Minéraux'
        d['CreationDate'] = datetime.now()
    
    buffer.seek(0)
    return buffer.getvalue()

# --- Parsing .dat robuste avec cache ---
@st.cache_data
def detect_encoding(file_bytes):
    """Détecte l'encodage depuis les bytes du fichier"""
    result = chardet.detect(file_bytes[:100000])
    return result['encoding'] or 'utf-8'

@st.cache_data
def parse_dat(file_content, encoding):
    """Parse le contenu du fichier .dat avec mise en cache"""
    try:
        from io import StringIO
        df = pd.read_csv(
            StringIO(file_content.decode(encoding)), 
            delim_whitespace=True, header=None, comment='#',
            names=['survey_point', 'depth', 'data', 'project'],
            on_bad_lines='skip', engine='python'
        )
        df['survey_point'] = pd.to_numeric(df['survey_point'], errors='coerce')
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
        df['data'] = pd.to_numeric(df['data'], errors='coerce')
        df = df.dropna(subset=['survey_point', 'depth', 'data'])
        return df
    except Exception as e:
        st.error(f"Erreur parsing : {e}")
        return pd.DataFrame()

# --- Tableau des types d'eau ---
water_html = """
<style>
.water-table th { background-color: #333; color: white; padding: 12px; text-align: center; }
.water-table td { padding: 12px; text-align: center; border-bottom: 1px solid #ddd; }
</style>
<table class="water-table" style="width:100%; border-collapse: collapse; margin: 20px 0;">
  <tr>
    <th>Type d'eau</th>
    <th>Résistivité (Ω.m)</th>
    <th>Couleur associée</th>
    <th>Description</th>
  </tr>
  <tr style="background-color: #FF4500; color: white;">
    <td><strong>Eau de mer</strong></td>
    <td>0.1 – 1</td>
    <td>Rouge vif / Orange</td>
    <td>Eau océanique hautement salée (∼35 g/L de sel). Très forte conductivité électrique due aux ions Na⁺ et Cl⁻. Typique des mers et océans.</td>
  </tr>
  <tr style="background-color: #FFD700; color: black;">
    <td><strong>Eau salée (nappe)</strong></td>
    <td>1 – 10</td>
    <td>Jaune / Orange</td>
    <td>Eau saumâtre dans les nappes phréatiques côtières (intrusion saline). Salinité intermédiaire, souvent non potable sans traitement.</td>
  </tr>
  <tr style="background-color: #90EE90; color: black;">
    <td><strong>Eau douce</strong></td>
    <td>10 – 100</td>
    <td>Vert / Bleu clair</td>
    <td>Eau potable standard (rivières, lacs, nappes intérieures). Faiblement minéralisée, conductivité modérée.</td>
  </tr>
  <tr style="background-color: #00008B; color: white;">
    <td><strong>Eau très pure</strong></td>
    <td>> 100</td>
    <td>Bleu foncé</td>
    <td>Eau ultra-pure (distillée, déminéralisée, pluie). Presque pas d'ions → très faible conductivité. Utilisée en laboratoire/industrie.</td>
  </tr>
</table>
"""

# --- Tableau complet des matériaux géologiques (sols, roches, minéraux et eaux) ---
geology_html = """
<style>
.geo-table th { background-color: #1e3a8a; color: white; padding: 10px; text-align: center; font-weight: bold; }
.geo-table td { padding: 10px; text-align: center; border-bottom: 1px solid #ccc; }
.geo-table tr:hover { background-color: #f0f0f0; }
</style>
<table class="geo-table" style="width:100%; border-collapse: collapse; margin: 20px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
  <tr>
    <th colspan="5" style="background-color: #0f172a; font-size: 18px;">📊 CLASSIFICATION COMPLÈTE DES RÉSISTIVITÉS GÉOLOGIQUES</th>
  </tr>
  <tr>
    <th>Catégorie</th>
    <th>Matériau</th>
    <th>Résistivité (Ω.m)</th>
    <th>Couleur</th>
    <th>Description / Usage</th>
  </tr>
  
  <!-- EAUX -->
  <tr style="background-color: #fef3c7;">
    <td rowspan="4" style="background-color: #3b82f6; color: white; font-weight: bold; vertical-align: middle;">💧<br>EAUX</td>
    <td><strong>Eau de mer</strong></td>
    <td>0.1 – 1</td>
    <td style="background-color: #FF4500; color: white;">🔴 Rouge</td>
    <td>Océans, forte salinité (35 g/L NaCl)</td>
  </tr>
  <tr style="background-color: #fef3c7;">
    <td><strong>Eau salée/saumâtre</strong></td>
    <td>1 – 10</td>
    <td style="background-color: #FFD700;">🟡 Jaune-Orange</td>
    <td>Nappes côtières, intrusion saline</td>
  </tr>
  <tr style="background-color: #fef3c7;">
    <td><strong>Eau douce</strong></td>
    <td>10 – 100</td>
    <td style="background-color: #90EE90;">🟢 Vert-Bleu clair</td>
    <td>Nappes phréatiques, rivières, lacs</td>
  </tr>
  <tr style="background-color: #fef3c7;">
    <td><strong>Eau ultra-pure</strong></td>
    <td>100 – 1000</td>
    <td style="background-color: #00008B; color: white;">🔵 Bleu foncé</td>
    <td>Eau distillée, pluie, laboratoire</td>
  </tr>
  
  <!-- SOLS SATURÉS / ARGILES -->
  <tr style="background-color: #fee2e2;">
    <td rowspan="3" style="background-color: #dc2626; color: white; font-weight: bold; vertical-align: middle;">🧱<br>ARGILES<br>& SOLS<br>SATURÉS</td>
    <td><strong>Argile marine saturée</strong></td>
    <td>1 – 10</td>
    <td style="background-color: #8B4513; color: white;">🟤 Brun rouge</td>
    <td>Très conductrice, riche en sels</td>
  </tr>
  <tr style="background-color: #fee2e2;">
    <td><strong>Argile compacte humide</strong></td>
    <td>10 – 50</td>
    <td style="background-color: #A0522D; color: white;">🟫 Brun</td>
    <td>Formations imperméables, rétention d'eau</td>
  </tr>
  <tr style="background-color: #fee2e2;">
    <td><strong>Limon/Silt saturé</strong></td>
    <td>20 – 100</td>
    <td style="background-color: #D2B48C;">🟨 Beige</td>
    <td>Sol fin avec eau interstitielle</td>
  </tr>
  
  <!-- SABLES ET GRAVIERS -->
  <tr style="background-color: #fef9c3;">
    <td rowspan="3" style="background-color: #eab308; font-weight: bold; vertical-align: middle;">🏖️<br>SABLES<br>& GRAVIERS</td>
    <td><strong>Sable saturé (eau douce)</strong></td>
    <td>50 – 200</td>
    <td style="background-color: #F4A460;">🟧 Sable</td>
    <td>Aquifère perméable, bon pour puits</td>
  </tr>
  <tr style="background-color: #fef9c3;">
    <td><strong>Sable sec</strong></td>
    <td>200 – 1000</td>
    <td style="background-color: #FFE4B5;">🟨 Beige clair</td>
    <td>Zone non saturée, faible conductivité</td>
  </tr>
  <tr style="background-color: #fef9c3;">
    <td><strong>Gravier saturé</strong></td>
    <td>100 – 500</td>
    <td style="background-color: #BDB76B;">⚫ Gris-vert</td>
    <td>Très perméable, aquifère productif</td>
  </tr>
  
  <!-- ROCHES SÉDIMENTAIRES -->
  <tr style="background-color: #e0e7ff;">
    <td rowspan="4" style="background-color: #6366f1; color: white; font-weight: bold; vertical-align: middle;">🪨<br>ROCHES<br>SÉDIMEN-<br>TAIRES</td>
    <td><strong>Calcaire fissuré (saturé)</strong></td>
    <td>100 – 1000</td>
    <td style="background-color: #D3D3D3;">⚪ Gris clair</td>
    <td>Karst, aquifère calcaire, grottes</td>
  </tr>
  <tr style="background-color: #e0e7ff;">
    <td><strong>Calcaire compact</strong></td>
    <td>1000 – 5000</td>
    <td style="background-color: #C0C0C0;">⚪ Gris</td>
    <td>Peu poreux, faible perméabilité</td>
  </tr>
  <tr style="background-color: #e0e7ff;">
    <td><strong>Grès poreux saturé</strong></td>
    <td>200 – 2000</td>
    <td style="background-color: #DAA520;">🟫 Or terne</td>
    <td>Réservoir aquifère important</td>
  </tr>
  <tr style="background-color: #e0e7ff;">
    <td><strong>Schiste argileux</strong></td>
    <td>10 – 100</td>
    <td style="background-color: #696969; color: white;">⚫ Gris foncé</td>
    <td>Conducteur, riche en minéraux argileux</td>
  </tr>
  
  <!-- ROCHES IGNÉES ET MÉTAMORPHIQUES -->
  <tr style="background-color: #fce7f3;">
    <td rowspan="4" style="background-color: #ec4899; color: white; font-weight: bold; vertical-align: middle;">🌋<br>ROCHES<br>IGNÉES<br>& MÉTA.</td>
    <td><strong>Granite</strong></td>
    <td>5000 – 100000</td>
    <td style="background-color: #FFB6C1;">🩷 Rose</td>
    <td>Très résistif, socle cristallin</td>
  </tr>
  <tr style="background-color: #fce7f3;">
    <td><strong>Basalte compact</strong></td>
    <td>1000 – 10000</td>
    <td style="background-color: #2F4F4F; color: white;">⚫ Noir-gris</td>
    <td>Roche volcanique dense</td>
  </tr>
  <tr style="background-color: #fce7f3;">
    <td><strong>Basalte fracturé (saturé)</strong></td>
    <td>200 – 2000</td>
    <td style="background-color: #556B2F; color: white;">🟢 Vert sombre</td>
    <td>Aquifère volcanique</td>
  </tr>
  <tr style="background-color: #fce7f3;">
    <td><strong>Quartzite</strong></td>
    <td>10000 – 100000</td>
    <td style="background-color: #F5F5DC;">⚪ Blanc cassé</td>
    <td>Métamorphique, très résistant</td>
  </tr>
  
  <!-- MINÉRAUX SPÉCIAUX -->
  <tr style="background-color: #ddd6fe;">
    <td rowspan="3" style="background-color: #7c3aed; color: white; font-weight: bold; vertical-align: middle;">💎<br>MINÉRAUX<br>& ORES</td>
    <td><strong>Minerais métalliques (cuivre, or)</strong></td>
    <td>0.01 – 1</td>
    <td style="background-color: #FFD700;">🟡 Doré</td>
    <td>Très conducteurs, cibles minières</td>
  </tr>
  <tr style="background-color: #ddd6fe;">
    <td><strong>Graphite</strong></td>
    <td>0.001 – 0.1</td>
    <td style="background-color: #000000; color: white;">⚫ Noir</td>
    <td>Extrêmement conducteur</td>
  </tr>
  <tr style="background-color: #ddd6fe;">
    <td><strong>Quartz pur</strong></td>
    <td>> 100000</td>
    <td style="background-color: #FFFFFF; border: 2px solid #000;">⚪ Transparent</td>
    <td>Isolant électrique parfait</td>
  </tr>
</table>
"""

# --- Seed pour reproductibilité des exemples ---
np.random.seed(42)

# --- Interface Streamlit ---
st.set_page_config(
    page_title="SETRAF - Subaquifère ERT Analysis", 
    page_icon="💧",
    layout="wide", 
    initial_sidebar_state="expanded"
)
st.title("💧 SETRAF - Subaquifère ERT Analysis Tool (08 Novembre 2025)")

tab1, tab2, tab3, tab4 = st.tabs([
    "🌡️ Calculateur Réglage Température", 
    "📊 Analyse Fichiers .dat", 
    "🌍 ERT Pseudo-sections 2D/3D",
    "🪨 Stratigraphie Complète (Sols + Eaux)"
])

# ===================== TAB 1 : TEMPÉRATURE =====================
with tab1:
    st.header("Calculateur de réglage Ts (Table officielle Ravensgate)")
    st.markdown("""
    Entrez la température de l'eau du puits (**Tw**) et la température moyenne quotidienne de surface (**Tg**).  
    L'app arrondit **conventionnellement (half-up)** aux pas du tableau et clamp automatiquement.
    
    **Exemple du manuel** : Tw = 58 °F (14 °C), Tg = 85 °F (29 °C) → **Ts = 62 °F** (17 °C).
    """)

    unit = st.radio("Unité", options=["°F", "°C"], horizontal=True)

    if unit == "°C":
        col1, col2 = st.columns(2)
        with col1:
            tw_c = st.number_input("Tw – Température eau puits (°C)", value=10.0, min_value=-10.0, max_value=50.0, step=0.1)
        with col2:
            tg_c = st.number_input("Tg – Température surface moyenne (°C)", value=20.0, min_value=-30.0, max_value=50.0, step=0.1)
        tw_f = tw_c * 9/5 + 32
        tg_f = tg_c * 9/5 + 32
    else:
        col1, col2 = st.columns(2)
        with col1:
            tw_f = st.number_input("Tw – Température eau puits (°F)", value=60.0, min_value=20.0, max_value=120.0, step=0.5)
        with col2:
            tg_f = st.number_input("Tg – Température surface moyenne (°F)", value=70.0, min_value=-20.0, max_value=120.0, step=0.5)

    if st.button("🔥 Calculer Ts", type="primary", use_container_width=True):
        ts = get_ts(tw_f, tg_f)
        tw_used = max(36, min(90, int(tw_f / 2 + 0.5) * 2))
        tg_used = max(0, min(95, int(tg_f / 5 + 0.5) * 5))

        st.success(f"**Réglage recommandé sur l'appareil → Ts = {ts} °F**")

        if unit == "°C":
            st.info(f"Tw utilisée → {tw_used} °F ({(tw_used - 32)*5/9:.1f} °C) | Tg utilisée → {tg_used} °F ({(tg_used - 32)*5/9:.1f} °C)")
        else:
            st.info(f"Tw utilisée → {tw_used} °F | Tg utilisée → {tg_used} °F")

    with st.expander("📋 Tableau complet Ravensgate (cliquer pour déplier)"):
        tg_cols = list(range(0, 96, 5))
        df_table = pd.DataFrame.from_dict(temperature_control_table, orient='index', columns=tg_cols)
        df_table.index.name = "Tw \\ Tg"
        df_table = df_table.sort_index()
        df_table.insert(0, "Tw (°F)", df_table.index)
        st.dataframe(df_table.style.background_gradient(cmap='coolwarm', axis=None), use_container_width=True)

    with st.expander("💧 Valeurs typiques pour l'eau – Résistivité & Couleurs associées"):
        st.markdown("### **2. Valeurs typiques pour l'eau**")
        st.markdown(water_html, unsafe_allow_html=True)
        st.caption("Ces valeurs sont indicatives. Les couleurs sont couramment utilisées dans les cartes de résistivité électrique (ERT) pour visualiser la salinité/qualité de l'eau souterraine.")

# ===================== TAB 2 : ANALYSE .DAT =====================
with tab2:
    st.header("2 Analyse de fichiers .dat de Ravensgate Sonic Water Level Meter")
    
    st.markdown("""
    ### Format attendu dans le .dat :
    - **Date** : Format YYYY/MM/DD HH:MM:SS
    - **Survey Point** (Point de forage)
    - **Depth From** et **Depth To** (Profondeur de mesure)
    - **Data** : Niveau d'eau (DTW - Depth To Water)
    """)
    
    # Initialiser l'état de session
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = None
    
    uploaded_file = st.file_uploader("📂 Uploader un fichier .dat", type=["dat"])
    
    if uploaded_file is not None:
        # Lire le contenu du fichier en bytes (avec cache)
        file_bytes = uploaded_file.read()
        encoding = detect_encoding(file_bytes)
        
        # Parser le fichier (avec cache)
        df = parse_dat(file_bytes, encoding)
        
        # Déterminer l'unité
        unit = 'm'  # Par défaut
        
        if not df.empty:
            st.success(f"✅ {len(df)} lignes chargées avec succès")
            
            # Sauvegarder dans l'état de session pour l'onglet 3
            st.session_state['uploaded_data'] = df.copy()
            st.session_state['unit'] = unit
            
            # Affichage du DataFrame
            st.dataframe(df.head(50), use_container_width=True)
            
            # Statistiques de base
            st.subheader("📊 Statistiques descriptives")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total mesures", len(df))
            with col2:
                st.metric("Points de sondage", df['survey_point'].nunique())
            with col3:
                st.metric(f"DTW moyen ({unit})", f"{df['data'].mean():.2f}")
            with col4:
                st.metric(f"DTW max ({unit})", f"{df['data'].max():.2f}")
            
            # Graphique temporel
            st.subheader("📈 Évolution temporelle du niveau d'eau")
            
            # Dictionnaire pour stocker toutes les figures
            figures_dict = {}
            
            # Vérifier si colonne 'date' existe
            if 'date' in df.columns:
                fig_time, ax = plt.subplots(figsize=(12, 5), dpi=150)
                for sp in sorted(df['survey_point'].unique()):
                    subset = df[df['survey_point'] == sp]
                    ax.plot(subset['date'], subset['data'], marker='o', label=f'SP {int(sp)}', markersize=4)
                ax.set_xlabel('Date', fontsize=11)
                ax.set_ylabel(f'DTW ({unit})', fontsize=11)
                ax.set_title('Niveau d\'eau par point de sondage', fontsize=13, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig_time)
                
                # Sauvegarder pour PDF
                figures_dict['temporal_evolution'] = fig_time
            else:
                st.info("⚠️ Pas de colonne 'date' dans le fichier - graphique temporel indisponible")
                fig_time = None
            
            # Détection d'anomalies
            st.subheader("🔍 Détection d'anomalies (K-Means)")
            n_clusters = st.slider("Nombre de clusters", 2, 5, 3, key='kmeans_slider')
            
            # Cache du calcul KMeans basé sur les données + nombre de clusters
            @st.cache_data
            def compute_kmeans(data_hash, n_clust):
                """Calcul KMeans avec cache"""
                X = df[['survey_point', 'depth', 'data']].values
                kmeans = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
                return kmeans.fit_predict(X)
            
            # Hash unique des données pour invalidation du cache
            data_hash = hash(tuple(df[['survey_point', 'depth', 'data']].values.flatten()))
            clusters = compute_kmeans(data_hash, n_clusters)
            df_viz = df.copy()
            df_viz['cluster'] = clusters
            
            fig_cluster, ax = plt.subplots(figsize=(12, 6), dpi=150)
            scatter = ax.scatter(df_viz['survey_point'], df_viz['depth'], c=df_viz['cluster'], 
                                cmap='viridis', s=50, alpha=0.6, edgecolors='black', linewidths=0.5)
            plt.colorbar(scatter, ax=ax, label='Cluster')
            ax.set_xlabel('Point de sondage', fontsize=11)
            ax.set_ylabel(f'Profondeur ({unit})', fontsize=11)
            ax.set_title(f'Classification en {n_clusters} groupes', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_cluster)
            
            # Sauvegarder pour PDF
            figures_dict['kmeans_clustering'] = fig_cluster
            
            # Coupe de niveaux d'eau avec couleurs de résistivité
            st.subheader("🌊 Coupe géologique - Niveaux d'eau avec résistivité")
            
            # Préparer les données pour la coupe
            survey_points = sorted(df['survey_point'].unique())
            depths = sorted(df['depth'].unique())
            
            if len(survey_points) >= 2 and len(depths) >= 2:
                # Créer une grille 2D
                from scipy.interpolate import griddata
                
                X_grid = []
                Z_grid = []
                DTW_grid = []
                
                for sp in survey_points:
                    for depth in depths:
                        subset = df[(df['survey_point'] == sp) & (df['depth'] == depth)]
                        if len(subset) > 0:
                            X_grid.append(float(sp))
                            Z_grid.append(abs(float(depth)))
                            DTW_grid.append(float(subset['data'].values[0]))
                
                X_grid = np.array(X_grid)
                Z_grid = np.array(Z_grid)
                DTW_grid = np.array(DTW_grid)
                
                # Interpolation pour avoir une grille lisse
                xi = np.linspace(X_grid.min(), X_grid.max(), 150)
                zi = np.linspace(Z_grid.min(), Z_grid.max(), 100)
                Xi, Zi = np.meshgrid(xi, zi)
                DTWi = griddata((X_grid, Z_grid), DTW_grid, (Xi, Zi), method='cubic')
                
                # Convertir DTW en résistivité apparente (simulation)
                # Plus le DTW est élevé, plus l'eau est profonde, donc moins conductrice
                # Résistivité ~ proportionnelle au DTW (valeurs indicatives)
                rho_apparent = np.where(DTWi < 5, 2,      # Eau très peu profonde → salée (2 Ω·m)
                                np.where(DTWi < 15, 8,     # Eau peu profonde → saumâtre (8 Ω·m)
                                np.where(DTWi < 30, 40,    # Eau moyenne profondeur → douce (40 Ω·m)
                                np.where(DTWi < 50, 150,   # Eau profonde → pure (150 Ω·m)
                                         500))))           # Très profond → roche sèche (500 Ω·m)
                
                # Créer la figure avec colormap ERT standard
                fig_water, ax_water = plt.subplots(figsize=(14, 7), dpi=150)
                
                # Utiliser jet_r (rouge=conducteur/salé, bleu=résistif/sec)
                pcm = ax_water.pcolormesh(Xi, Zi, rho_apparent, cmap='jet_r', 
                                         norm=LogNorm(vmin=1, vmax=1000), shading='auto')
                
                # Ajouter les points de mesure
                scatter = ax_water.scatter(X_grid, Z_grid, c=DTW_grid, cmap='coolwarm', 
                                          s=80, edgecolors='black', linewidths=1, 
                                          alpha=0.8, zorder=10, marker='o')
                
                # Colorbar pour la résistivité
                cbar = fig_water.colorbar(pcm, ax=ax_water, label='Résistivité apparente (Ω·m)', extend='both')
                
                ax_water.invert_yaxis()
                ax_water.set_xlabel('Point de sondage (Survey Point)', fontsize=11)
                ax_water.set_ylabel(f'Profondeur ({unit})', fontsize=11)
                ax_water.set_title('Coupe géologique - Distribution des niveaux d\'eau et résistivité', 
                                  fontsize=13, fontweight='bold')
                ax_water.grid(True, alpha=0.3, linestyle='--', color='white', linewidth=0.5)
                plt.tight_layout()
                
                st.pyplot(fig_water)
                
                # Sauvegarder pour PDF
                figures_dict['water_level_section'] = fig_water
                
                # Légende d'interprétation
                st.markdown(f"""
**Interprétation de la coupe :**
- 🔴 **Rouge/Orange** (1-10 Ω·m) : Eau salée/saumâtre - Nappe peu profonde (DTW < 15 {unit})
- 🟡 **Jaune** (10-100 Ω·m) : Eau douce - Nappe intermédiaire (DTW 15-30 {unit})
- 🟢 **Vert** (100-300 Ω·m) : Eau pure - Nappe profonde (DTW 30-50 {unit})
- 🔵 **Bleu** (>300 Ω·m) : Roche sèche/résistive - Niveau très profond (DTW > 50 {unit})

**Points noirs** : Mesures réelles du fichier .dat (colorés selon la profondeur)
                """)
            else:
                st.warning("⚠️ Pas assez de points de mesure pour créer une coupe 2D (minimum 2 points de sondage et 2 profondeurs)")
            
            # Coupes détaillées par type d'eau avec mesures réelles
            st.markdown("---")
            st.subheader("📊 Coupes détaillées par type d'eau - Mesures de résistivité réelles")
            
            # Afficher le tableau de référence
            st.markdown("""
            ### 📋 Tableau de référence - Valeurs typiques pour l'eau
            """)
            
            water_reference = pd.DataFrame({
                'Type d\'eau': ['Eau de mer', 'Eau salée (nappe)', 'Eau douce', 'Eau très pure'],
                'Résistivité (Ω.m)': ['0.1 - 1', '1 - 10', '10 - 100', '> 100'],
                'Couleur associée': ['🔴 Rouge vif / Orange', '🟡 Jaune / Orange', '🟢 Vert / Bleu clair', '🔵 Bleu foncé']
            })
            
            st.dataframe(water_reference, use_container_width=True, hide_index=True)
            
            # Coupe 1: Zone Eau de Mer (0.1 - 1 Ω·m)
            with st.expander("🔴 Coupe 1 - Zone d'eau de mer (0.1 - 1 Ω·m)", expanded=False):
                # Filtrer les données correspondant à cette plage
                seawater_mask = (df['data'] <= 1.0)
                if seawater_mask.sum() > 0:
                    df_sea = df[seawater_mask]
                    
                    fig_sea, ax_sea = plt.subplots(figsize=(14, 6), dpi=150)
                    
                    # Créer des données synthétiques représentatives
                    x_sea = np.linspace(0, 200, 100)
                    z_sea = np.linspace(0, 30, 60)
                    X_sea, Z_sea = np.meshgrid(x_sea, z_sea)
                    
                    # Résistivité pour eau de mer
                    rho_sea = np.ones_like(X_sea) * 0.5 + np.random.rand(*X_sea.shape) * 0.4
                    
                    pcm_sea = ax_sea.pcolormesh(X_sea, Z_sea, rho_sea, cmap='Reds', 
                                               vmin=0.1, vmax=1.0, shading='auto')
                    
                    # Ajouter les mesures réelles si disponibles
                    if len(df_sea) > 0:
                        ax_sea.scatter(df_sea['survey_point'], df_sea['depth'], 
                                      c='darkred', s=100, edgecolors='black', 
                                      linewidths=2, marker='s', zorder=10,
                                      label=f'Mesures réelles ({len(df_sea)} points)')
                    
                    fig_sea.colorbar(pcm_sea, ax=ax_sea, label='Résistivité (Ω.m)')
                    ax_sea.invert_yaxis()
                    ax_sea.set_xlabel('Distance (m, précision: mm)', fontsize=11)
                    ax_sea.set_ylabel('Profondeur (m, précision: mm)', fontsize=11)
                    ax_sea.set_title('Zone d\'eau de mer - Résistivité 0.1-1 Ω·m (Précision mm)', 
                                    fontsize=13, fontweight='bold')
                    ax_sea.legend(loc='upper right')
                    ax_sea.grid(True, alpha=0.3)
                    
                    # Définir ticks avec valeurs mesurées
                    if len(df_sea) > 0:
                        unique_depths_sea = np.unique(np.abs(df_sea['depth'].values))
                        unique_dist_sea = np.unique(df_sea['survey_point'].values)
                        
                        if len(unique_depths_sea) > 20:
                            ax_sea.set_yticks(unique_depths_sea[::len(unique_depths_sea)//20])
                        else:
                            ax_sea.set_yticks(unique_depths_sea)
                        
                        if len(unique_dist_sea) > 20:
                            ax_sea.set_xticks(unique_dist_sea[::len(unique_dist_sea)//20])
                        else:
                            ax_sea.set_xticks(unique_dist_sea)
                    
                    # Format des axes avec 3 décimales
                    ax_sea.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    ax_sea.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    
                    plt.tight_layout()
                    st.pyplot(fig_sea)
                    figures_dict['seawater_section'] = fig_sea
                    
                    st.markdown("""
                    **Caractéristiques :**
                    - **Résistivité** : 0.1 - 1 Ω·m
                    - **Couleur** : 🔴 Rouge vif / Orange
                    - **Description** : Eau océanique hautement salée (~35 g/L de sel)
                    - **Conductivité** : Très forte conductivité électrique due aux ions Na⁺ et Cl⁻
                    - **Contexte** : Typique des mers et océans, intrusion saline côtière
                    """)
                else:
                    st.info("Aucune mesure dans cette plage de résistivité dans vos données")
            
            # Coupe 2: Zone Eau Salée Nappe (1 - 10 Ω·m)
            with st.expander("🟡 Coupe 2 - Nappe d'eau salée (1 - 10 Ω·m)", expanded=False):
                saline_mask = (df['data'] > 1.0) & (df['data'] <= 10.0)
                if saline_mask.sum() > 0:
                    df_saline = df[saline_mask]
                    
                    fig_saline, ax_saline = plt.subplots(figsize=(14, 6), dpi=150)
                    
                    x_sal = np.linspace(0, 250, 120)
                    z_sal = np.linspace(0, 40, 70)
                    X_sal, Z_sal = np.meshgrid(x_sal, z_sal)
                    
                    # Gradient de résistivité pour nappe salée
                    rho_sal = 3 + np.random.rand(*X_sal.shape) * 5 + Z_sal * 0.05
                    rho_sal = np.clip(rho_sal, 1, 10)
                    
                    pcm_sal = ax_saline.pcolormesh(X_sal, Z_sal, rho_sal, cmap='YlOrRd', 
                                                  vmin=1, vmax=10, shading='auto')
                    
                    if len(df_saline) > 0:
                        ax_saline.scatter(df_saline['survey_point'], df_saline['depth'], 
                                        c='orange', s=100, edgecolors='black', 
                                        linewidths=2, marker='o', zorder=10,
                                        label=f'Mesures réelles ({len(df_saline)} points)')
                    
                    fig_saline.colorbar(pcm_sal, ax=ax_saline, label='Résistivité (Ω.m)')
                    ax_saline.invert_yaxis()
                    ax_saline.set_xlabel('Distance (m, précision: mm)', fontsize=11)
                    ax_saline.set_ylabel('Profondeur (m, précision: mm)', fontsize=11)
                    ax_saline.set_title('Nappe phréatique salée - Résistivité 1-10 Ω·m (Précision mm)', 
                                       fontsize=13, fontweight='bold')
                    ax_saline.legend(loc='upper right')
                    ax_saline.grid(True, alpha=0.3)
                    
                    # Définir ticks avec valeurs mesurées
                    if len(df_saline) > 0:
                        unique_depths_sal = np.unique(np.abs(df_saline['depth'].values))
                        unique_dist_sal = np.unique(df_saline['survey_point'].values)
                        
                        if len(unique_depths_sal) > 20:
                            ax_saline.set_yticks(unique_depths_sal[::len(unique_depths_sal)//20])
                        else:
                            ax_saline.set_yticks(unique_depths_sal)
                        
                        if len(unique_dist_sal) > 20:
                            ax_saline.set_xticks(unique_dist_sal[::len(unique_dist_sal)//20])
                        else:
                            ax_saline.set_xticks(unique_dist_sal)
                    
                    # Format des axes avec 3 décimales
                    ax_saline.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    ax_saline.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    
                    plt.tight_layout()
                    st.pyplot(fig_saline)
                    figures_dict['saline_section'] = fig_saline
                    
                    st.markdown("""
                    **Caractéristiques :**
                    - **Résistivité** : 1 - 10 Ω·m
                    - **Couleur** : 🟡 Jaune / Orange
                    - **Description** : Eau saumâtre dans les nappes phréatiques côtières
                    - **Salinité** : Intermédiaire, intrusion saline
                    - **Potabilité** : Souvent non potable sans traitement
                    - **Contexte** : Zones côtières, pollution par remontée saline
                    """)
                else:
                    st.info("Aucune mesure dans cette plage de résistivité dans vos données")
            
            # Coupe 3: Zone Eau Douce (10 - 100 Ω·m)
            with st.expander("🟢 Coupe 3 - Aquifère d'eau douce (10 - 100 Ω·m)", expanded=False):
                fresh_mask = (df['data'] > 10.0) & (df['data'] <= 100.0)
                if fresh_mask.sum() > 0:
                    df_fresh = df[fresh_mask]
                    
                    fig_fresh, ax_fresh = plt.subplots(figsize=(14, 6), dpi=150)
                    
                    x_fresh = np.linspace(0, 300, 140)
                    z_fresh = np.linspace(0, 50, 80)
                    X_fresh, Z_fresh = np.meshgrid(x_fresh, z_fresh)
                    
                    # Résistivité pour eau douce
                    rho_fresh = 30 + np.random.rand(*X_fresh.shape) * 50 + Z_fresh * 0.3
                    rho_fresh = np.clip(rho_fresh, 10, 100)
                    
                    pcm_fresh = ax_fresh.pcolormesh(X_fresh, Z_fresh, rho_fresh, cmap='YlGn', 
                                                   vmin=10, vmax=100, shading='auto')
                    
                    if len(df_fresh) > 0:
                        ax_fresh.scatter(df_fresh['survey_point'], df_fresh['depth'], 
                                       c='green', s=100, edgecolors='black', 
                                       linewidths=2, marker='D', zorder=10,
                                       label=f'Mesures réelles ({len(df_fresh)} points)')
                    
                    fig_fresh.colorbar(pcm_fresh, ax=ax_fresh, label='Résistivité (Ω.m)')
                    ax_fresh.invert_yaxis()
                    ax_fresh.set_xlabel('Distance (m, précision: mm)', fontsize=11)
                    ax_fresh.set_ylabel('Profondeur (m, précision: mm)', fontsize=11)
                    ax_fresh.set_title('Aquifère d\'eau douce - Résistivité 10-100 Ω·m (Précision mm)', 
                                      fontsize=13, fontweight='bold')
                    ax_fresh.legend(loc='upper right')
                    ax_fresh.grid(True, alpha=0.3)
                    
                    # Définir ticks avec valeurs mesurées
                    if len(df_fresh) > 0:
                        unique_depths_fresh = np.unique(np.abs(df_fresh['depth'].values))
                        unique_dist_fresh = np.unique(df_fresh['survey_point'].values)
                        
                        if len(unique_depths_fresh) > 20:
                            ax_fresh.set_yticks(unique_depths_fresh[::len(unique_depths_fresh)//20])
                        else:
                            ax_fresh.set_yticks(unique_depths_fresh)
                        
                        if len(unique_dist_fresh) > 20:
                            ax_fresh.set_xticks(unique_dist_fresh[::len(unique_dist_fresh)//20])
                        else:
                            ax_fresh.set_xticks(unique_dist_fresh)
                    
                    # Format des axes avec 3 décimales
                    ax_fresh.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    ax_fresh.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    
                    plt.tight_layout()
                    st.pyplot(fig_fresh)
                    figures_dict['freshwater_section'] = fig_fresh
                    
                    st.markdown("""
                    **Caractéristiques :**
                    - **Résistivité** : 10 - 100 Ω·m
                    - **Couleur** : 🟢 Vert / Bleu clair
                    - **Description** : Eau douce continentale (rivières, lacs, nappes)
                    - **Salinité** : Faible (< 1 g/L TDS)
                    - **Minéraux** : Calcium, magnésium, bicarbonates en faibles concentrations
                    - **Potabilité** : Généralement potable, bonne qualité
                    - **Contexte** : Aquifères captifs, zones agricoles, forêts
                    """)
                else:
                    st.info("Aucune mesure dans cette plage de résistivité dans vos données")
            
            # Coupe 4: Zone Eau Très Pure (> 100 Ω·m)
            with st.expander("🔵 Coupe 4 - Eau très pure / Roche sèche (> 100 Ω·m)", expanded=False):
                pure_mask = (df['data'] > 100.0)
                if pure_mask.sum() > 0:
                    df_pure = df[pure_mask]
                    
                    fig_pure, ax_pure = plt.subplots(figsize=(14, 6), dpi=150)
                    
                    x_pure = np.linspace(0, 200, 100)
                    z_pure = np.linspace(0, 60, 90)
                    X_pure, Z_pure = np.meshgrid(x_pure, z_pure)
                    
                    # Résistivité pour eau pure/roche
                    rho_pure = 200 + np.random.rand(*X_pure.shape) * 300 + Z_pure * 2
                    rho_pure = np.clip(rho_pure, 100, 1000)
                    
                    pcm_pure = ax_pure.pcolormesh(X_pure, Z_pure, rho_pure, cmap='Blues', 
                                                 shading='auto', 
                                                 norm=LogNorm(vmin=100, vmax=1000))
                    
                    if len(df_pure) > 0:
                        ax_pure.scatter(df_pure['survey_point'], df_pure['depth'], 
                                      c='darkblue', s=100, edgecolors='black', 
                                      linewidths=2, marker='^', zorder=10,
                                      label=f'Mesures réelles ({len(df_pure)} points)')
                    
                    fig_pure.colorbar(pcm_pure, ax=ax_pure, label='Résistivité (Ω.m)')
                    ax_pure.invert_yaxis()
                    ax_pure.set_xlabel('Distance (m, précision: mm)', fontsize=11)
                    ax_pure.set_ylabel('Profondeur (m, précision: mm)', fontsize=11)
                    ax_pure.set_title('Eau très pure / Roche résistive - Résistivité > 100 Ω·m (Précision mm)', 
                                     fontsize=13, fontweight='bold')
                    ax_pure.legend(loc='upper right')
                    ax_pure.grid(True, alpha=0.3)
                    
                    # Définir ticks avec valeurs mesurées
                    if len(df_pure) > 0:
                        unique_depths_pure = np.unique(np.abs(df_pure['depth'].values))
                        unique_dist_pure = np.unique(df_pure['survey_point'].values)
                        
                        if len(unique_depths_pure) > 20:
                            ax_pure.set_yticks(unique_depths_pure[::len(unique_depths_pure)//20])
                        else:
                            ax_pure.set_yticks(unique_depths_pure)
                        
                        if len(unique_dist_pure) > 20:
                            ax_pure.set_xticks(unique_dist_pure[::len(unique_dist_pure)//20])
                        else:
                            ax_pure.set_xticks(unique_dist_pure)
                    
                    # Format des axes avec 3 décimales
                    ax_pure.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    ax_pure.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    plt.tight_layout()
                    st.pyplot(fig_pure)
                    figures_dict['purewater_section'] = fig_pure
                    
                    st.markdown("""
                    **Caractéristiques :**
                    - **Résistivité** : > 100 Ω·m
                    - **Couleur** : 🔵 Bleu foncé
                    - **Description** : Eau très pure avec minéraux dissous très faibles
                    - **TDS** : < 50 mg/L (eau ultrapure)
                    - **Minéraux** : Quartz, feldspath, granite (roche cristalline)
                    - **Contexte** : Aquifères en socle cristallin, eau de fonte glaciaire, roche sèche
                    - **Propriétés** : Très peu d'ions, conductivité électrique minimale
                    """)
                else:
                    st.info("Aucune mesure dans cette plage de résistivité dans vos données")
            
            # Export
            st.subheader("💾 Exporter les résultats")
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 CSV", csv, "analysis.csv", "text/csv", key='download_csv')
            with col2:
                # Créer Excel uniquement à la demande (lazy loading)
                if st.button("� Préparer Excel", key='prepare_excel'):
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Data')
                    st.session_state['excel_buffer'] = buffer.getvalue()
                    st.success("✅ Excel prêt !")
                
                if 'excel_buffer' in st.session_state:
                    st.download_button("📥 Excel", st.session_state['excel_buffer'], 
                                      "analysis.xlsx", 
                                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                      key='download_excel')
            with col3:
                # Générer PDF avec tous les graphiques et tableaux
                if st.button("📄 Générer Rapport PDF", key='generate_pdf'):
                    with st.spinner('Génération du PDF en cours...'):
                        pdf_bytes = create_pdf_report(df, unit, figures_dict)
                        st.session_state['pdf_buffer'] = pdf_bytes
                        st.success("✅ PDF prêt !")
                
                if 'pdf_buffer' in st.session_state:
                    st.download_button(
                        "📥 PDF Complet",
                        st.session_state['pdf_buffer'],
                        f"rapport_ert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        "application/pdf",
                        key='download_pdf'
                    )
# ===================== TAB 3 : ERT PSEUDO-SECTIONS 2D/3D =====================
with tab3:
    st.header("4 Interprétation des pseudo-sections et modèles de résistivité (FicheERT.pdf)")

    st.subheader("4.1 Définition d'une pseudo-section")
    st.markdown("""
La première étape dans l'interprétation des données en tomographie électrique consiste à construire une **pseudo-section**. Une pseudo-section est une carte de résultat qui présente les valeurs des résistivités apparentes calculées à partir de la différence de potentiel mesurée aux bornes de deux électrodes de mesure ainsi que de la valeur du courant injecté entre les deux électrodes d'injection.

La couleur d'un point sur la pseudo-section représente donc la valeur de la résistivité apparente en ce point.
    """)

    # Vérifier si des données ont été chargées dans l'onglet 2
    if st.session_state.get('uploaded_data') is not None:
        df = st.session_state['uploaded_data']
        unit = st.session_state.get('unit', 'm')
        
        st.success(f"✅ Utilisation des données du fichier uploadé : {len(df)} mesures")
        
        st.markdown("**Pseudo-sections générées à partir de vos données réelles**")
        
        # Cache de la préparation des données 2D
        @st.cache_data
        def prepare_2d_data(data_hash):
            """Prépare les données pour visualisation 2D avec cache"""
            survey_points = sorted(df['survey_point'].unique())
            depths = sorted(df['depth'].unique())
            
            X_real = []
            Z_real = []
            Rho_real = []
            
            for sp in survey_points:
                for depth in depths:
                    subset = df[(df['survey_point'] == sp) & (df['depth'] == depth)]
                    if len(subset) > 0:
                        X_real.append(float(sp))
                        Z_real.append(abs(float(depth)))
                        Rho_real.append(float(subset['data'].values[0]))
            
            return np.array(X_real), np.array(Z_real), np.array(Rho_real)
        
        # Cache de l'interpolation (très coûteuse)
        @st.cache_data
        def interpolate_grid(X, Z, Rho, data_hash):
            """Interpolation cubique avec cache"""
            from scipy.interpolate import griddata
            xi = np.linspace(X.min(), X.max(), 100)
            zi = np.linspace(Z.min(), Z.max(), 50)
            Xi, Zi = np.meshgrid(xi, zi)
            Rhoi = griddata((X, Z), Rho, (Xi, Zi), method='cubic')
            return Xi, Zi, Rhoi, xi, zi
        
        # Hash unique des données
        data_hash = hash(tuple(df[['survey_point', 'depth', 'data']].values.flatten()))
        
        st.subheader("📊 Pseudo-section 2D - Données réelles du fichier .dat")
        
        # Dictionnaire pour stocker les figures du Tab 3
        figures_tab3 = {}
        
        # Préparer les données (avec cache)
        X_real, Z_real, Rho_real = prepare_2d_data(data_hash)
        
        # Interpoler (avec cache)
        Xi, Zi, Rhoi, xi, zi = interpolate_grid(X_real, Z_real, Rho_real, data_hash)
        
        # Pseudo-section 2D avec données réelles (haute résolution pour PDF)
        fig_real, ax = plt.subplots(figsize=(14, 7), dpi=150)
        
        # Utiliser une échelle de couleur adaptée aux valeurs d'eau
        vmin, vmax = Rho_real.min(), Rho_real.max()
        
        pcm = ax.pcolormesh(Xi, Zi, Rhoi, cmap='jet_r', shading='auto', 
                           vmin=vmin, vmax=vmax)
        
        # Ajouter les points de mesure réels
        scatter = ax.scatter(X_real, Z_real, c=Rho_real, cmap='jet_r', 
                            s=50, edgecolors='black', linewidths=0.5,
                            vmin=vmin, vmax=vmax, zorder=10)
        
        fig_real.colorbar(pcm, ax=ax, label=f'Niveau d\'eau DTW ({unit})', extend='both')
        ax.invert_yaxis()
        ax.set_xlabel('Point de sondage (Survey Point)', fontsize=11)
        ax.set_ylabel(f'Profondeur totale ({unit})', fontsize=11)
        ax.set_title(f'Pseudo-section 2D - Données réelles ({len(df)} mesures)', 
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        st.pyplot(fig_real)
        
        # Sauvegarder pour PDF
        figures_tab3['pseudo_section_2d'] = fig_real
        
        # Légende des couleurs basée sur les valeurs réelles
        st.markdown(f"""
**Interprétation des couleurs (basée sur vos données) :**
- Valeur minimale : **{vmin:.2f} {unit}** (niveau d'eau le plus bas) → couleur bleue
- Valeur moyenne : **{Rho_real.mean():.2f} {unit}** → couleur intermédiaire
- Valeur maximale : **{vmax:.2f} {unit}** (niveau d'eau le plus haut) → couleur rouge

Les zones rouges indiquent des niveaux d'eau plus élevés (DTW plus grand).
Les zones bleues indiquent des niveaux d'eau plus bas (nappe plus proche de la surface).
        """)
        
        # Vue 3D des données réelles
        survey_points = sorted(df['survey_point'].unique())
        depths = sorted(df['depth'].unique())
        
        if len(survey_points) > 2 and len(depths) > 2:
            st.subheader("🌐 Modèle 3D - Volume d'eau (données réelles)")
            
            fig3d_real = go.Figure(data=go.Scatter3d(
                x=X_real,
                y=np.zeros_like(X_real),  # Y=0 pour profil 2D
                z=-Z_real,  # Négatif pour afficher en profondeur
                mode='markers',
                marker=dict(
                    size=8,
                    color=Rho_real,
                    colorscale='Jet',
                    showscale=True,
                    colorbar=dict(title=f'DTW ({unit})'),
                    line=dict(width=0.5, color='black')
                ),
                text=[f'SP: {int(X_real[i])}<br>Depth: {Z_real[i]:.1f}{unit}<br>DTW: {Rho_real[i]:.2f}{unit}' 
                      for i in range(len(X_real))],
                hoverinfo='text'
            ))
            
            fig3d_real.update_layout(
                scene=dict(
                    xaxis_title='Point de sondage',
                    yaxis_title='Transect (m)',
                    zaxis_title=f'Profondeur ({unit})',
                    aspectmode='data'
                ),
                title='Visualisation 3D des mesures de niveau d\'eau',
                height=600
            )
            
            st.plotly_chart(fig3d_real, use_container_width=True)
        
        # Statistiques par profondeur
        st.subheader("📈 Analyse par profondeur")
        
        # Cache du calcul statistique
        @st.cache_data
        def compute_depth_stats(data_hash):
            """Calcul des statistiques par profondeur avec cache"""
            depth_stats = df.groupby('depth')['data'].agg(['mean', 'min', 'max', 'std']).round(2)
            depth_stats.columns = ['Moyenne DTW', 'Min DTW', 'Max DTW', 'Écart-type']
            return depth_stats
        
        depth_stats = compute_depth_stats(data_hash)
        st.dataframe(depth_stats.style.background_gradient(cmap='RdYlBu_r', axis=0), use_container_width=True)
        
        # Coupes comparatives avec mesures réelles incrustées
        st.markdown("---")
        st.subheader("🎯 Coupes comparatives - Mesures réelles vs Modèles théoriques")
        
        # Coupe comparative 1: Intrusion saline
        with st.expander("🌊 Coupe comparative 1 - Intrusion saline côtière avec mesures", expanded=False):
            fig_comp1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
            
            # Modèle théorique
            x_model = np.linspace(0, 300, 150)
            z_model = np.linspace(0, 40, 80)
            X_model, Z_model = np.meshgrid(x_model, z_model)
            
            # Gradient d'intrusion saline (mer vers terre)
            rho_model = np.ones_like(X_model) * 0.5  # Eau de mer
            rho_model[Z_model > 10 + 0.05 * X_model] = 3  # Eau salée nappe
            rho_model[Z_model > 25] = 50  # Eau douce profonde
            rho_model *= (1 + np.random.randn(*rho_model.shape) * 0.1)
            rho_model = np.clip(rho_model, 0.1, 100)
            
            # Graphique modèle
            pcm1 = ax1.pcolormesh(X_model, Z_model, rho_model, cmap='jet_r', 
                                 norm=LogNorm(vmin=0.1, vmax=100), shading='auto')
            ax1.invert_yaxis()
            ax1.set_title('Modèle théorique - Intrusion saline (Précision mm)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Distance depuis la côte (m, précision: mm)')
            ax1.set_ylabel('Profondeur (m, précision: mm)')
            
            # Format des axes avec 3 décimales
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            
            fig_comp1.colorbar(pcm1, ax=ax1, label='Résistivité (Ω.m)')
            
            # Annoter les zones
            ax1.text(50, 5, 'Eau de mer\n0.1-1 Ω·m', 
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                    fontsize=9, ha='center', color='white', fontweight='bold')
            ax1.text(150, 18, 'Eau salée\n1-10 Ω·m', 
                    bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7),
                    fontsize=9, ha='center', fontweight='bold')
            ax1.text(250, 32, 'Eau douce\n10-100 Ω·m', 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    fontsize=9, ha='center', fontweight='bold')
            
            # Données réelles
            if len(df) > 0:
                # Interpoler les données réelles - Conversion explicite en float
                X_real_data = pd.to_numeric(df['survey_point'], errors='coerce').values
                Z_real_data = np.abs(pd.to_numeric(df['depth'], errors='coerce').values)
                Rho_real_data = pd.to_numeric(df['data'], errors='coerce').values
                
                # Filtrer les valeurs NaN
                mask = ~(np.isnan(X_real_data) | np.isnan(Z_real_data) | np.isnan(Rho_real_data))
                X_real_data = X_real_data[mask]
                Z_real_data = Z_real_data[mask]
                Rho_real_data = Rho_real_data[mask]
                
                # Créer une grille pour les données réelles
                from scipy.interpolate import griddata
                if len(X_real_data) > 0:
                    xi_real = np.linspace(X_real_data.min(), X_real_data.max(), 100)
                    zi_real = np.linspace(Z_real_data.min(), Z_real_data.max(), 60)
                    Xi_real, Zi_real = np.meshgrid(xi_real, zi_real)
                    Rhoi_real = griddata((X_real_data, Z_real_data), Rho_real_data, 
                                        (Xi_real, Zi_real), method='cubic')
                    
                    pcm2 = ax2.pcolormesh(Xi_real, Zi_real, Rhoi_real, cmap='jet_r', 
                                         norm=LogNorm(vmin=max(0.1, Rho_real_data.min()), 
                                                     vmax=Rho_real_data.max()), shading='auto')
                    ax2.scatter(X_real_data, Z_real_data, c='black', s=50, 
                               edgecolors='white', linewidths=1.5, marker='o', zorder=10,
                               label=f'{len(X_real_data)} mesures')
                    ax2.invert_yaxis()
                    ax2.set_title(f'Données réelles - {len(X_real_data)} mesures (Précision mm)', 
                                 fontsize=12, fontweight='bold')
                    
                    # Format des axes avec 3 décimales
                    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    
                ax2.set_xlabel('Point de sondage (précision: mm)')
                ax2.set_ylabel('Profondeur (m, précision: mm)')
                ax2.legend(loc='upper right')
                fig_comp1.colorbar(pcm2, ax=ax2, label='Résistivité mesurée (Ω.m)')
            
            plt.tight_layout()
            st.pyplot(fig_comp1)
            figures_tab3['comparative_1'] = fig_comp1
            
            st.markdown("""
            **Analyse comparative :**
            - **Gauche** : Modèle théorique d'intrusion saline typique
            - **Droite** : Vos mesures réelles interpolées avec points de mesure (noirs)
            - Permet d'identifier les zones d'intrusion marine dans vos données
            """)
        
        # Coupe comparative 2: Aquifère multicouche
        with st.expander("🏔️ Coupe comparative 2 - Aquifère multicouche avec résistivités", expanded=False):
            fig_comp2, ax_multi = plt.subplots(figsize=(14, 7), dpi=150)
            
            # Créer un modèle multicouche
            x_multi = np.linspace(0, 250, 140)
            z_multi = np.linspace(0, 50, 90)
            X_multi, Z_multi = np.meshgrid(x_multi, z_multi)
            
            # Couches avec résistivités différentes
            rho_multi = np.ones_like(X_multi) * 200  # Sol sec surface
            rho_multi[(Z_multi > 8) & (Z_multi < 15)] = 60  # Aquifère peu profond (eau douce)
            rho_multi[(Z_multi >= 15) & (Z_multi < 25)] = 5  # Argile conductive
            rho_multi[(Z_multi >= 25) & (Z_multi < 40)] = 80  # Aquifère profond (eau douce)
            rho_multi[Z_multi >= 40] = 400  # Substrat rocheux
            
            # Ajouter du bruit
            rho_multi *= (1 + np.random.randn(*rho_multi.shape) * 0.08)
            rho_multi = np.clip(rho_multi, 1, 500)
            
            pcm_multi = ax_multi.pcolormesh(X_multi, Z_multi, rho_multi, cmap='jet_r', 
                                           norm=LogNorm(vmin=1, vmax=500), shading='auto')
            
            # Superposer les mesures réelles si disponibles
            if len(df) > 0:
                ax_multi.scatter(df['survey_point'], np.abs(df['depth']), 
                               c=df['data'], cmap='jet_r', s=120, 
                               edgecolors='black', linewidths=2, marker='s',
                               norm=LogNorm(vmin=max(0.1, df['data'].min()), 
                                          vmax=df['data'].max()),
                               zorder=10, label='Mesures réelles')
                
                # Annoter quelques points avec leurs valeurs
                for i in range(min(5, len(df))):
                    row = df.iloc[i]
                    ax_multi.annotate(f'{row["data"]:.2f} Ω·m\n@{np.abs(row["depth"]):.3f}m', 
                                    xy=(row['survey_point'], np.abs(row['depth'])),
                                    xytext=(10, 10), textcoords='offset points',
                                    fontsize=7, ha='left',
                                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            fig_comp2.colorbar(pcm_multi, ax=ax_multi, label='Résistivité (Ω.m)')
            ax_multi.invert_yaxis()
            ax_multi.set_xlabel('Distance (m, précision: mm)', fontsize=11)
            ax_multi.set_ylabel('Profondeur (m, précision: mm)', fontsize=11)
            
            # Format des axes avec 3 décimales
            ax_multi.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            ax_multi.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            
            ax_multi.set_title('Modèle multicouche avec mesures réelles (Précision mm)', 
                              fontsize=13, fontweight='bold')
            if len(df) > 0:
                ax_multi.legend(loc='upper right')
            ax_multi.grid(True, alpha=0.2, color='white', linestyle='--')
            
            # Ajouter légende des couches
            ax_multi.text(0.02, 0.98, 'Couches géologiques:', transform=ax_multi.transAxes,
                         fontsize=10, va='top', fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax_multi.text(0.02, 0.92, '• 0-8m: Sol sec (200 Ω·m)', transform=ax_multi.transAxes,
                         fontsize=8, va='top')
            ax_multi.text(0.02, 0.88, '• 8-15m: Aquifère peu profond (60 Ω·m)', transform=ax_multi.transAxes,
                         fontsize=8, va='top')
            ax_multi.text(0.02, 0.84, '• 15-25m: Argile conductive (5 Ω·m)', transform=ax_multi.transAxes,
                         fontsize=8, va='top')
            ax_multi.text(0.02, 0.80, '• 25-40m: Aquifère profond (80 Ω·m)', transform=ax_multi.transAxes,
                         fontsize=8, va='top')
            ax_multi.text(0.02, 0.76, '• >40m: Substrat rocheux (400 Ω·m)', transform=ax_multi.transAxes,
                         fontsize=8, va='top')
            
            plt.tight_layout()
            st.pyplot(fig_comp2)
            figures_tab3['comparative_2'] = fig_comp2
            
            st.markdown("""
            **Interprétation multicouche :**
            - **Carrés noirs** : Vos mesures réelles avec annotations de valeurs
            - **Fond coloré** : Modèle théorique multicouche
            - Les zones bleues (haute résistivité) indiquent des formations sèches ou rocheuses
            - Les zones rouges/orange (faible résistivité) indiquent de l'argile ou de l'eau salée
            - Les zones vertes/jaunes (résistivité moyenne) indiquent des aquifères d'eau douce
            """)
        
        # Export PDF des pseudo-sections
        st.subheader("📄 Export PDF des Pseudo-sections")
        col_pdf1, col_pdf2 = st.columns([1, 2])
        with col_pdf1:
            if st.button("📄 Générer PDF Pseudo-sections", key='generate_pdf_tab3'):
                with st.spinner('Génération du PDF des pseudo-sections...'):
                    pdf_bytes = create_pdf_report(df, unit, figures_tab3)
                    st.session_state['pdf_tab3_buffer'] = pdf_bytes
                    st.success("✅ PDF pseudo-sections prêt !")
        
        with col_pdf2:
            if 'pdf_tab3_buffer' in st.session_state:
                st.download_button(
                    "📥 Télécharger PDF Pseudo-sections",
                    st.session_state['pdf_tab3_buffer'],
                    f"pseudo_sections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    "application/pdf",
                    key='download_pdf_tab3'
                )
        
    else:
        st.warning("⚠️ Aucune donnée chargée. Veuillez d'abord uploader un fichier .dat dans l'onglet 'Analyse Fichiers .dat'")
        st.info("💡 Uploadez un fichier .dat dans l'onglet 'Analyse Fichiers .dat' pour visualiser vos données avec interprétation des couleurs de résistivité.")

# ===================== TAB 4 : STRATIGRAPHIE COMPLÈTE =====================
with tab4:
    st.header("🪨 Stratigraphie Complète - Classification Géologique avec Résistivités")
    
    st.markdown("""
    ### 📊 Vue d'ensemble des matériaux géologiques
    Cette section présente **toutes les formations géologiques** (eaux, sols, roches, minéraux) avec leurs résistivités caractéristiques.
    Cela permet d'identifier précisément la **nature des couches** à chaque niveau de profondeur.
    """)
    
    # Afficher le tableau complet
    st.markdown(geology_html, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Section graphiques de stratigraphie
    if 'uploaded_data' in st.session_state and st.session_state['uploaded_data'] is not None:
        df = st.session_state['uploaded_data']
        
        if len(df) > 0:
            st.subheader("🎨 Coupes Stratigraphiques Multi-Niveaux")
            st.markdown("""
            Ces coupes montrent la **distribution des matériaux géologiques** selon les valeurs de résistivité mesurées.
            Chaque plage de résistivité correspond à un type de matériau spécifique (eau, argile, sable, roche, etc.).
            """)
            
            # Créer les plages de résistivité étendues
            resistivity_ranges = {
                'Minéraux métalliques\n(Graphite, Cuivre, Or)': (0.001, 1, 'Spectral', 'Très conducteurs - Cibles minières'),
                'Eaux de mer + Argiles marines': (0.1, 10, 'YlOrRd', 'Zone conductrice - Salinité élevée'),
                'Argiles compactes + Eaux salées': (10, 50, 'RdYlBu', 'Formations imperméables saturées'),
                'Eaux douces + Limons + Schistes': (50, 200, 'YlGn', 'Aquifères argileux-sableux'),
                'Sables saturés + Graviers': (200, 1000, 'GnBu', 'Aquifères perméables productifs'),
                'Calcaires + Grès + Basaltes fracturés': (1000, 5000, 'PuBu', 'Formations carbonatées/volcaniques'),
                'Roches ignées + Granites': (5000, 100000, 'Purples', 'Socle cristallin - Très résistif'),
                'Quartzites + Minéraux isolants': (10000, 1000000, 'gray', 'Formations ultra-résistives')
            }
            
            cols_strat = st.columns(2)
            
            for idx, (name, (rho_min, rho_max, cmap, description)) in enumerate(resistivity_ranges.items()):
                with cols_strat[idx % 2]:
                    with st.expander(f"📍 **{name}** ({rho_min}-{rho_max} Ω·m)", expanded=False):
                        st.caption(f"*{description}*")
                        
                        # Filtrer les données dans cette plage
                        mask = (df['data'] >= rho_min) & (df['data'] <= rho_max)
                        df_filtered = df[mask]
                        
                        if len(df_filtered) > 3:
                            fig_strat, ax_strat = plt.subplots(figsize=(10, 6))
                            
                            # Convertir les données en float
                            X_strat = pd.to_numeric(df_filtered['survey_point'], errors='coerce').values
                            Z_strat = np.abs(pd.to_numeric(df_filtered['depth'], errors='coerce').values)
                            Rho_strat = pd.to_numeric(df_filtered['data'], errors='coerce').values
                            
                            # Filtrer NaN
                            mask_valid = ~(np.isnan(X_strat) | np.isnan(Z_strat) | np.isnan(Rho_strat))
                            X_strat = X_strat[mask_valid]
                            Z_strat = Z_strat[mask_valid]
                            Rho_strat = Rho_strat[mask_valid]
                            
                            if len(X_strat) > 3:
                                # Interpolation
                                from scipy.interpolate import griddata
                                xi_strat = np.linspace(X_strat.min(), X_strat.max(), 120)
                                zi_strat = np.linspace(Z_strat.min(), Z_strat.max(), 80)
                                Xi_strat, Zi_strat = np.meshgrid(xi_strat, zi_strat)
                                Rhoi_strat = griddata((X_strat, Z_strat), Rho_strat, 
                                                     (Xi_strat, Zi_strat), method='cubic')
                                
                                # Affichage avec échelle log si plage large
                                if rho_max / rho_min > 10:
                                    pcm_strat = ax_strat.pcolormesh(Xi_strat, Zi_strat, Rhoi_strat, 
                                                                   cmap=cmap, shading='auto',
                                                                   norm=LogNorm(vmin=rho_min, vmax=rho_max))
                                else:
                                    pcm_strat = ax_strat.pcolormesh(Xi_strat, Zi_strat, Rhoi_strat, 
                                                                   cmap=cmap, shading='auto',
                                                                   vmin=rho_min, vmax=rho_max)
                                
                                # Points de mesure
                                ax_strat.scatter(X_strat, Z_strat, c='black', s=30, 
                                               edgecolors='white', linewidths=1, marker='o', 
                                               alpha=0.6, zorder=10)
                                
                                ax_strat.invert_yaxis()
                                ax_strat.set_xlabel('Distance (m, précision: mm)', fontsize=11, fontweight='bold')
                                ax_strat.set_ylabel('Profondeur (m, précision: mm)', fontsize=11, fontweight='bold')
                                ax_strat.set_title(f'{name}\n{len(df_filtered)} mesures - Résistivité : {rho_min}-{rho_max} Ω·m',
                                                 fontsize=11, fontweight='bold', pad=15)
                                ax_strat.grid(True, alpha=0.3, linestyle='--')
                                
                                # Définir les ticks avec TOUTES les valeurs mesurées
                                unique_depths = np.unique(Z_strat)
                                unique_distances = np.unique(X_strat)
                                
                                # Limiter à 20 ticks max pour lisibilité
                                if len(unique_depths) > 20:
                                    step_depth = len(unique_depths) // 20
                                    ax_strat.set_yticks(unique_depths[::step_depth])
                                else:
                                    ax_strat.set_yticks(unique_depths)
                                
                                if len(unique_distances) > 20:
                                    step_dist = len(unique_distances) // 20
                                    ax_strat.set_xticks(unique_distances[::step_dist])
                                else:
                                    ax_strat.set_xticks(unique_distances)
                                
                                # Format des ticks avec 3 décimales
                                ax_strat.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                                ax_strat.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                                
                                cbar_strat = plt.colorbar(pcm_strat, ax=ax_strat, pad=0.02)
                                cbar_strat.set_label('Résistivité (Ω·m)', fontsize=10, fontweight='bold')
                                
                                plt.tight_layout()
                                st.pyplot(fig_strat)
                                plt.close()
                            else:
                                st.info(f"✓ {len(df_filtered)} mesure(s) détectée(s) mais insuffisantes pour interpolation")
                        else:
                            st.info(f"ℹ️ Aucune ou trop peu de mesures ({len(df_filtered)}) dans cette plage de résistivité")
            
            st.markdown("---")
            
            # Graphique synthétique de distribution
            st.subheader("📊 Distribution des Matériaux par Profondeur")
            
            fig_dist, (ax_hist, ax_depth) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Histogramme des résistivités (échelle log)
            rho_data = pd.to_numeric(df['data'], errors='coerce').dropna()
            ax_hist.hist(rho_data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            ax_hist.set_xscale('log')
            ax_hist.set_xlabel('Résistivité (Ω·m) - Échelle log', fontsize=11, fontweight='bold')
            ax_hist.set_ylabel('Nombre de mesures', fontsize=11, fontweight='bold')
            ax_hist.set_title('Distribution des Résistivités Mesurées', fontsize=12, fontweight='bold')
            ax_hist.grid(True, alpha=0.3, axis='y')
            
            # Zones colorées pour les matériaux
            ax_hist.axvspan(0.001, 1, alpha=0.2, color='gold', label='Minéraux métalliques')
            ax_hist.axvspan(1, 10, alpha=0.2, color='red', label='Eaux salées + Argiles')
            ax_hist.axvspan(10, 100, alpha=0.2, color='yellow', label='Eaux douces + Sols')
            ax_hist.axvspan(100, 1000, alpha=0.2, color='green', label='Sables + Graviers')
            ax_hist.axvspan(1000, 10000, alpha=0.2, color='blue', label='Roches sédimentaires')
            ax_hist.axvspan(10000, 1000000, alpha=0.2, color='purple', label='Roches ignées')
            ax_hist.legend(loc='upper right', fontsize=8)
            
            # Profil résistivité vs profondeur
            depth_data = np.abs(pd.to_numeric(df['depth'], errors='coerce').dropna())
            rho_for_depth = pd.to_numeric(df.loc[depth_data.index, 'data'], errors='coerce')
            
            scatter = ax_depth.scatter(rho_for_depth, depth_data, c=rho_for_depth, 
                                      cmap='viridis', s=50, alpha=0.6, 
                                      edgecolors='black', linewidths=0.5,
                                      norm=LogNorm(vmin=max(0.1, rho_for_depth.min()), 
                                                  vmax=rho_for_depth.max()))
            ax_depth.set_xscale('log')
            ax_depth.invert_yaxis()
            ax_depth.set_xlabel('Résistivité (Ω·m) - Échelle log', fontsize=11, fontweight='bold')
            ax_depth.set_ylabel('Profondeur (m, précision: mm)', fontsize=11, fontweight='bold')
            ax_depth.set_title('Résistivité en fonction de la Profondeur (Précision Millimétrique)', 
                              fontsize=12, fontweight='bold')
            ax_depth.grid(True, alpha=0.3)
            
            # Définir ticks avec toutes les profondeurs mesurées
            unique_depths_all = np.unique(depth_data)
            if len(unique_depths_all) > 20:
                ax_depth.set_yticks(unique_depths_all[::len(unique_depths_all)//20])
            else:
                ax_depth.set_yticks(unique_depths_all)
            
            # Format Y axis avec 3 décimales
            ax_depth.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
            
            cbar_dist = plt.colorbar(scatter, ax=ax_depth)
            cbar_dist.set_label('Résistivité (Ω·m)', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig_dist)
            plt.close()
            
            st.markdown("---")
            
            # ========== VISUALISATION 3D DES MINÉRAUX PAR COUCHES ==========
            st.subheader("🌐 Coupe Stratigraphique 3D")
            st.markdown("""
            Vue tridimensionnelle montrant les **couches géologiques** basées sur la résistivité.
            - **Axe X (horizontal)** : Distance le long du profil ERT (m)
            - **Axe Y (horizontal)** : Log₁₀ de la Résistivité - forme des **couches**
            - **Axe Z (VERTICAL)** : ⬇️ Profondeur (m) - descend vers le bas
            
            Les **couleurs** représentent les **8 catégories géologiques** (même résistivité = même couche).  
            **Rotation interactive** : Clic + glisser pour explorer les couches en 3D.
            """)
            
            # Préparer les données 3D
            # X = Distance horizontale du profil, Y = Offset transversal (jitter pour visualisation), Z = Profondeur
            X_3d_dist = pd.to_numeric(df['survey_point'], errors='coerce').values
            Z_3d_depth = -np.abs(pd.to_numeric(df['depth'], errors='coerce').values)  # Négatif pour descendre
            Y_3d_rho = pd.to_numeric(df['data'], errors='coerce').values
            
            # Filtrer les NaN
            mask_3d = ~(np.isnan(X_3d_dist) | np.isnan(Z_3d_depth) | np.isnan(Y_3d_rho))
            X_3d_dist = X_3d_dist[mask_3d]
            Z_3d_depth = Z_3d_depth[mask_3d]
            Y_3d_rho = Y_3d_rho[mask_3d]
            
            if len(X_3d_dist) > 0:
                # Créer la figure 3D avec plotly pour interactivité
                import plotly.graph_objects as go
                
                # Pour une vraie stratigraphie, utiliser directement la résistivité comme Y
                # Cela crée des "couches" géologiques visibles dans le profil
                Y_3d_rho_log = np.log10(Y_3d_rho + 0.001)  # Échelle logarithmique simple
                
                # Définir les catégories avec couleurs
                def get_material_category(resistivity):
                    if resistivity < 1:
                        return '💎 Minéraux métalliques', '#FFD700'
                    elif resistivity < 10:
                        return '💧 Eaux salées + Argiles', '#FF4500'
                    elif resistivity < 50:
                        return '🧱 Argiles compactes', '#8B4513'
                    elif resistivity < 200:
                        return '💧 Eaux douces + Sols', '#90EE90'
                    elif resistivity < 1000:
                        return '🏖️ Sables + Graviers', '#F4A460'
                    elif resistivity < 5000:
                        return '🪨 Roches sédimentaires', '#87CEEB'
                    elif resistivity < 100000:
                        return '🌋 Roches ignées (Granite)', '#FFB6C1'
                    else:
                        return '💎 Quartzite', '#E0E0E0'
                
                # Classifier chaque point
                categories_3d = [get_material_category(rho) for rho in Y_3d_rho]
                materials = [cat[0] for cat in categories_3d]
                colors = [cat[1] for cat in categories_3d]
                
                # Créer le scatter 3D
                fig_3d = go.Figure()
                
                # Grouper par catégorie pour la légende
                unique_materials = list(set(materials))
                for material in unique_materials:
                    mask_mat = np.array([m == material for m in materials])
                    fig_3d.add_trace(go.Scatter3d(
                        x=X_3d_dist[mask_mat],
                        y=Y_3d_rho_log[mask_mat],  # Log(résistivité) - couches horizontales
                        z=Z_3d_depth[mask_mat],    # Profondeur verticale (négatif = vers le bas)
                        mode='markers',
                        name=material,
                        marker=dict(
                            size=6,
                            color=colors[materials.index(material)],
                            opacity=0.8,
                            line=dict(color='white', width=0.5)
                        ),
                        text=[f'Distance: {x:.3f} m<br>Profondeur: {abs(z):.3f} m (≈{abs(z)*1000:.0f} mm)<br>Résistivité: {rho:.2f} Ω·m<br>Matériau: {mat}' 
                              for x, z, rho, mat in zip(X_3d_dist[mask_mat], Z_3d_depth[mask_mat], 
                                                        Y_3d_rho[mask_mat], np.array(materials)[mask_mat])],
                        hovertemplate='%{text}<extra></extra>'
                    ))
                
                fig_3d.update_layout(
                    title=dict(
                        text='Coupe Stratigraphique 3D<br><sub>Profondeur verticale | Couches par résistivité</sub>',
                        font=dict(size=16, family='Arial Black')
                    ),
                    scene=dict(
                        xaxis=dict(title='Distance (m, précision: mm)', backgroundcolor='lightgray'),
                        yaxis=dict(title='Log₁₀(Résistivité)', backgroundcolor='lightgray'),
                        zaxis=dict(title='⬇️ Profondeur (m, précision: mm)', backgroundcolor='lightgray'),
                        camera=dict(
                            eye=dict(x=1.5, y=-1.5, z=1.2)  # Vue latérale pour voir les couches
                        ),
                        aspectmode='manual',
                        aspectratio=dict(x=3, y=1.5, z=2)  # Profil étiré, couches visibles
                    ),
                    width=900,
                    height=700,
                    showlegend=True,
                    legend=dict(
                        title='Catégories',
                        yanchor='top',
                        y=0.99,
                        xanchor='left',
                        x=0.01,
                        bgcolor='rgba(255,255,255,0.8)'
                    )
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Sauvegarder la figure 3D pour le PDF (version matplotlib)
                from mpl_toolkits.mplot3d import Axes3D
                fig_3d_pdf = plt.figure(figsize=(12, 8), dpi=150)
                ax_3d_pdf = fig_3d_pdf.add_subplot(111, projection='3d')
                
                # Plot par catégorie
                for material in unique_materials:
                    mask_mat = np.array([m == material for m in materials])
                    color_hex = colors[materials.index(material)]
                    ax_3d_pdf.scatter(X_3d_dist[mask_mat], 
                                     Y_3d_rho_log[mask_mat],  # Log simple sans multiplication
                                     Z_3d_depth[mask_mat],
                                     c=color_hex, s=50, alpha=0.7, 
                                     edgecolors='white', linewidths=0.5,
                                     label=material)
                
                ax_3d_pdf.set_xlabel('Distance (m, précision: mm)', fontsize=11, fontweight='bold')
                ax_3d_pdf.set_ylabel('Log₁₀(Résistivité)', fontsize=11, fontweight='bold')
                ax_3d_pdf.set_zlabel('⬇️ Profondeur (m, précision: mm)', fontsize=11, fontweight='bold')
                ax_3d_pdf.set_title('Coupe Stratigraphique 3D\nCouches Géologiques par Résistivité (Précision Millimétrique)',
                                   fontsize=13, fontweight='bold', pad=20)
                ax_3d_pdf.legend(loc='upper left', fontsize=8, framealpha=0.9)
                ax_3d_pdf.grid(True, alpha=0.3)
                
                # Ajuster le ratio pour voir les couches horizontales
                ax_3d_pdf.set_box_aspect([3, 1.5, 2])  # Profil étiré, couches visibles
                plt.tight_layout()
                
                st.success(f"""
                ✅ **Visualisation 3D générée avec succès**
                - {len(X_3d_dist)} points cartographiés
                - {len(unique_materials)} catégories géologiques distinctes
                - Modèle interactif avec rotation 360°
                """)
            else:
                st.warning("⚠️ Données insuffisantes pour la visualisation 3D")
                fig_3d_pdf = None
            
            st.markdown("---")
            
            # ========== EXPORT PDF DU RAPPORT STRATIGRAPHIQUE ==========
            st.subheader("📄 Génération du Rapport PDF Complet")
            st.markdown("""
            Téléchargez un **rapport PDF professionnel** incluant :
            - 📊 Tableau de classification complète (30+ matériaux)
            - 📈 Graphiques de distribution (histogramme + profil)
            - 🌐 Visualisation 3D des couches géologiques
            - 📋 Statistiques détaillées et interprétation
            """)
            
            if st.button("🎯 Générer le Rapport PDF Stratigraphique", key="btn_pdf_strat"):
                with st.spinner("🔄 Génération du rapport PDF en cours..."):
                    # Créer un dictionnaire avec toutes les figures
                    figures_strat = {}
                    
                    # Figure 1: Distribution
                    figures_strat['distribution'] = fig_dist
                    
                    # Figure 2: 3D (si disponible)
                    if fig_3d_pdf is not None:
                        figures_strat['3d_view'] = fig_3d_pdf
                    
                    # Générer le PDF
                    pdf_bytes = create_stratigraphy_pdf_report(df, figures_strat)
                    
                    # Bouton de téléchargement
                    st.download_button(
                        label="⬇️ Télécharger le Rapport Stratigraphique (PDF)",
                        data=pdf_bytes,
                        file_name=f"Rapport_Stratigraphie_ERT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_pdf_strat"
                    )
                    
                    st.success("✅ Rapport PDF généré avec succès ! Cliquez sur le bouton ci-dessus pour télécharger.")
            
            st.markdown("---")
            
            st.success(f"""
            ✅ **Analyse complète effectuée**
            - {len(df)} mesures analysées
            - Profondeur max : {depth_data.max():.3f} m (≈{depth_data.max()*1000:.0f} mm)
            - Résistivité min/max : {rho_data.min():.2f} - {rho_data.max():.0f} Ω·m
            - Identification automatique des formations géologiques
            - Visualisation 3D interactive disponible
            - Export PDF professionnel prêt
            """)
        else:
            st.info("ℹ️ Le fichier uploadé ne contient pas de données valides.")
    else:
        st.warning("⚠️ Aucune donnée chargée. Veuillez d'abord uploader un fichier .dat dans l'onglet 'Analyse Fichiers .dat'")
        st.info("💡 Une fois les données chargées, vous pourrez visualiser la stratigraphie complète avec identification automatique des formations.")

# --- Sidebar ---
st.sidebar.image("logo_belikan.png", width="stretch")
st.sidebar.markdown("""
**SETRAF - Subaquifère ERT Analysis**  
💧 Outil d'analyse géophysique avancé  
Expert en hydrogéologie et tomographie électrique

**Version Optimisée – 08 Novembre 2025**  
✅ Calculateur Ts intelligent (Ravensgate Sonic)  
✅ Analyse .dat + détection anomalies (K-Means avec cache)  
✅ Tableau résistivité eau (descriptions détaillées)  
✅ Pseudo-sections 2D/3D basées sur vos données réelles  
✅ **NOUVEAU** : Stratigraphie complète (sols + eaux + roches + minéraux)  
✅ **NOUVEAU** : Visualisation 3D interactive des matériaux par couches  
✅ **NOUVEAU** : Précision millimétrique (3 décimales sur tous les axes)  
✅ Interprétation multi-matériaux : 8 catégories géologiques  
✅ Performance optimisée avec @st.cache_data  
✅ Interpolation cubique cachée pour fluidité  
✅ Ticks basés sur mesures réelles (0.1, 0.2, 0.3...)  
✅ **Export PDF** : Rapports complets avec tous les graphiques

**Exports disponibles** :  
📥 CSV - Données brutes  
📊 Excel - Tableaux formatés  
📄 PDF Standard - Rapport d'analyse DTW (150 DPI)  
📄 PDF Stratigraphique - Classification géologique complète (150 DPI)

**Visualisations avancées** :  
🎨 Coupes 2D par type de matériau (8 plages de résistivité)  
🌐 Modèle 3D interactif (rotation 360°, zoom)  
📊 Histogrammes et profils de distribution  
🗺️ Cartographie spatiale des formations géologiques

**Catégories géologiques identifiées** :  
💧 Eaux (mer, salée, douce, pure)  
🧱 Argiles & sols saturés  
🏖️ Sables & graviers  
🪨 Roches sédimentaires (calcaire, grès, schiste)  
🌋 Roches ignées & métamorphiques (granite, basalte)  
� Minéraux & minerais (graphite, cuivre, or, quartz)

**Plages de résistivité** :  
- 0.001-1 Ω·m : Minéraux métalliques  
- 0.1-10 Ω·m : Eaux salées + argiles marines  
- 10-100 Ω·m : Eaux douces + sols fins  
- 100-1000 Ω·m : Sables saturés + graviers  
- 1000-10000 Ω·m : Roches sédimentaires  
- >10000 Ω·m : Socle cristallin (granite, quartzite)  
""")

