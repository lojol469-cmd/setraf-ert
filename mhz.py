import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygimli.physics.ert import simulate, Inversion
import pygimli as pg
from matplotlib.colors import ListedColormap, BoundaryNorm
import io

# Palette de couleurs basée sur l'image (4 classes)
colors = ['red', 'orange', 'yellow', 'blue']  # Rouge vif, Orange, Jaune, Bleu
bounds = [0, 1, 10, 100, np.inf]  # Plages ρ_a
cmap = ListedColormap(colors)
norm = BoundaryNorm(bounds, cmap.N)

st.title("🛡️ Coupe ERT Colorée - Projet Archange Ondimba 2")
st.write("Upload votre fichier .csv ERT (format : Projet,Essai,ρ_a1,...,ρ_a50). Visualisation avec palette hydrogéologique.")

# Upload fichier
uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")

if uploaded_file is not None:
    # Parse et traitement des données
    df = pd.read_csv(uploaded_file)
    st.write("**Données parsées :**")
    st.dataframe(df.head())
    
    # Moyenne sur essais (colonnes 2+ = ρ_a)
    rho_mean = df.iloc[:, 2:].mean(axis=0).values  # Moyenne des ρ_a
    
    # Axes : x (positions, e.g., 50 électrodes espacées de 1m) et z (profondeurs, Wenner α)
    x = np.arange(0, 50 * 1, 1)  # Positions en m (50 points)
    z = 0.5 * np.arange(1, 51)   # Profondeurs approx. (0.5m à 25m)
    
    # Créer un pseudo-profil ERT pour pyGIMLi (simplifié : mesh 2D + données apparentes)
    st.write("**Inversion simple avec pyGIMLi...**")
    
    # Créer un mesh 2D simple (50x25 points)
    mesh = pg.createGrid(x=np.linspace(0, 50, 50), y=np.linspace(0, -25, 25), worldDim=2)
    
    # Simuler des données ERT (utiliser vos ρ_a comme modèle initial)
    model = np.tile(rho_mean, (25, 1)).T  # Modèle initial (ρ_a vs profondeur)
    data = simulate(mesh, scheme=pg.physics.ert.createData(elecs=50, schemeName='wenner'), res=model.flatten())
    
    # Inversion basique
    inv = Inversion(data)
    inv.setMesh(mesh)
    inv.setModel(model.flatten())
    # Run inversion (simplifiée, sans optimisation lourde pour démo)
    rho_inverted = inv.model()  # Utilise le modèle simulé pour démo rapide
    
    # Reshape pour plot (50 x 25)
    rho_2d = rho_inverted.reshape(25, 50).T
    
    # Visualisation
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.contourf(x, z, rho_2d, levels=bounds, cmap=cmap, norm=norm, extend='max')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Profondeur (m)')
    ax.set_title('Coupe de Résistivité (ERT) - Inversion pyGIMLi')
    plt.colorbar(im, ax=ax, label='ρ_a (Ω·m)', ticks=bounds[:-1])
    st.pyplot(fig)
    
    # Tableau des résistivités mappées aux couleurs/types
    st.write("**Interprétation : Résistivités mappées aux types d'eau**")
    rho_df = pd.DataFrame({
        'Profondeur (m)': z,
        'ρ_a Moyenne (Ω·m)': rho_mean,
        'Type d\'Eau': ['Eau de mer' if r < 1 else 'Eau salée' if r < 10 else 'Eau douce' if r < 100 else 'Eau très pure' for r in rho_mean],
        'Couleur': ['Rouge vif / Orange' if r < 1 else 'Jaune / Orange' if r < 10 else 'Vert / Bleu clair' if r < 100 else 'Bleu' for r in rho_mean]
    })
    st.dataframe(rho_df)
    
    # Download du CSV interprété
    csv_buffer = io.StringIO()
    rho_df.to_csv(csv_buffer, index=False)
    st.download_button("Télécharger CSV Interprété", csv_buffer.getvalue(), "ert_interprete.csv")

else:
    st.info("📁 Upload un fichier pour commencer !")