#!/usr/bin/env python3
"""
Test script pour vérifier l'intégration pyGIMLi dans SETRAF
"""

import pandas as pd
import numpy as np
import pygimli as pg
from pygimli.physics.ert import ERTManager, simulate
import matplotlib.pyplot as plt

def test_pygimli_integration():
    """Test basique de l'inversion pyGIMLi"""
    print("🔬 Test de l'intégration pyGIMLi...")

    # Créer des données de test simples
    n_electrodes = 10
    spacing = 1.0

    # Mesh simple
    x = np.arange(0, n_electrodes * spacing, spacing)
    mesh = pg.createGrid(x=np.linspace(0, n_electrodes * spacing, n_electrodes),
                        y=np.linspace(0, -5, 5), worldDim=2)

    # Modèle simple (résistivité constante)
    model = np.ones(mesh.cellCount()) * 100  # 100 Ω·m

    # Créer un schéma de mesure simple
    scheme = pg.physics.ert.createData(elecs=n_electrodes, schemeName='wenner')

    # Simuler les données
    data = simulate(mesh, scheme=scheme, res=model)

    print(f"✅ Données simulées: {len(data)} mesures")

    # Inversion
    ert_manager = ERTManager()
    ert_manager.setMesh(mesh)
    ert_manager.setData(data)

    # Paramètres simples pour le test
    ert_manager.inv.setLambda(10)
    ert_manager.inv.setMaxIter(5)

    try:
        model_inv = ert_manager.invert()
        rho_result = ert_manager.inv.model()

        print("✅ Inversion réussie!"        print(".3f"        print(".3f"        print(f"   Modèle final: {len(rho_result)} cellules")

        return True

    except Exception as e:
        print(f"❌ Erreur lors de l'inversion: {e}")
        return False

if __name__ == "__main__":
    success = test_pygimli_integration()
    if success:
        print("\n🎉 Intégration pyGIMLi validée avec succès!")
    else:
        print("\n❌ Échec de l'intégration pyGIMLi")