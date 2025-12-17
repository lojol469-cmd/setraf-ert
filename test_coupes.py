import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
import numpy as np
import sys
import os

# Ajouter le répertoire au path
sys.path.insert(0, os.path.dirname(__file__))

# Test de génération de coupe simple
try:
    # Simuler des données de résistivité
    rho_data = np.random.rand(50, 50) * 100 + 10  # Résistivité entre 10-110 Ω·m
    
    # Créer une figure simple
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(rho_data, cmap='rainbow', aspect='auto')
    ax.set_title('Test Coupe Géologique')
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Profondeur (m)')
    plt.colorbar(im, ax=ax, label='Résistivité (Ω·m)')
    
    # Sauvegarder
    plt.savefig('/tmp/test_coupe.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Test de génération de coupe réussi")
    print("Figure sauvegardée dans /tmp/test_coupe.png")
    
except Exception as e:
    print(f"❌ Erreur génération coupe: {e}")
    import traceback
    traceback.print_exc()
