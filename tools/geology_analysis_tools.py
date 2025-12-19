"""
OUTILS D'ANALYSE STATISTIQUE GÉOLOGIQUE
Calculs avancés pour l'interprétation des données ERT
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class GeologyStatisticsTool:
    """Outil d'analyse statistique spécialisé en géologie"""

    def __init__(self):
        self.scaler = StandardScaler()

    def calculate_resistivity_statistics(self, resistivity_values: List[float]) -> Dict:
        """
        Calculs statistiques complets sur les données de résistivité

        Args:
            resistivity_values: Liste des valeurs de résistivité

        Returns:
            Dictionnaire avec toutes les statistiques
        """
        if not resistivity_values:
            return {"error": "Aucune donnée de résistivité"}

        data = np.array(resistivity_values)

        stats_dict = {
            "count": len(data),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "range": float(np.max(data) - np.min(data)),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data)),
            "variance": float(np.var(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
            "cv": float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0,  # Coefficient de variation
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data))
        }

        # Classification géologique basée sur la résistivité
        stats_dict["geological_classification"] = self._classify_resistivity_ranges(stats_dict)

        return stats_dict

    def _classify_resistivity_ranges(self, stats: Dict) -> str:
        """Classification géologique basée sur les plages de résistivité"""
        mean_res = stats.get('mean', 50)

        if mean_res < 10:
            return "Très conducteur - Possible: argiles saturées, eaux souterraines, minerais conducteurs"
        elif mean_res < 50:
            return "Conducteur - Possible: argiles, marnes humides, aquifères"
        elif mean_res < 200:
            return "Semi-conducteur - Possible: sables, limons, roches sédimentaires"
        elif mean_res < 1000:
            return "Résistant - Possible: grès, calcaires, roches cristallines fracturées"
        else:
            return "Très résistant - Possible: granites, basaltes, roches saines"

    def analyze_depth_distribution(self, depths: List[float], resistivities: List[float]) -> Dict:
        """
        Analyse de la distribution en fonction de la profondeur

        Args:
            depths: Liste des profondeurs
            resistivities: Liste des résistivités correspondantes

        Returns:
            Analyse par couches géologiques
        """
        if len(depths) != len(resistivities):
            return {"error": "Données de profondeur et résistivité incompatibles"}

        # Création d'un DataFrame pour analyse
        df = pd.DataFrame({
            'depth': depths,
            'resistivity': resistivities
        })

        # Analyse par quartiles de profondeur
        depth_quartiles = df['depth'].quantile([0.25, 0.5, 0.75]).values

        layers = {
            "surface": df[df['depth'] <= depth_quartiles[0]],
            "intermédiaire": df[(df['depth'] > depth_quartiles[0]) & (df['depth'] <= depth_quartiles[1])],
            "profonde": df[(df['depth'] > depth_quartiles[1]) & (df['depth'] <= depth_quartiles[2])],
            "très_profonde": df[df['depth'] > depth_quartiles[2]]
        }

        layer_analysis = {}
        for layer_name, layer_data in layers.items():
            if len(layer_data) > 0:
                layer_analysis[layer_name] = {
                    "count": len(layer_data),
                    "depth_range": f"{layer_data['depth'].min():.1f}-{layer_data['depth'].max():.1f}m",
                    "resistivity_mean": layer_data['resistivity'].mean(),
                    "resistivity_std": layer_data['resistivity'].std(),
                    "geological_interpretation": self._interpret_layer(layer_data['resistivity'].mean())
                }

        return layer_analysis

    def _interpret_layer(self, mean_resistivity: float) -> str:
        """Interprétation géologique d'une couche"""
        if mean_resistivity < 20:
            return "Couche conductrice - aquifère potentiel ou argiles saturées"
        elif mean_resistivity < 100:
            return "Couche semi-conductrice - marnes ou sédiments fins"
        elif mean_resistivity < 500:
            return "Couche résistante - sables ou roches sédimentaires"
        else:
            return "Couche très résistante - substratum rocheux ou formations dures"

    def detect_anomalies(self, resistivities: List[float], threshold: float = 2.0) -> Dict:
        """
        Détection d'anomalies dans les données de résistivité

        Args:
            resistivities: Liste des valeurs de résistivité
            threshold: Seuil pour la détection d'anomalies (en écarts-types)

        Returns:
            Analyse des anomalies détectées
        """
        data = np.array(resistivities)
        mean_val = np.mean(data)
        std_val = np.std(data)

        # Anomalies hautes et basses
        high_anomalies = data[data > mean_val + threshold * std_val]
        low_anomalies = data[data < mean_val - threshold * std_val]

        return {
            "total_points": len(data),
            "high_anomalies_count": len(high_anomalies),
            "low_anomalies_count": len(low_anomalies),
            "high_anomalies_values": high_anomalies.tolist()[:10],  # Top 10
            "low_anomalies_values": low_anomalies.tolist()[:10],   # Top 10
            "anomaly_percentage": (len(high_anomalies) + len(low_anomalies)) / len(data) * 100,
            "interpretation": self._interpret_anomalies(len(high_anomalies), len(low_anomalies))
        }

    def _interpret_anomalies(self, high_count: int, low_count: int) -> str:
        """Interprétation des anomalies détectées"""
        if high_count > low_count * 2:
            return "Prédominance d'anomalies hautes - possible pollution métallique ou formations résistantes"
        elif low_count > high_count * 2:
            return "Prédominance d'anomalies basses - possible aquifères ou argiles conductrices"
        elif high_count + low_count > 10:
            return "Nombreuses anomalies - hétérogénéité géologique importante"
        else:
            return "Faibles anomalies - formation géologique homogène"

    def cluster_analysis(self, resistivities: List[float], depths: List[float], n_clusters: int = 3) -> Dict:
        """
        Analyse par clustering pour identifier des groupes géologiques

        Args:
            resistivities: Valeurs de résistivité
            depths: Profondeurs correspondantes
            n_clusters: Nombre de clusters à identifier

        Returns:
            Analyse des clusters identifiés
        """
        if len(resistivities) != len(depths):
            return {"error": "Données incompatibles"}

        # Préparation des données
        X = np.column_stack([resistivities, depths])
        X_scaled = self.scaler.fit_transform(X)

        # Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # Analyse des clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            mask = clusters == i
            cluster_data = X[mask]

            cluster_analysis[f"cluster_{i+1}"] = {
                "size": len(cluster_data),
                "resistivity_center": float(kmeans.cluster_centers_[i][0]),
                "depth_center": float(kmeans.cluster_centers_[i][1]),
                "resistivity_range": f"{cluster_data[:, 0].min():.1f}-{cluster_data[:, 0].max():.1f}",
                "depth_range": f"{cluster_data[:, 1].min():.1f}-{cluster_data[:, 1].max():.1f}",
                "geological_interpretation": self._interpret_cluster(
                    float(kmeans.cluster_centers_[i][0]),
                    float(kmeans.cluster_centers_[i][1])
                )
            }

        return cluster_analysis

    def _interpret_cluster(self, resistivity: float, depth: float) -> str:
        """Interprétation géologique d'un cluster"""
        if resistivity < 30 and depth < 50:
            return "Aquifère superficiel - zone de recharge probable"
        elif resistivity > 200 and depth > 100:
            return "Formation rocheuse profonde - substratum géologique"
        elif 50 < resistivity < 150 and depth < 75:
            return "Couche sédimentaire intermédiaire - aquifère potentiel"
        else:
            return "Formation géologique mixte - nécessite investigation complémentaire"


# Instance globale
geology_stats_tool = GeologyStatisticsTool()