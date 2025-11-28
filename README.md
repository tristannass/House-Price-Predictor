# House Price Prediction Model

**Technologies :** Python, Scikit-Learn, Random Forest, Joblib.

## Description
Développement d'un modèle de Machine Learning supervisé capable d'estimer le prix d'un bien immobilier en fonction de ses caractéristiques (Surface, Localisation, Pièces).

## Pipeline ML
1. **Preprocessing** : Encodage des variables catégorielles (One-Hot Encoding).
2. **Modeling** : Utilisation d'un algorithme **Random Forest Regressor** pour sa robustesse.
3. **Evaluation** : Validation du modèle sur un jeu de test (Métriques : MAE, R²).
4. **Serialization** : Export du modèle (.pkl) pour mise en production via API.

## Performance
- Précision du modèle (R²) : ~95% (sur données synthétiques).
- Erreur moyenne : +/- 4500€.
