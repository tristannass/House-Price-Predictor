import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# 1. Chargement des données
print(" Chargement des données...")
df = pd.read_csv("immobilier_clean.csv")

# 2. Préparation des Features (X) et de la Target (y)
# On transforme la variable catégorielle 'ville' en chiffres
df_encoded = pd.get_dummies(df, columns=['ville'], drop_first=True)

features = ['surface_m2', 'nb_pieces'] + [col for col in df_encoded.columns if 'ville_' in col]
X = df_encoded[features]
y = df_encoded['prix']

# 3. Split Train / Test (Vital en ML !)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Entraînement du modèle (Random Forest)
print(" Entraînement du modèle en cours...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Évaluation
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"--- RÉSULTATS ---")
print(f"Erreur moyenne (MAE) : {round(mae, 2)} €")
print(f"Précision (R²) : {round(r2 * 100, 2)} %")

# 6. Sauvegarde du modèle pour utilisation future
joblib.dump(model, "house_price_model.pkl")
print("💾 Modèle sauvegardé sous 'house_price_model.pkl'")
