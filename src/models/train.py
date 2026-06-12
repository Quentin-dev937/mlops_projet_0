import os
import skops.io as sio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import des modules internes
from src.data.load_data import load_data
from src.evaluation.evaluate import evaluate_model

MODEL_PATH = os.path.join("models", "model.skops")

def train():
    print("1. Chargement des données...")
    X, y = load_data()
    
    print("2. Split des données...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. Entraînement du modèle...")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("4. Évaluation...")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)
    
    print("5. Sauvegarde du modèle...")
    os.makedirs("models", exist_ok=True)
    sio.dump(model, MODEL_PATH)
    print(f"✅ Modèle sauvegardé dans {MODEL_PATH}")

if __name__ == "__main__":
    train()