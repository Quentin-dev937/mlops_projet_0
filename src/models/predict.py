import os
import skops.io as sio

# Chemin relatif vers le modèle entraîné
MODEL_PATH = os.path.join("models", "model.skops")

def predict_single_sample(input_data=None):
    """
    Charge le modèle et prédit sur un échantillon.
    input_data : liste de valeurs (ex: [5.1, 3.5, 1.4, 0.2])
    """
    
    # 1. Vérifier que le modèle existe
    if not os.path.exists(MODEL_PATH):
        # Astuce pour le test local : si le modèle n'existe pas, on lance l'entraînement d'abord
        print("Modèle non trouvé. Lancement de l'entraînement automatique...")
        from src.train import train
        train()
    
    # 2. Charger le modèle
    model = sio.load(MODEL_PATH, trusted=sio.get_untrusted_types(file=MODEL_PATH))
    
    # 3. Données par défaut si rien n'est fourni (pour le test)
    if input_data is None:
        # Remplacez ces valeurs par celles correspondant à vos colonnes réelles
        input_data = [[5.1, 3.5, 1.4, 0.2]] 
    
    # 4. Prédire
    prediction = model.predict(input_data)
    proba = model.predict_proba(input_data)
    
    print(f"🔮 Prédiction : Classe {prediction[0]}")
    return prediction[0]

if __name__ == "__main__":
    predict_single_sample()