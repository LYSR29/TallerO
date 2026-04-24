import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import joblib

def entrenar_bosque_aleatorio():
    df_train = pd.read_csv('train.csv')
    
    X_train = df_train[['frecuencia_cardiaca', 'potencia', 'cadencia', 'tiempo', 'temperatura', 'pendiente', 'velocidad']]
    y_train = df_train['fatiga']
    
    modelo = Pipeline([
        ("modelo", RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        ))
    ])
    
    modelo.fit(X_train, y_train)
    
    # Guardar modelo
    joblib.dump(modelo, 'modelo_rf.pkl')
    
    return modelo


def cargar_modelo_rf():
    return joblib.load('modelo_rf.pkl')