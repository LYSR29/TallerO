import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

def generar_archivos_separados(ruta_original):
    df = pd.read_csv(ruta_original)
    df = df.dropna().drop_duplicates()
    train, test = train_test_split(df, test_size=0.20, random_state=42)
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)
    return train, test

def entrenar_modelos_distancia():
    df_train = pd.read_csv('train.csv')
    X_train = df_train[['frecuencia_cardiaca', 'potencia', 'cadencia', 'tiempo', 'temperatura', 'pendiente', 'velocidad']]
    y_train = df_train['fatiga']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr = LinearRegression().fit(X_train_scaled, y_train)
    knn = KNeighborsRegressor(n_neighbors=5).fit(X_train_scaled, y_train)
    
    return lr, knn, scaler