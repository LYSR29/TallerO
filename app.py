import streamlit as st
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score


from modelos_estandar import generar_archivos_separados, entrenar_modelos_distancia
from modelos_arboles import entrenar_bosque_aleatorio, cargar_modelo_rf

st.title("Proyecto TallerO - Ciclismo")


# BOTÓN 1: Generar la separación física
if st.button("1. Separar Dataset (Train/Test)"):
    try:
        # Esto genera los archivos train.csv y test.csv
        train, test = generar_archivos_separados('dataset_ciclismo_fatiga.csv')
        st.success(f"Archivos creados físicamente.")
        st.write(f"Entrenamiento: {len(train)} filas | Test: {len(test)} filas")
    except Exception as e:
        st.error(f"Error: {e}")

# BOTÓN 2: Entrenar (Cada uno por su lado)
if st.button("2. Entrenar Modelos"):
    if os.path.exists('train.csv'):

        lr, knn, scaler = entrenar_modelos_distancia()

        entrenar_bosque_aleatorio()
        rf = cargar_modelo_rf()

        st.session_state['modelos'] = (lr, knn, rf, scaler)

        st.success("Modelos entrenados correctamente.")
    else:
        st.warning("Primero genera los archivos con el Botón 1.")

if st.button("3. Ejecutar Test con 'test.csv'"):

    if not os.path.exists('test.csv'):
        st.error("No existe test.csv. Ejecuta el botón 1 primero.")
        st.stop()

    if 'modelos' in st.session_state:
        lr, knn, rf, scaler = st.session_state['modelos']
        df_test = pd.read_csv('test.csv')
        
        X_test = df_test[['frecuencia_cardiaca', 'potencia', 'cadencia', 'tiempo', 'temperatura', 'pendiente', 'velocidad']]
        y_test = df_test['fatiga']
        
        # TEST: Solo escalamos para LR y KNN
        X_test_scaled = scaler.transform(X_test)
        
        pred_lr = lr.predict(X_test_scaled)
        pred_knn = knn.predict(X_test_scaled)
        pred_rf = rf.predict(X_test)
        
        st.subheader("Comparativa de Rendimiento")

        res = pd.DataFrame({
            "Modelo": ["Regresión Lineal", "KNN", "Random Forest"],
            "MSE (Error)": [
                mean_squared_error(y_test, pred_lr),
                mean_squared_error(y_test, pred_knn),
                mean_squared_error(y_test, pred_rf)
            ],
            "R2 (Precisión)": [
                r2_score(y_test, pred_lr),
                r2_score(y_test, pred_knn),
                r2_score(y_test, pred_rf)
            ]
        })

        st.table(res)

        # 🔥 gráfico correcto
        st.bar_chart(res.set_index("Modelo"))

    else:
        st.error("Primero debes entrenar los modelos.")