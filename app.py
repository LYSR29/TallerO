import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# Importación de tus funciones locales
from modelos_estandar import generar_archivos_separados, entrenar_modelos_distancia
from modelos_arboles import entrenar_bosque_aleatorio, cargar_modelo_rf

# 1. Configuración técnica de la página
st.set_page_config(page_title="Análisis de Fatiga - Ciclismo", layout="wide")

# 2. Estilo Formal y Minimalista (CSS)
st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    .stButton>button {
        border-radius: 4px;
        border: 1px solid #e0e0e0;
        background-color: #ffffff;
        color: #262730;
        width: 100%;
    }
    .stButton>button:hover {
        border-color: #000000;
        color: #000000;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
        font-weight: 400;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. Barra Lateral: Procesos de Configuración
with st.sidebar:
    st.title("Configuración")
    st.markdown("---")
    
    st.subheader("Fase 1: Preparación")
    if st.button("Separar Dataset (Train/Test)"):
        try:
            train, test = generar_archivos_separados('dataset_ciclismo_fatiga.csv')
            st.success("Dataset procesado.")
        except Exception as e:
            st.error(f"Error: {e}")
            
    st.subheader("Fase 2: Entrenamiento")
    if st.button("Entrenar Algoritmos"):
        if os.path.exists('train.csv'):
            lr, knn, scaler = entrenar_modelos_distancia()
            entrenar_bosque_aleatorio()
            rf = cargar_modelo_rf()
            st.session_state['modelos'] = (lr, knn, rf, scaler)
            st.success("Modelos listos.")
        else:
            st.warning("Falta el archivo train.csv.")

    st.markdown("---")
    if st.button("Reiniciar Sistema"):
        for f in ['train.csv', 'test.csv', 'modelo_lr.pkl', 'modelo_knn.pkl', 'modelo_rf.pkl', 'scaler.pkl']:
            if os.path.exists(f): os.remove(f)
        st.session_state.clear()
        st.rerun()

# 4. Área Principal: Visualización y Test
st.title("Sistema de Predicción de Fatiga en Ciclismo")

tab_eval, tab_sim = st.tabs(["Evaluación de Modelos", "Prueba Individual"])

with tab_eval:
    st.write("Compare el rendimiento de los modelos entrenados sobre el conjunto de prueba.")
    
    if st.button("Ejecutar Evaluación"):
        if 'modelos' in st.session_state and os.path.exists('test.csv'):
            lr, knn, rf, scaler = st.session_state['modelos']
            df_test = pd.read_csv('test.csv')
            
            X_test = df_test[['frecuencia_cardiaca', 'potencia', 'cadencia', 'tiempo', 'temperatura', 'pendiente', 'velocidad']]
            y_test = df_test['fatiga']
            
            # Predicciones
            X_test_scaled = scaler.transform(X_test)
            p_lr = lr.predict(X_test_scaled)
            p_knn = knn.predict(X_test_scaled)
            p_rf = rf.predict(X_test)
            
            # Métricas
            res = pd.DataFrame({
                "Modelo": ["Regresión Lineal", "KNN", "Random Forest"],
                "MSE": [mean_squared_error(y_test, p_lr), mean_squared_error(y_test, p_knn), mean_squared_error(y_test, p_rf)],
                "R2 Score": [r2_score(y_test, p_lr), r2_score(y_test, p_knn), r2_score(y_test, p_rf)]
            })

            st.table(res)
            
            # Identificación automática del mejor modelo
            mejor = res.loc[res["R2 Score"].idxmax()]
            st.info(f"Análisis: El modelo '{mejor['Modelo']}' es el más preciso para este dataset (R2: {mejor['R2 Score']:.4f}).")
            
            st.bar_chart(res.set_index("Modelo")["R2 Score"])
        else:
            st.error("Primero debe completar la configuración en la barra lateral.")

with tab_sim:
    st.write("Sección disponible para pruebas con datos manuales.")
    # Aquí podrías agregar sliders para que el profesor pruebe valores