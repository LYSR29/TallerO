import streamlit as st
import pandas as pd
import os
import joblib
import numpy as np
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
            
            # Métricas calculando RMSE (Raíz de MSE)
            res = pd.DataFrame({
                "Modelo": ["Regresión Lineal", "KNN", "Random Forest"],
                "RMSE": [
                    mean_squared_error(y_test, p_lr, squared=False),
                    mean_squared_error(y_test, p_knn, squared=False),
                    mean_squared_error(y_test, p_rf, squared=False)
                ],
                "R2 Score": [r2_score(y_test, p_lr), r2_score(y_test, p_knn), r2_score(y_test, p_rf)]
            })

            # Guardar métricas para uso en el simulador
            st.session_state['metricas_globales'] = res
            st.table(res)
            
            # Identificación del mejor modelo
            mejor_r2 = res.loc[res["R2 Score"].idxmax()]
            mejor_rmse = res.loc[res["RMSE"].idxmin()]
            st.session_state['mejor_modelo_nombre'] = mejor_r2['Modelo']
            
            st.info(f"**Análisis:** El modelo '{mejor_r2['Modelo']}' es el líder en precisión (R2: {mejor_r2['R2 Score']:.4f}), mientras que '{mejor_rmse['Modelo']}' tiene el menor error promedio (RMSE: {mejor_rmse['RMSE']:.4f}).")
            
            st.bar_chart(res.set_index("Modelo")["R2 Score"])
        else:
            st.error("Primero debe completar la configuración en la barra lateral.")

with tab_sim:
    st.header("Simulador de Entrenamiento")
    
    if 'modelos' in st.session_state:
        lr, knn, rf, scaler = st.session_state['modelos']
        
        # Inicializar estado de interacción
        if 'sim_interactuado' not in st.session_state:
            st.session_state['sim_interactuado'] = False

        # --- SECCIÓN DE AVISOS ---
        if not st.session_state['sim_interactuado']:
            st.warning("ℹ️ Valores base (brutos). Mueva los sliders para generar predicciones personalizadas.")
        else:
            msg = "✅ Cuadro comparativo actualizado."
            if 'mejor_modelo_nombre' in st.session_state:
                msg += f" Sugerencia técnica: Priorice el resultado de **{st.session_state['mejor_modelo_nombre']}**."
            st.success(msg)
        
        st.write("Ajuste los parámetros para ver cómo responden los tres modelos simultáneamente.")
        
        # Controles de usuario
        col1, col2, col3 = st.columns(3)
        with col1:
            fc = st.slider("Frecuencia Cardíaca (bpm)", 60, 200, 140)
            pot = st.slider("Potencia (Watts)", 0, 500, 200)
        with col2:
            cad = st.slider("Cadencia (rpm)", 40, 120, 85)
            vel = st.slider("Velocidad (km/h)", 5, 60, 30)
        with col3:
            temp = st.slider("Temperatura (°C)", 5, 45, 25)
            pend = st.slider("Pendiente (%)", -10, 15, 2)
            
        tiempo = st.number_input("Tiempo total de actividad (min)", 1, 300, 60)

        # Detectar interacción
        if fc != 140 or pot != 200 or cad != 85: 
            st.session_state['sim_interactuado'] = True

        # Preparación de datos (Input Manual)
        datos_entrada = pd.DataFrame([[fc, pot, cad, tiempo, temp, pend, vel]], 
                                    columns=['frecuencia_cardiaca', 'potencia', 'cadencia', 'tiempo', 'temperatura', 'pendiente', 'velocidad'])
        
        # Predicción (Diferenciando modelos estandarizados de árboles)
        datos_escalados = scaler.transform(datos_entrada)
        p_lr = lr.predict(datos_escalados)[0]
        p_knn = knn.predict(datos_escalados)[0]
        p_rf = rf.predict(datos_entrada)[0]

        st.markdown("---")
        st.subheader("Cuadro Comparativo: Predicción vs Calidad del Modelo")

        # Construcción del Cuadro Comparativo Final
        if 'metricas_globales' in st.session_state:
            m = st.session_state['metricas_globales']
            cuadro_final = pd.DataFrame({
                "Modelo": ["Regresión Lineal", "KNN", "Random Forest"],
                "Predicción (%)": [p_lr, p_knn, p_rf],
                "RMSE (Error)": m["RMSE"].values,
                "R2 (Precisión)": m["R2 Score"].values
            })
            
            # Mostrar tabla con formato
            st.table(cuadro_final.style.format({
                "Predicción (%)": "{:.2f}",
                "RMSE (Error)": "{:.4f}",
                "R2 (Precisión)": "{:.4f}"
            }))
            st.caption("Nota: El RMSE indica el margen de error promedio del modelo sobre datos desconocidos.")
        else:
            # Si no han evaluado, mostrar solo predicciones básicas
            m1, m2, m3 = st.columns(3)
            m1.metric("Regresión Lineal", f"{p_lr:.2f}%")
            m2.metric("KNN", f"{p_knn:.2f}%")
            m3.metric("Random Forest", f"{p_rf:.2f}%")
            st.info("💡 Ejecute 'Evaluación de Modelos' en la otra pestaña para ver RMSE y R2 aquí.")

        # Gráfico comparativo
        res_sim = pd.DataFrame({"Fatiga (%)": [p_lr, p_knn, p_rf]}, index=["Regresión Lineal", "KNN", "Random Forest"])
        st.bar_chart(res_sim)

    else:
        st.info("⚠️ Los modelos aún no han sido entrenados. Use la barra lateral para comenzar.")