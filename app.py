import streamlit as st
import pandas as pd
import joblib as jd

def main():
    st.title("Clasificación de Calidad del Aire")

    # Cargar modelos
    knn_model = jd.load("KNN.joblib")
    svm_model = jd.load("SVM.joblib")
    rf_model = jd.load("Random Forest.joblib")

    # (Opcional) Cargar el escalador si lo usaste en el entrenamiento
    # scaler = joblib.load("scaler.joblib")

    # Seleccionar el modelo
    model_choice = st.selectbox("Seleccione el modelo", ("KNN", "SVM", "Random Forest"))

    # Pedir los datos al usuario con los mismos nombres que en el entrenamiento
    pm1 = st.number_input("PM1 (µg/m³)", value=10.0)
    pm10 = st.number_input("PM10 (µg/m³)", value=30.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    no2 = st.number_input("NO2 (ppb)", value=5.0)
    co = st.number_input("CO (ppb)", value=1.0)
    o3 = st.number_input("O3 (ppb)", value=15.0)

    if st.button("Predecir"):
        # Crear DataFrame con los mismos nombres de columnas
        input_data = pd.DataFrame({
            "PM1 (µg/m³)": [pm1],
            "PM10 (µg/m³)": [pm10],
            "Temperature (°C)": [temperature],
            "NO2 (ppb)": [no2],
            "CO (ppb)": [co],
            "O3 (ppb)": [o3]
        })

        # (Opcional) Aplicar el mismo escalador que se usó en entrenamiento
        # input_scaled = scaler.transform(input_data)

        # Seleccionar el modelo y predecir
        if model_choice == "KNN":
            prediction = knn_model.predict(input_data)
        elif model_choice == "SVM":
            prediction = svm_model.predict(input_data)
        else:  # Random Forest
            prediction = rf_model.predict(input_data)

        st.write(f"La predicción es: {prediction[0]}")

if __name__ == "__main__":
    main()
