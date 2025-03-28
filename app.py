import streamlit as st
import pandas as pd
import joblib as jd

def main():
    st.title("Clasificación de Calidad del Aire")

    # Cargar modelos y scaler
    knn_model = jd.load("KNN (1).joblib")
    scaler = jd.load("scaler.joblib")


    # Pedir los datos al usuario con los mismos nombres que en el entrenamiento
    pm25 = st.number_input("PM25 (µg/m³)", value=10.0)
    pm10 = st.number_input("PM10 (µg/m³)", value=30.0)
    temperature = st.number_input("Temperature (°C)", value=25.0)
    no2 = st.number_input("NO2 (ppb)", value=5.0)
    o3 = st.number_input("O3 (ppb)", value=15.0)

    if st.button("Predecir"):
        # Crear DataFrame con los mismos nombres de columnas
        input_data = pd.DataFrame({
            "PM25 (µg/m³)": [pm25],
            "PM10 (µg/m³)": [pm10],
            "Temperature (°C)": [temperature],
            "NO2 (ppb)": [no2],
            "O3 (ppb)": [o3]
        })

        # Aplicar el mismo escalador que se usó en entrenamiento
        input_scaled = scaler.transform(input_data)

        # Predecir con el modelo KNN
        prediction = knn_model.predict(input_scaled)

        st.write(f"La predicción es: {prediction[0]}")

if __name__ == "__main__":
    main()
