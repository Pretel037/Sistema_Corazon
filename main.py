import json
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os


# Define la ruta base donde se encuentran los modelos
base_path = os.path.join(os.getcwd(), 'models')

# Carga de modelo en caché
@st.cache_resource
def load_model(model_path):
    if model_path.endswith('.tflite'):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    else:
        return tf.keras.models.load_model(model_path)


with st.spinner('Cargando modelos...'):
    models = {
        "DenseNet121": load_model(os.path.join(base_path, 'densetnet_121.tflite')),
        "Efficientnetb_30Lite": load_model(os.path.join(base_path, 'Efficientnetb_30Lite.tflite')),
        "Inseptionv3": load_model(os.path.join(base_path, 'Inception.tflite')),
        
    }

# Función para predecir usando el modelo seleccionado
def image_prediction(image, model):
    # Preprocesar imagen
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (265, 265))
    image = image.reshape(1, 265, 265, 3).astype(np.float32)

    # Predicción dependiendo del tipo de modelo
    if isinstance(model, tf.keras.Model):  # Para modelos Keras
        pred = model.predict(image)
    else:  # Para modelos TFLite
        input_details = model.get_input_details()
        output_details = model.get_output_details()
        model.set_tensor(input_details[0]['index'], image)
        model.invoke()
        pred = model.get_tensor(output_details[0]['index'])

    # Obtener clase y precisión
    pred_class = np.argmax(pred, axis=1)[0]
    accuracy = pred[0][pred_class]
    labels = ["Con alteracion", "Sin Alteracion"]
    return labels[pred_class], accuracy
st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")

st.image('Logo_SmartRegions.gif')

# Interfaz de Streamlit
st.title("Sistema de clasificacion de alteraciones de heridas de operacion corazon abierto")
with st.sidebar:
        st.image('foto.png')
        st.title("El diagnositico AI ")
        st.subheader("Clasificacion de imagenes operacion de corazon abierto usando Deep Learning y 3 arquitecturas de redes neuronales de tranfer learning ")

# Entrada de cámara
image = st.camera_input("Captura una imagen para analizar")

# Procesar la imagen y mostrar los resultados si se captura una imagen
if image:
    image_file = Image.open(image)

    # Seleccionar modelo
    selected_model_name = st.selectbox("Selecciona el modelo", list(models.keys()))
    selected_model = models[selected_model_name]

    # Predicción
    result, accuracy = image_prediction(image_file, selected_model)
    accuracy_text = f"{accuracy * 100:.2f}%"  # Convierte el valor de precisión en porcentaje

    # Mostrar imagen y tabla de información una al lado de la otra
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image_file, caption=f"Predicción: {result} (Precisión: {accuracy_text})", use_column_width=True)

    # Mensaje de predicción y precisión
    st.info(f"El modelo predice que esto es {result} con una precisión de {accuracy_text}")