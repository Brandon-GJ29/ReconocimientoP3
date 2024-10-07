import joblib  # Para guardar el modelo entrenado 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# 1. Lista de imágenes de entrenamiento
imagenes_ruta = ['Entrenamiento1.jpg', 'Entrenamiento2.jpg', 'Entrenamiento3.jpg']

# Inicializar listas para almacenar características y etiquetas
X_train = []  # Características (media de píxeles en HSV)
y_train = []  # Etiquetas (en este caso, todos serán 1 para blanco)

# Procesar cada imagen
for img_ruta in imagenes_ruta:
    # Cargar la imagen
    imagen = cv2.imread(img_ruta)
    
    # 1. Aplicar un filtro gaussiano
    imagen_gaussiana = cv2.GaussianBlur(imagen, (5, 5), 0)

    # 2. Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(imagen_gaussiana, cv2.COLOR_BGR2HSV)

    # 3. Definir límites de color blanco para segmentación
    limite_inferior = np.array([0, 0, 200], dtype=np.uint8)
    limite_superior = np.array([180, 50, 255], dtype=np.uint8)

    # 4. Crear máscara para los píxeles que cumplen con el límite
    mascara_blancos = cv2.inRange(imagen_hsv, limite_inferior, limite_superior)

    # Extraer los píxeles correspondientes al color blanco
    pixeles_blancos = imagen_hsv[mascara_blancos != 0]

    # Calcular la media de los píxeles extraídos
    if len(pixeles_blancos) > 0:
        media_pixeles = np.mean(pixeles_blancos, axis=0)
        X_train.append(media_pixeles)
        y_train.append(1)  # Etiqueta para blanco
        
        # Calcular y mostrar la matriz de covarianza
        covarianza = np.cov(pixeles_blancos, rowvar=False)
        
        # 5. Mostrar imágenes y resultados intermedios
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mascara_blancos, cmap='gray')
        plt.title('Máscara de Blancos')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2RGB))
        plt.title('Imagen HSV')
        
        plt.suptitle(f'Imagen: {img_ruta}, Media: {media_pixeles}, Covarianza: {covarianza}')
        plt.show()

# Convertir listas a arrays de numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# 4. Entrenar el clasificador Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Clasificación de una nueva imagen
nueva_imagen = cv2.imread('Prueba2.jpg')  # Cargar la nueva imagen
nueva_imagen_gaussiana = cv2.GaussianBlur(nueva_imagen, (5, 5), 0)
nueva_imagen_hsv = cv2.cvtColor(nueva_imagen_gaussiana, cv2.COLOR_BGR2HSV)

# 6. Aplicar el mismo límite de color blanco
mascara_nueva = cv2.inRange(nueva_imagen_hsv, limite_inferior, limite_superior)
pixeles_nuevos = nueva_imagen_hsv[mascara_nueva != 0]

# 7. Clasificar los píxeles usando el modelo entrenado
X_nueva = np.mean(pixeles_nuevos, axis=0).reshape(1, -1)  # Obtener características
prediccion = modelo.predict(X_nueva)

# Crear nueva imagen para visualizar clases
clases_imagen = np.zeros(nueva_imagen.shape[:2], dtype=np.uint8)  # Imagen en blanco

# Asignar valores de gris: 0 para fondo, 128 para halo y 255 para objeto de interés (blanco)
clases_imagen[mascara_nueva != 0] = 255  # Blanco

# 5. Mostrar la imagen clasificada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(nueva_imagen, cv2.COLOR_BGR2RGB))
plt.title('Nueva Imagen Original')

plt.subplot(1, 2, 2)
plt.imshow(clases_imagen, cmap='gray')
plt.title('Imagen Clasificada')

plt.show()

# 5. Mostrar resultados intermedios de la nueva imagen
media_nueva = np.mean(pixeles_nuevos, axis=0)
covarianza_nueva = np.cov(pixeles_nuevos, rowvar=False)

print(f'Media de la nueva imagen: {media_nueva}')
print(f'Matriz de covarianza de la nueva imagen:\n {covarianza_nueva}')

# 8. Mostrar clasificación utilizando la función del clasificador de Bayes de scikit-learn
prediccion_sklearn = modelo.predict(X_nueva)  # Clasificación con scikit-learn
print(f'Predicción utilizando scikit-learn: {prediccion_sklearn}')
