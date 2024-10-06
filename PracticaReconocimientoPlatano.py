import joblib  # Para guardar el modelo entrenado 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Lista de imágenes de entrenamiento
imagenes_ruta = ['Entrenamiento1.jpg', 'Entrenamiento2.jpg', 'Entrenamiento3.jpg']

# Inicializar listas para almacenar características y etiquetas
X_train = []  # Características (media de píxeles en HSV)
y_train = []  # Etiquetas (en este caso, todos serán 1 para amarillo)

# Procesar cada imagen
for img_ruta in imagenes_ruta:
    # Cargar la imagen
    imagen = cv2.imread(img_ruta)
    
    # Aplicar un filtro gaussiano
    imagen_gaussiana = cv2.GaussianBlur(imagen, (5, 5), 0)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(imagen_gaussiana, cv2.COLOR_BGR2HSV)

    # Definir límites de color amarillo y marrón para segmentación
    limite_inferior_yellow = np.array([20, 100, 100], dtype=np.uint8)  # Amarillo
    limite_superior_yellow = np.array([30, 255, 255], dtype=np.uint8)  

    limite_inferior_brown = np.array([10, 100, 100], dtype=np.uint8)  # Marrón claro
    limite_superior_brown = np.array([20, 255, 255], dtype=np.uint8)  

    # Crear máscara para los píxeles amarillos y marrones
    mascara_amarillos = cv2.inRange(imagen_hsv, limite_inferior_yellow, limite_superior_yellow)
    mascara_marrones = cv2.inRange(imagen_hsv, limite_inferior_brown, limite_superior_brown)

    # Combinar máscaras
    mascara_total = cv2.bitwise_or(mascara_amarillos, mascara_marrones)

    # Extraer los píxeles correspondientes a los colores amarillos y marrones
    pixeles_segmentados = imagen_hsv[mascara_total != 0]

    # Calcular la media de los píxeles extraídos
    if len(pixeles_segmentados) > 0:
        media_pixeles = np.mean(pixeles_segmentados, axis=0)
        X_train.append(media_pixeles)
        y_train.append(1)  # Etiqueta para amarillo/marrón
        
        # Calcular y mostrar la matriz de covarianza
        covarianza = np.cov(pixeles_segmentados, rowvar=False)
        
        # Mostrar imágenes y resultados intermedios
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mascara_total, cmap='gray')
        plt.title('Máscara de Amarillos y Marrones')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2RGB))
        plt.title('Imagen HSV')
        
        plt.suptitle(f'Imagen: {img_ruta}, Media: {media_pixeles}, Covarianza: {covarianza}')
        plt.show()
    else:
        print(f'No se encontraron píxeles amarillos o marrones en {img_ruta}.')

# Convertir listas a arrays de numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# Verificar las dimensiones de X_train y y_train
print(f'Número de características (X_train): {len(X_train)}')
print(f'Número de etiquetas (y_train): {len(y_train)}')

# Entrenar el clasificador Naive Bayes
if len(X_train) > 0 and len(X_train) == len(y_train):
    modelo = GaussianNB()
    modelo.fit(X_train, y_train)

    # Guardar el modelo entrenado 
    # joblib.dump(modelo, 'modeloEntrenadoPlatanos.pkl')  # Esto se omite, ya no se ocupa 
    # print("Modelo entrenado y guardado como 'modeloEntrenadoPlatanos.pkl'")

    # Clasificación de una nueva imagen
    nueva_imagen = cv2.imread('Prueba1.jpg')  # Cargar la nueva imagen
    nueva_imagen_gaussiana = cv2.GaussianBlur(nueva_imagen, (5, 5), 0)
    nueva_imagen_hsv = cv2.cvtColor(nueva_imagen_gaussiana, cv2.COLOR_BGR2HSV)

    # Aplicar los mismos límites de color
    mascara_nueva_yellow = cv2.inRange(nueva_imagen_hsv, limite_inferior_yellow, limite_superior_yellow)
    mascara_nueva_brown = cv2.inRange(nueva_imagen_hsv, limite_inferior_brown, limite_superior_brown)
    mascara_nueva_total = cv2.bitwise_or(mascara_nueva_yellow, mascara_nueva_brown)

    # Clasificar los píxeles usando el modelo entrenado
    pixeles_nuevos = nueva_imagen_hsv[mascara_nueva_total != 0]

    if len(pixeles_nuevos) > 0:
        X_nueva = np.mean(pixeles_nuevos, axis=0).reshape(1, -1)  # Obtener características
        prediccion = modelo.predict(X_nueva)

        # Crear nueva imagen para visualizar clases
        clases_imagen = np.zeros(nueva_imagen.shape[:2], dtype=np.uint8)  # Imagen en blanco

        # Asignar valores de gris: 0 para fondo, 255 para objeto de interés (amarillo/marrón)
        clases_imagen[mascara_nueva_total != 0] = 255  # Amarillo o Marrón

        # Mostrar la imagen clasificada
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(nueva_imagen, cv2.COLOR_BGR2RGB))
        plt.title('Nueva Imagen Original')

        plt.subplot(1, 2, 2)
        plt.imshow(clases_imagen, cmap='gray')
        plt.title('Imagen Clasificada')

        plt.show()

        # Mostrar resultados intermedios de la nueva imagen
        media_nueva = np.mean(pixeles_nuevos, axis=0)
        covarianza_nueva = np.cov(pixeles_nuevos, rowvar=False)

        print(f'Media de la nueva imagen: {media_nueva}')
        print(f'Matriz de covarianza de la nueva imagen:\n {covarianza_nueva}')
    else:
        print('No se encontraron píxeles amarillos o marrones en la nueva imagen.')
else:
    print('No se pudo entrenar el modelo debido a la inconsistencia en el número de muestras.')
