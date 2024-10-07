import joblib  # Para guardar el modelo entrenado 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Lista de imágenes de entrenamiento
imagenes_ruta = ['Entrenamiento1.jpg', 'Entrenamiento2.jpg', 'Entrenamiento3.jpg']

# Inicializar listas para almacenar características y etiquetas
X_train = []  # Características (media de píxeles en HSV)
y_train = []  # Etiquetas
clases = ['platanos', 'huevos', 'chiles']  # Nombres de las clases

# 1. Preprocesamiento de imágenes (Filtro gaussiano)
for img_ruta in imagenes_ruta:
    # Cargar la imagen
    imagen = cv2.imread(img_ruta)
    
    # 1. Filtro Gaussiano para suavizar la imagen
    imagen_suavizada = cv2.GaussianBlur(imagen, (5, 5), 0)

    # 2. Convertir imagen de BGR a HSV y seleccionar regiones a clasificar
    imagen_hsv = cv2.cvtColor(imagen_suavizada, cv2.COLOR_BGR2HSV)

    # 3. Definir los límites de las máscaras para las zonas de interés
    limite_inferior_fuerte = np.array([40, 100, 100], dtype=np.uint8)  
    limite_superior_fuerte = np.array([80, 255, 255], dtype=np.uint8)  

    limite_inferior_claro = np.array([30, 40, 40], dtype=np.uint8)
    limite_superior_claro = np.array([90, 255, 255], dtype=np.uint8) 

    limite_inferior_blanco = np.array([0, 0, 200], dtype=np.uint8)
    limite_superior_blanco = np.array([180, 25, 255], dtype=np.uint8)

    limite_inferior_yellow = np.array([20, 100, 100], dtype=np.uint8)  
    limite_superior_yellow = np.array([30, 255, 255], dtype=np.uint8)

    limite_inferior_brown = np.array([10, 100, 100], dtype=np.uint8)  
    limite_superior_brown = np.array([20, 255, 255], dtype=np.uint8)  

    # 3. Crear máscaras para las zonas de interés
    mascara_verde_fuerte = cv2.inRange(imagen_hsv, limite_inferior_fuerte, limite_superior_fuerte)
    mascara_verde_claro = cv2.inRange(imagen_hsv, limite_inferior_claro, limite_superior_claro)
    mascara_blanco = cv2.inRange(imagen_hsv, limite_inferior_blanco, limite_superior_blanco)
    mascara_yellow = cv2.inRange(imagen_hsv, limite_inferior_yellow, limite_superior_yellow)
    mascara_brown = cv2.inRange(imagen_hsv, limite_inferior_brown, limite_superior_brown)

    # Combinar todas las máscaras
    mascara_final = cv2.bitwise_or(mascara_verde_fuerte, mascara_verde_claro)
    mascara_final = cv2.bitwise_or(mascara_final, mascara_blanco)
    mascara_final = cv2.bitwise_or(mascara_final, mascara_yellow)
    mascara_final = cv2.bitwise_or(mascara_final, mascara_brown)

    # Extraer los píxeles correspondientes a las máscaras
    pixeles = imagen_hsv[mascara_final != 0]

    # Asignar una etiqueta a la clase en función de la máscara (por simplicidad usaremos una sola clase por imagen)
    if len(pixeles) > 0:
        media_pixeles = np.mean(pixeles, axis=0)
        X_train.append(media_pixeles)
        
        # En este caso, se asigna una etiqueta diferente por imagen (1 para la primera clase, 2 para la segunda, etc.)
        y_train.append(clases[imagenes_ruta.index(img_ruta)])  # Etiqueta para los colores de interés
        
        # Calcular y mostrar la matriz de covarianza
        covarianza = np.cov(pixeles, rowvar=False)
        
        # Imprimir los valores de la media y la covarianza
        print(f'Clase: {clases[imagenes_ruta.index(img_ruta)]}, Media: {media_pixeles}, Covarianza:\n {covarianza}')
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mascara_final, cmap='gray')
        plt.title('Máscara Combinada')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2RGB))
        plt.title('Imagen HSV')
        
        plt.suptitle(f'Imagen: {img_ruta}, Media: {media_pixeles}, Covarianza: {covarianza}')
        plt.show()

# Convertir listas a arrays de numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# 4. Implementar el clasificador Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Guardar el modelo entrenado (opcional)
# joblib.dump(modelo, 'modeloEntrenadoVerdeFuerte.pkl')

# 6. Clasificación de una nueva imagen usando probabilidad Gaussiana
nuevaImagen= 'Prueba1.jpg'
nueva_imagen = cv2.imread(nuevaImagen)  # Cargar nueva imagen
nueva_imagen_suavizada = cv2.GaussianBlur(nueva_imagen, (5, 5), 0)
nueva_imagen_hsv = cv2.cvtColor(nueva_imagen_suavizada, cv2.COLOR_BGR2HSV)

# Aplicar el mismo límite de color
mascara_nueva_fuerte = cv2.inRange(nueva_imagen_hsv, limite_inferior_fuerte, limite_superior_fuerte)
mascara_nueva_claro = cv2.inRange(nueva_imagen_hsv, limite_inferior_claro, limite_superior_claro)
mascara_nueva_blanco = cv2.inRange(nueva_imagen_hsv, limite_inferior_blanco, limite_superior_blanco)
mascara_nueva_yellow = cv2.inRange(nueva_imagen_hsv, limite_inferior_yellow, limite_superior_yellow)
mascara_nueva_brown = cv2.inRange(nueva_imagen_hsv, limite_inferior_brown, limite_superior_brown)

# Combinar todas las máscaras
mascara_nueva = cv2.bitwise_or(mascara_nueva_fuerte, mascara_nueva_claro)
mascara_nueva = cv2.bitwise_or(mascara_nueva, mascara_nueva_blanco)
mascara_nueva = cv2.bitwise_or(mascara_nueva, mascara_nueva_yellow)
mascara_nueva = cv2.bitwise_or(mascara_nueva, mascara_nueva_brown)

pixeles_nuevos = nueva_imagen_hsv[mascara_nueva != 0]

# 6. Clasificar los píxeles con el clasificador
X_nueva = np.mean(pixeles_nuevos, axis=0).reshape(1, -1)  # Obtener características
prediccion = modelo.predict(X_nueva)

# 7. Crear nueva imagen con las clases resultantes (0 para fondo, 255 para objeto de interés)
clases_imagen = np.zeros(nueva_imagen.shape[:2], dtype=np.uint8)
clases_imagen[mascara_nueva != 0] = 255  

# Mostrar la imagen clasificada
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(nueva_imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen'+nuevaImagen)

plt.subplot(1, 2, 2)
plt.imshow(clases_imagen, cmap='gray')
plt.title('Imagen Clasificada')

plt.show()

# 8. Usar la función de clasificador de Bayes de scikit-learn
prediccion_sklearn = modelo.predict(X_nueva)  # Clasificación con scikit-learn

# Comparar con los resultados anteriores
print(f'Predicción utilizando scikit-learn: {prediccion_sklearn}')
print(f'Predicción obtenida con cálculo manual: {prediccion}')

# Mostrar la imagen clasificada por scikit-learn
clases_sklearn = np.zeros(nueva_imagen.shape[:2], dtype=np.uint8)
clases_sklearn[mascara_nueva != 0] = 255  

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(nueva_imagen, cv2.COLOR_BGR2RGB))
plt.title('Imagen'+nuevaImagen)

plt.subplot(1, 2, 2)
plt.imshow(clases_sklearn, cmap='gray')
plt.title('Clasificación con scikit-learn')

plt.show()
