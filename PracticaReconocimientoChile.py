import joblib  # Para guardar el modelo entrenado 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

# Lista de imágenes de entrenamiento
imagenes_ruta = ['Entrenamiento1.jpg', 'Entrenamiento2.jpg', 'Entrenamiento3.jpg']

# Inicializar listas para almacenar características y etiquetas
X_train = []  # Características (media de píxeles en HSV)
y_train = []  # Etiquetas (en este caso, todos serán 1 para verde fuerte)

# Procesar cada imagen
for img_ruta in imagenes_ruta:
    # Cargar la imagen
    imagen = cv2.imread(img_ruta)
    
    # Aplicar un filtro bilateral para preservar bordes
    imagen_suavizada = cv2.bilateralFilter(imagen, 9, 75, 75)

    # Convertir la imagen de BGR a HSV
    imagen_hsv = cv2.cvtColor(imagen_suavizada, cv2.COLOR_BGR2HSV)

    # Definir límites de color verde fuerte para segmentación
    limite_inferior_fuerte = np.array([40, 100, 100], dtype=np.uint8)  
    limite_superior_fuerte = np.array([80, 255, 255], dtype=np.uint8)  

    # Definir límites de color verde claro (para sombras o zonas más claras)
    limite_inferior_claro = np.array([30, 40, 40], dtype=np.uint8)
    limite_superior_claro = np.array([70, 255, 255], dtype=np.uint8)

    # Crear máscaras para los píxeles que cumplen con los límites de verde fuerte y verde claro
    mascara_verde_fuerte = cv2.inRange(imagen_hsv, limite_inferior_fuerte, limite_superior_fuerte)
    mascara_verde_claro = cv2.inRange(imagen_hsv, limite_inferior_claro, limite_superior_claro)

    # Combinar ambas máscaras
    mascara_verde = cv2.bitwise_or(mascara_verde_fuerte, mascara_verde_claro)

    # Extraer los píxeles correspondientes al verde fuerte y claro
    pixeles_verdes = imagen_hsv[mascara_verde != 0]

    # Calcular la media de los píxeles extraídos
    if len(pixeles_verdes) > 0:
        media_pixeles = np.mean(pixeles_verdes, axis=0)
        X_train.append(media_pixeles)
        y_train.append(1)  # Etiqueta para verde fuerte o claro
        
        # Calcular y mostrar la matriz de covarianza
        covarianza = np.cov(pixeles_verdes, rowvar=False)
        
        # Mostrar imágenes y resultados intermedios
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
        plt.title('Imagen Original')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mascara_verde, cmap='gray')
        plt.title('Máscara Combinada de Verde')
        
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(imagen_hsv, cv2.COLOR_HSV2RGB))
        plt.title('Imagen HSV')
        
        plt.suptitle(f'Imagen: {img_ruta}, Media: {media_pixeles}, Covarianza: {covarianza}')
        plt.show()

# Convertir listas a arrays de numpy
X_train = np.array(X_train)
y_train = np.array(y_train)

# Entrenar el clasificador Naive Bayes
modelo = GaussianNB()
modelo.fit(X_train, y_train)

# Guardar el modelo entrenado (comentado si no es necesario)
#joblib.dump(modelo, 'modeloEntrenadoVerdeFuerte.pkl')
#print("Modelo entrenado y guardado como 'modeloEntrenadoVerdeFuerte.pkl'")

# Clasificación de una nueva imagen
nueva_imagen = cv2.imread('Prueba2.jpg')  # Cargar la nueva imagen
nueva_imagen_suavizada = cv2.bilateralFilter(nueva_imagen, 9, 75, 75)
nueva_imagen_hsv = cv2.cvtColor(nueva_imagen_suavizada, cv2.COLOR_BGR2HSV)

# Aplicar el mismo límite de color verde fuerte y verde claro
mascara_nueva_fuerte = cv2.inRange(nueva_imagen_hsv, limite_inferior_fuerte, limite_superior_fuerte)
mascara_nueva_claro = cv2.inRange(nueva_imagen_hsv, limite_inferior_claro, limite_superior_claro)

# Combinar ambas máscaras
mascara_nueva = cv2.bitwise_or(mascara_nueva_fuerte, mascara_nueva_claro)
pixeles_nuevos = nueva_imagen_hsv[mascara_nueva != 0]

# Clasificar los píxeles usando el modelo entrenado
X_nueva = np.mean(pixeles_nuevos, axis=0).reshape(1, -1)  # Obtener características
prediccion = modelo.predict(X_nueva)

# Crear nueva imagen para visualizar clases
clases_imagen = np.zeros(nueva_imagen.shape[:2], dtype=np.uint8)  # Imagen en blanco

# Asignar valores de gris: 0 para fondo, 255 para objeto de interés (verde fuerte o claro)
clases_imagen[mascara_nueva != 0] = 255  

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
