"""
transforms.py
==============

Funciones auxiliares para la tarea "Transformaciones de Intensidad en
Imágenes". Aquí están todas las operaciones matemáticas (lineales, gamma,
por canal, etc.) y las funciones para mostrar imágenes/histogramas.

La idea de tener esto en un .py separado es que el notebook (demo.ipynb)
quede limpio: solo se llaman estas funciones y se comentan los resultados,
sin llenar el notebook de código repetido.

Todas las funciones trabajan con imágenes como arreglos de NumPy, en el
formato que da OpenCV al leer con cv2.imread():
    - Escala de grises: arreglo 2D, shape (alto, ancho), dtype uint8
    - Color: arreglo 3D, shape (alto, ancho, 3), en orden de canales BGR
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Utilidades básicas de lectura y despliegue
# ---------------------------------------------------------------------------

def leer_gris(ruta):
    """Lee una imagen desde disco y la devuelve en escala de grises (uint8)."""
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return img


def leer_color(ruta):
    """
    Lee una imagen a color desde disco y la devuelve en orden RGB (uint8),
    que es el orden que espera matplotlib para mostrarla correctamente.
    OpenCV lee en BGR por defecto, por eso hacemos la conversión aquí.
    """
    img_bgr = cv2.imread(ruta, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def mostrar_par(original, transformada, titulo_izq="Original",
                 titulo_der="Transformada", cmap=None, figsize=(9, 4.5)):
    """
    Muestra dos imágenes lado a lado (original vs transformada).
    Si las imágenes son en escala de grises, se pasa cmap='gray'.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(original, cmap=cmap, vmin=0, vmax=255 if cmap == "gray" else None)
    ax[0].set_title(titulo_izq)
    ax[0].axis("off")

    ax[1].imshow(transformada, cmap=cmap, vmin=0, vmax=255 if cmap == "gray" else None)
    ax[1].set_title(titulo_der)
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def histograma_gris(imagen, ax=None, titulo="Histograma", color="black"):
    """Dibuja el histograma (256 niveles) de una imagen en escala de grises."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(imagen.ravel(), bins=256, range=(0, 255), color=color)
    ax.set_title(titulo)
    ax.set_xlabel("Nivel de intensidad")
    ax.set_ylabel("Número de píxeles")
    return ax


def histogramas_rgb(imagen_rgb, ax=None, titulo="Histograma RGB"):
    """
    Dibuja, en un mismo eje, los 3 histogramas (R, G, B) de una imagen a
    color, cada uno con su color correspondiente.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    colores = ("red", "green", "blue")
    etiquetas = ("R", "G", "B")
    for canal, color, etiqueta in zip(range(3), colores, etiquetas):
        ax.hist(imagen_rgb[:, :, canal].ravel(), bins=256, range=(0, 255),
                 color=color, alpha=0.5, label=etiqueta)
    ax.set_title(titulo)
    ax.set_xlabel("Nivel de intensidad")
    ax.set_ylabel("Número de píxeles")
    ax.legend()
    return ax


def mostrar_imagen_e_histograma(imagen, cmap=None, titulo_img="Imagen",
                                 titulo_hist="Histograma", figsize=(9, 4)):
    """
    Muestra una imagen junto a su(s) histograma(s) en una sola figura.
    Detecta automáticamente si es gris (2D) o RGB (3D) para escoger el
    histograma adecuado.
    """
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].imshow(imagen, cmap=cmap, vmin=0, vmax=255 if cmap == "gray" else None)
    ax[0].set_title(titulo_img)
    ax[0].axis("off")

    if imagen.ndim == 2:
        histograma_gris(imagen, ax=ax[1], titulo=titulo_hist)
    else:
        histogramas_rgb(imagen, ax=ax[1], titulo=titulo_hist)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. Parte 1: Transformaciones lineales (escala de grises)
# ---------------------------------------------------------------------------

def transform_lineal(imagen, a, b, modo="clip"):
    """
    Aplica la transformación lineal  I' = a*I + b

    Parámetros
    ----------
    imagen : np.ndarray (uint8)
        Imagen de entrada (gris o color).
    a : float
        Controla el contraste (a > 1 aumenta contraste, a < 1 lo reduce).
    b : float
        Controla el brillo (se suma a cada píxel).
    modo : "clip" o "normalizar"
        - "clip": los valores fuera de [0, 255] se recortan (saturan).
        - "normalizar": la imagen resultante se reescala completa para que
          el mínimo y el máximo queden dentro de [0, 255], sin perder el
          rango relativo entre píxeles.

    Devuelve
    --------
    np.ndarray (uint8) con el mismo tamaño que la imagen de entrada.
    """
    # Trabajamos en float para no perder información antes de recortar/normalizar
    resultado = imagen.astype(np.float32) * a + b

    if modo == "clip":
        resultado = np.clip(resultado, 0, 255)
    elif modo == "normalizar":
        minimo, maximo = resultado.min(), resultado.max()
        if maximo - minimo < 1e-6:
            # Imagen constante: evitamos dividir por cero
            resultado = np.zeros_like(resultado)
        else:
            resultado = (resultado - minimo) / (maximo - minimo) * 255
    else:
        raise ValueError("modo debe ser 'clip' o 'normalizar'")

    return resultado.astype(np.uint8)


def transform_piecewise(imagen, p1=(80, 40), p2=(180, 200)):
    """
    Transformación por tramos (piecewise linear), pensada para:
        - oscurecer las intensidades bajas
        - mantener aprox. las intensidades medias
        - aumentar las intensidades altas

    Se define con dos puntos de control (p1 y p2), cada uno como
    (entrada, salida). Por defecto:
        (0,0) -> p1=(80,40) -> p2=(180,200) -> (255,255)
    Esto arma 3 segmentos lineales que oscurecen las sombras (0-80),
    dejan casi igual los tonos medios (80-180) y realzan las luces (180-255).

    Funciona tanto para imágenes en escala de grises como a color (se
    aplica el mismo mapeo a cada canal).
    """
    (x1, y1), (x2, y2) = p1, p2

    # Construimos una tabla de mapeo (LUT) de 256 valores: para cada nivel
    # de entrada (0-255) calculamos a qué nivel de salida corresponde.
    lut = np.zeros(256, dtype=np.float32)
    for i in range(256):
        if i <= x1:
            # Tramo 1: de (0,0) a (x1,y1)
            lut[i] = (y1 / x1) * i if x1 != 0 else 0
        elif i <= x2:
            # Tramo 2: de (x1,y1) a (x2,y2)
            lut[i] = y1 + (y2 - y1) * (i - x1) / (x2 - x1)
        else:
            # Tramo 3: de (x2,y2) a (255,255)
            lut[i] = y2 + (255 - y2) * (i - x2) / (255 - x2) if x2 != 255 else y2

    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return cv2.LUT(imagen, lut)


def invertir(imagen):
    """Inversión de intensidades: I' = 255 - I (efecto 'negativo')."""
    return (255 - imagen.astype(np.int16)).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Parte 2: Transformaciones no lineales (gamma)
# ---------------------------------------------------------------------------

def correccion_gamma(imagen, gamma, c=1.0):
    """
    Aplica la corrección gamma:  I' = c * I^gamma

    La imagen se normaliza primero a [0,1], se aplica la potencia y se
    vuelve a escalar a [0,255]. gamma < 1 aclara la imagen (realza sombras),
    gamma > 1 la oscurece (realza luces).
    """
    normalizada = imagen.astype(np.float32) / 255.0
    resultado = c * np.power(normalizada, gamma)
    resultado = np.clip(resultado, 0, 1) * 255
    return resultado.astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Parte 3: Transformaciones en imágenes RGB (por canal)
# ---------------------------------------------------------------------------

def transform_lineal_por_canal(imagen_rgb, params):
    """
    Aplica una transformación lineal distinta a cada canal.

    params : dict con claves 'R', 'G', 'B', cada una como tupla (a, b)
        Ejemplo: {'R': (1.2, 10), 'G': (1.0, 0), 'B': (0.8, -10)}
    """
    salida = imagen_rgb.copy().astype(np.float32)
    for idx, canal in enumerate(("R", "G", "B")):
        a, b = params[canal]
        salida[:, :, idx] = np.clip(salida[:, :, idx] * a + b, 0, 255)
    return salida.astype(np.uint8)


def transform_canal_unico(imagen_rgb, canal, a, b):
    """
    Aplica una transformación lineal (a*I + b) SOLO al canal indicado
    ('R', 'G' o 'B'), dejando los demás intactos.
    """
    indices = {"R": 0, "G": 1, "B": 2}
    idx = indices[canal]
    salida = imagen_rgb.copy().astype(np.float32)
    salida[:, :, idx] = np.clip(salida[:, :, idx] * a + b, 0, 255)
    return salida.astype(np.uint8)


def eliminar_canal(imagen_rgb, canal):
    """Pone en cero el canal indicado ('R', 'G' o 'B')."""
    indices = {"R": 0, "G": 1, "B": 2}
    idx = indices[canal]
    salida = imagen_rgb.copy()
    salida[:, :, idx] = 0
    return salida


def gamma_por_canal(imagen_rgb, gamma_r, gamma_g, gamma_b):
    """Aplica una corrección gamma distinta a cada canal R, G, B."""
    salida = np.zeros_like(imagen_rgb)
    gammas = (gamma_r, gamma_g, gamma_b)
    for idx in range(3):
        normalizado = imagen_rgb[:, :, idx].astype(np.float32) / 255.0
        canal_transformado = np.power(normalizado, gammas[idx]) * 255
        salida[:, :, idx] = np.clip(canal_transformado, 0, 255).astype(np.uint8)
    return salida


def intercambiar_canales(imagen_rgb, canal_a, canal_b):
    """
    Intercambia dos canales de la imagen (por ejemplo R<->B o R<->G).
    canal_a, canal_b: 'R', 'G' o 'B'
    """
    indices = {"R": 0, "G": 1, "B": 2}
    ia, ib = indices[canal_a], indices[canal_b]
    salida = imagen_rgb.copy()
    salida[:, :, [ia, ib]] = salida[:, :, [ib, ia]]
    return salida


def a_escala_grises(imagen_rgb):
    """Convierte una imagen RGB a escala de grises usando la fórmula estándar de luminancia."""
    return cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2GRAY)
