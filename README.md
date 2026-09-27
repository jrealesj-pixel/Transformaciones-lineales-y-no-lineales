# Transformaciones de Intensidad en Imágenes

*(Borrador de README — ajústalo a tu manera antes de entregar, esto es
solo una base con toda la información que pide el profesor)*

## ¿Qué hace este proyecto?

Este repo desarrolla la tarea de **Visión por Computador** sobre
transformaciones puntuales de intensidad (lineales y no lineales) en
imágenes en escala de grises y en color (RGB). Se implementaron y
probaron los 15 ejercicios pedidos: transformaciones lineales, manejo de
saturación (clipping vs normalización), transformación por tramos,
inversión, corrección gamma, y varias manipulaciones por canal en RGB
(escalado por canal, eliminación de canal, gamma por canal, intercambio
de canales, comparación con escala de grises, y diseño de un estilo
visual propio).

## Estructura del repositorio

```
.
├── transforms.py     # Todas las funciones de transformación e imágenes/histogramas
├── demo.ipynb         # Notebook con los 15 ejercicios, ya ejecutado (imágenes + histogramas + interpretación)
├── images/            # Imágenes de entrada usadas en los ejercicios
│   ├── foto_gris.jpg     # imagen en escala de grises
│   ├── foto_oscura.jpg   # versión oscura de la misma imagen (para los ejercicios de gamma)
│   └── foto_color.jpg    # imagen a color (RGB)
├── requirements.txt   # Librerías necesarias
├── .gitignore
└── README.md
```

## Cómo ejecutar el proyecto (en VS Code)

1. Clona o descarga el repositorio y ábrelo en VS Code.
2. Crea un entorno virtual (opcional pero recomendado):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Abre `demo.ipynb` en VS Code (con la extensión de **Jupyter** instalada)
   y selecciona el intérprete/kernel del entorno virtual creado.
5. Ejecuta las celdas en orden (`Run All`). El notebook ya viene ejecutado
   con todas las salidas, así que también se puede revisar directamente
   sin volver a correrlo.

## Sobre las imágenes usadas

Las imágenes de `images/` se generaron a partir de imágenes de prueba
estándar incluidas en la librería `scikit-image` (no son fotos propias),
para que el proyecto sea 100% reproducible sin depender de archivos
externos. Si quieres usar tus propias fotos, solo reemplaza los archivos
en `images/` (o cambia las rutas al inicio del notebook) manteniendo los
mismos nombres, o usando tu propio nombre y actualizando la celda de
carga de imágenes.

## Uso de herramientas de inteligencia artificial

**Nivel: (ALTO / MEDIO / BAJO / NINGUNO — completar según lo que
realmente hiciste)**

Descripción: *(explica aquí qué partes trabajaste con ayuda de IA —por
ejemplo, estructura del proyecto, funciones de `transforms.py`, redacción
de interpretaciones— y qué partes hiciste/revisaste tú directamente).*
