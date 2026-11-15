# EV03 · Detección de patentes

Proyecto de Deep Learning para localizar patentes vehiculares. La auditoría de etiquetas ayuda a identificar imágenes de entrenamiento que conviene revisar antes de entrenar el detector.

El repositorio documenta la [decisión de arquitectura y el origen del corpus](docs/decision-modelo-y-datos.md), junto con la [especificación de la auditoría y su interfaz](docs/especificacion-auditoria.md).

## Entrega y defensa

**Plazo de entrega: domingo 15 de noviembre de 2026 a las 23:00, hora de Santiago.** Fecha indicada por el equipo.

- Grabar un video expositivo que muestre el sistema funcionando y recorra el código que ejecuta la demostración.
- Preparar a ambos integrantes para explicar las estructuras de datos, los `import`, el módulo o biblioteca de origen y qué funciones se invocan durante la carga de imágenes, el entrenamiento y la inferencia.
- Entregar el código reproducible y el modelo entrenado en `.keras` o `.h5`, con la justificación del diseño y de los aumentos, curvas y métricas de evaluación. La auditoría de etiquetas puede avanzar en paralelo al detector.

## Corpus local

Configura `DATASET_DIR` con la ruta a `entregable/dataset` de EV02 en tu equipo. La raíz contiene `data.yaml` y las carpetas `train`, `valid` y `test`, cada una con `images` y `labels`. Configura `AUDIT_WORK_DIR` con una carpeta local escribible fuera del corpus y de este repositorio.

### Auditoría visual de las cajas

La aplicación de escritorio requiere Python con Tk y Pillow. Usa Ollama para evaluar los recortes en la misma máquina. Con Ollama instalado, descarga el modelo visual con `ollama pull qwen3-vl:4b-instruct`. Abre el auditor desde PowerShell, en la raíz del repositorio.

```powershell
$env:DATASET_DIR = '<ruta al corpus curado de EV02>'
$env:AUDIT_WORK_DIR = '<carpeta local fuera del corpus y del repositorio>'
$env:AUDIT_OLLAMA_URL = 'http://127.0.0.1:11434'
$env:AUDIT_MODEL = 'qwen3-vl:4b-instruct'
.\.venv\Scripts\python.exe audit_gui.py
```

Para abrirla con doble clic, copia `auditoria.env.example.bat` como `auditoria.env.bat`, completa las rutas y ejecuta `iniciar_auditoria.bat`. El archivo `auditoria.env.bat` es local y Git lo ignora. Si falta una variable, el script lo indica y no abre la ventana.

La pestaña **Historial** lista las cajas que el modelo ya auditó, con su veredicto, el motivo y la decisión humana si existe. Al seleccionar una fila, la ventana vuelve a la cola y abre esa caja.

Para grabar el flujo, activa **Mostrar todas las cajas**, selecciona una caja de una imagen sintética o autorizada y pulsa **Auditar esta caja**. Con **Seguir auditoría** activado, la ventana muestra el recorte con borde naranja, el progreso y el resultado. El **Registro de auditoría** muestra cuándo se envió la caja al modelo y qué respondió. También guarda el historial en `auditoria.log`, dentro de `AUDIT_WORK_DIR`.

Las cajas que el modelo aprueba salen de la cola normal. Las dudosas, mal delimitadas y las que parecen otro objeto quedan para revisión humana. Los veredictos automáticos no cambian las etiquetas.

Antes de auditar todo `train`, compara el resultado del modelo con las decisiones de la dupla en una muestra con patentes claras y errores conocidos. **Auditar 50 cajas** ensaya el lote con las primeras cajas pendientes y **Auditar esta caja** permite elegir casos concretos. **Exportar copia de train** aplica las decisiones guardadas en una carpeta nueva. El archivo `resumen.json` indica cuántas cajas quedaron sin resolver.

| Partición | Imágenes | Uso |
| --- | ---: | --- |
| Entrenamiento | 1.141 | Auditoría, aumento de datos y entrenamiento (1.101 de EV02 y 40 manuales) |
| Validación | 289 | Seguimiento durante el entrenamiento |
| Prueba | 139 | Evaluación final |

El corpus curado de EV02 deriva de [My First Project v2 en Roboflow Universe](https://universe.roboflow.com/luiss-workspace-jfeoo/my-first-project-yzokm/dataset/2), bajo licencia CC BY 4.0 según el `data.yaml` local. La descarga de origen y el corpus curado son versiones distintas. Quien necesite repetir la corrida exacta requiere acceso a la copia curada por un canal acordado con la dupla.

## Contenido previsto

Incluir código o cuaderno de entrenamiento, configuración de ejemplo, dependencias, instrucciones reproducibles, métricas y curvas agregadas, y ejemplos sintéticos o anonimizados. Incluir el modelo final `.keras` o `.h5` cuando exista y se haya comprobado que puede cargarse. Las fotos, etiquetas, recortes, registros de revisión y evidencia de EV02 se almacenan fuera del checkout. Para una demostración sin acceso al corpus, usar una imagen sintética o anonimizada.

## Piloto del detector

Con Python 3.11, preparar un entorno virtual e instalar las dependencias indicadas. Definir `DATASET_DIR` y `PILOT_OUTPUT_DIR`; esta última debe apuntar a una carpeta nueva fuera del corpus, preferiblemente bajo `runs/` en el checkout. Ejecutar estos comandos desde la raíz del repositorio en PowerShell:

```powershell
py -3.11 -m venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements-pilot.txt
.venv/Scripts/python.exe pilot.py train
.venv/Scripts/python.exe pilot.py verify
```

El piloto usa 16 imágenes de entrenamiento y 4 de validación durante 2 épocas. Comprueba la carga de etiquetas, el ajuste, el guardado de `pilot.keras` y la inferencia desde un proceso nuevo. `prediccion-local.png` muestra cajas candidatas con un umbral de confianza de cero para probar el flujo. Se guarda localmente porque contiene una fotografía del corpus. La evaluación final se realiza aparte con la partición de prueba y umbrales definidos antes de medirla.

La corrida verificada del 14 de noviembre de 2026 dejó `runs/pilot-ev03-verificado/pilot.keras` y `runs/pilot-ev03-verificado/prediccion-local.png` en este equipo. La imagen contiene tres cajas candidatas. La ruta `runs/` está excluida de Git.

Para el video y la defensa, seguir una imagen y su etiqueta a través de [pilot.py](pilot.py). `PIL.Image` abre y redimensiona la imagen. `numpy` agrupa las imágenes y las cajas en arreglos que consume el modelo. La función propia `box_xyxy` transforma centro y tamaño normalizados a esquinas en píxeles. `keras_cv.models` construye YOLOv8 y carga su extractor preentrenado. `keras` configura el optimizador y la parada temprana, y guarda y recarga el artefacto. `PIL.ImageDraw` dibuja las cajas que devuelve la inferencia.
