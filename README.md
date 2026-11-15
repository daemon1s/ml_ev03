# EV03 · Detección de patentes

Proyecto de Deep Learning para localizar patentes vehiculares. La auditoría de etiquetas ayuda a identificar imágenes de entrenamiento que conviene revisar antes de entrenar el detector.

El repositorio documenta la [decisión de arquitectura y el origen del corpus](docs/decision-modelo-y-datos.md), junto con la [especificación de la auditoría y su interfaz](docs/especificacion-auditoria.md). La aplicación, el entrenamiento y los resultados están pendientes.

## Corpus local

Configura `DATASET_DIR` con la ruta a `entregable/dataset` de EV02 en tu equipo. La raíz contiene `data.yaml` y las carpetas `train`, `valid` y `test`, cada una con `images` y `labels`. Configura `AUDIT_WORK_DIR` con una carpeta local escribible fuera del corpus y de este repositorio. Estas variables forman parte del contrato de la futura implementación.

| Partición | Imágenes | Uso |
| --- | ---: | --- |
| Entrenamiento | 1.101 | Auditoría, aumento de datos y entrenamiento |
| Validación | 289 | Seguimiento durante el entrenamiento |
| Prueba | 139 | Evaluación final |

El corpus curado de EV02 deriva de [My First Project v2 en Roboflow Universe](https://universe.roboflow.com/luiss-workspace-jfeoo/my-first-project-yzokm/dataset/2), bajo licencia CC BY 4.0 según el `data.yaml` local. La descarga de origen y el corpus curado son versiones distintas. Quien necesite repetir la corrida exacta requiere acceso a la copia curada por un canal acordado con la dupla.

## Contenido previsto

Incluir código o cuaderno de entrenamiento, configuración de ejemplo, dependencias, instrucciones reproducibles, métricas y curvas agregadas, y ejemplos sintéticos o anonimizados. Incluir el modelo final `.keras` o `.h5` cuando exista y se haya comprobado que puede cargarse. Las fotos, etiquetas, recortes, registros de revisión y evidencia de EV02 se almacenan fuera del checkout. Para una demostración sin acceso al corpus, usar una imagen sintética o anonimizada.
