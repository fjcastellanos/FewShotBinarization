# Ruta moderna de binarizacion con supervision parcial

Esta extension se ha integrado sobre la copia exacta de `antiguo.zip`, no sobre la instantanea legacy que venia dentro del ZIP moderno anterior.

El objetivo es que los experimentos historicos sigan ejecutandose con `main.py` y sus ficheros originales, mientras que los nuevos experimentos entren por `main_modern.py` y `modern_fewshot/`.

Para empezar, consulta `START_HERE_MODERN.md`.

## Garantia de retrocompatibilidad

Los ficheros procedentes de `antiguo.zip` no se modifican. Sus hashes estan registrados en `LEGACY_SHA256.txt` y pueden comprobarse con:

```bash
sha256sum -c LEGACY_SHA256.txt
```

El `Dockerfile` original tambien queda intacto. `Dockerfile.modern` es independiente.

## Componentes modernos

```text
main_modern.py
modern_fewshot/
  data.py
  inference.py
  losses.py
  models.py
  reporting.py
  train.py
scripts/
  run_modern_single.sh
  run_modern_ablation.sh
  run_modern_2x2_ablation.sh
  run_modern_all_models.sh
  run_modern_all_datasets_2x2.sh
  test_modern_checkpoint.sh
pytest_tests/
requirements-modern.txt
Dockerfile.modern
```

## Ablacion

| experimento | oversampling | masked loss |
|---|---:|---:|
| `baseline` | no | no |
| `oversampling` | si | no |
| `masking` | no | si |
| `full` | si | si |

Todos los experimentos usan el mismo numero de crops por epoca. Con una pagina y `NPATCHES=1024`, todos procesan exactamente 1024 crops por epoca.

Con la misma seed, `baseline`/`masking` comparten muestreo uniforme y `oversampling`/`full` comparten coordenadas y augmentations guiadas.

## Modelos

```text
segformer_b0
segformer_b1
segformer_b2
segformer_doc_b3
deeplabv3_resnet50
deeplabv3_resnet101
```

B0/B1/B2 parten de encoders MiT preentrenados. `segformer_doc_b3` usa un checkpoint ya especializado en binarizacion documental y debe interpretarse como referencia de transfer learning.

`deeplabv3_resnet50` y `deeplabv3_resnet101` usan DeepLabV3 de torchvision con backbone ResNet preentrenado en ImageNet y una cabeza de segmentacion de 2 clases inicializada para esta tarea. Se incluyen como baselines CNN modernos; no son transformers.

El lanzador `sh_modern_models.sh` incluye por defecto los 6 modelos anteriores y los 7 datasets (`Bickley`, `Dibco`, `Einsiedeln`, `ISOS`, `Palm`, `PHI`, `Salzinnes`). Con una sola configuracion `full`, `./sh_modern_models.sh train` lanza 6 x 7 = 42 entrenamientos secuenciales.

## Early stopping

Defaults:

```text
max epochs = 50
patience = 10
min_delta = 0.0001
physical batch = 4
gradient accumulation = 8
effective batch nominal = 32
```

La seleccion de checkpoint se hace por F1 sobre paginas completas de validacion. El threshold se selecciona sobre validacion y se almacena con el checkpoint.

## Compatibilidad de resultados

La ruta moderna exporta dos niveles de resultados:

1. JSON moderno (`.history.json` y `.summary.json`) con metadata completa.
2. Salida legacy para comparar directamente con los experimentos existentes.

La salida legacy mantiene:

- las mismas columnas `;` de `main.py` mediante `-res`;
- el mismo formateo porcentual, incluida la convencion historica de los campos de confusion;
- comparacion binaria estricta `prediction > threshold`;
- los sufijos de imagen `_gt.png`, `_gr.png`, `_pred.png`, `_pred_th.png`, `_annotated_regions.png`;
- `_gr.png` invertida, igual que la salida historica.

Los resultados modernos se separan por modelo/experimento en la ruta, en vez de anadir columnas nuevas al fichero legacy.

## Directorios

```text
models/modern/<modelo>/<experimento>/...
results/modern/<modelo>/<experimento>.txt
tests/modern/<modelo>/<experimento>/...
train/modern/<modelo>/<experimento>/...
```

`tests/` contiene resultados experimentales. Las pruebas de software estan en `pytest_tests/`.
