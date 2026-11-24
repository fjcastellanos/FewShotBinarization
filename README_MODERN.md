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

## Alternativas al oversampling (revision)

Dos experimentos independientes para comparar con el oversampling (`full` = masking + oversampling). No se combinan con el oversampling ni con MixUp/CutMix, y sin los flags nuevos todo funciona exactamente igual que antes (mismos checkpoints, mismos muestreos).

- **Centered larger context** (`centered_context`): el patch anotado de 256x256 se mantiene fijo; en cada muestra se extrae un crop centrado en el de lado `s * 256`, con `s ~ U(1, 2)` (`--context-min/--context-max`), y se redimensiona a 256x256 para mantener el batch. El contexto es imagen real pero no se supervisa (salvo que sea otro patch anotado).
- **Random output masking** (`output_masking`): se extrae siempre el patch anotado, sin desplazamiento; se elimina la supervision de una fraccion `r ~ U(0, 0.95)` (`--mask-min/--mask-max`) del patch en bloques cuadrados aleatorios de `--mask-block` px (32 por defecto). La imagen de entrada no se modifica.

Ambos usan el mismo presupuesto de muestras por epoca que el oversampling (`-npatches` por pagina). La implementacion comun esta en `modern_fewshot/context_sampling.py`.

Modernos (requiere `--experiment masking`; checkpoint con sufijo `__ctx1-2` o `__rom0-0.95-b32`):

```bash
MODEL=segformer_b0 bash scripts/run_modern_alt_sampling.sh
# o directamente:
python main_modern.py ... --experiment masking --sampling centered_context
python main_modern.py ... --experiment masking --sampling output_masking
```

SAE (`main.py`, `-aug none -npatches 1024`; checkpoint con sufijo `__ctx1-2` o `__rom0-.95b32`):

```bash
python3 scripts/run_sae_table_baselines.py --jobs centered_context,output_masking --gpu 0
# o directamente:
python main.py ... -aug none -npatches 1024 --sae_sampling centered_context
python main.py ... -aug none -npatches 1024 --sae_sampling output_masking
```

En el SAE la supervision normalmente se desactiva poniendo la entrada a -1. Para estas dos alternativas la entrada se muestra sin enmascarar y los pixeles no supervisados reciben `sample_weight = 0` por pixel; el modelo se recompila con `mse` ponderado para que `val_mse` (checkpoint/early stopping) solo mida pixeles anotados. La validacion usa los mismos crops que el oversampling historico, con imagen real y peso = regiones anotadas.
