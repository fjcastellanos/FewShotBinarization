# Inicio rapido: experimentos modernos

Esta version esta construida sobre `antiguo.zip`. Los ficheros legacy se conservan byte a byte y la ruta moderna esta separada.

## 1. Comprobar que el legacy no se ha tocado

```bash
sha256sum -c LEGACY_SHA256.txt
```

Debe terminar con `OK` para todos los ficheros.

## 2. Entorno moderno

### Opcion A: entorno Python separado

```bash
python3 -m venv .venv-modern
source .venv-modern/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-modern.txt
pip install pytest
pytest -q
```

### Opcion B: Docker separado

No se modifica el `Dockerfile` historico. Para la ruta moderna:

```bash
docker build -f Dockerfile.modern -t fewshot-modern .
docker run --rm -it --gpus all \
  -v "$PWD":/workspace \
  -w /workspace \
  fewshot-modern
```

Dentro del contenedor:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pytest -q
```

## 3. Estructura de datasets esperada

```text
datasets/
  Dibco/
    train/
      SRC/
      GT/
    test/
      SRC/
      GT/
```

Los nombres de fichero de `SRC` y `GT` deben coincidir.

Con los valores por defecto (`PAGES_TRAIN=1`, `VAL_PAGES=1`) hacen falta al menos dos paginas en `train`.

## 4. Primer experimento recomendado

Ablacion completa 2x2 con SegFormer B0:

```bash
MODEL=segformer_b0 bash scripts/run_modern_2x2_ablation.sh
```

Ejecuta:

```text
baseline      masking=no   oversampling=no
oversampling  masking=no   oversampling=yes
masking       masking=yes  oversampling=no
full          masking=yes  oversampling=yes
```

Todos usan el mismo presupuesto de crops por epoca.

## 5. Ejecutar todos los datasets historicos

```bash
MODEL=segformer_b0 bash scripts/run_modern_all_datasets_2x2.sh
```

Por defecto recorre:

```text
Dibco Einsiedeln Palm PHI Salzinnes
```

## 6. Resultados compatibles con el formato antiguo

### Fichero de resultados

Cada experimento escribe por defecto en:

```text
results/modern/<modelo>/<experimento>.txt
```

Cada linea tiene exactamente las mismas columnas y separadores que `main.py` legacy:

```text
Train;Test;PAG;Num pages train;ANN;Num annotations per page;PAT;Num random patches;Ink rate;VAL;Th_bin;F1-val;P-val;R-val;Num annotated patches-val;Maximum num annotated patches-val;TEST;F1-test;P-test;R-test;Num annotated patches-test;Maximum num annotated patches-test;IoU-test;Specificity-test;TP-test;TN-test;FP-test;FN-test;TimePerPage(s);
```

No se anaden columnas modernas para no romper tus scripts de comparacion. El modelo y la condicion experimental quedan identificados por la ruta del fichero.

Puedes cambiar el fichero:

```bash
RESULTS_FILE=results/mi_resultado.txt MODEL=segformer_b0 EXPERIMENT=full bash scripts/run_modern_single.sh
```

O desactivar el append:

```bash
RESULTS_FILE=none MODEL=segformer_b0 EXPERIMENT=full bash scripts/run_modern_single.sh
```

### Imagenes de test

`tests/` vuelve a estar reservado a resultados, no a codigo. Se guardan los mismos sufijos que en el legacy:

```text
*_gt.png
*_gr.png
*_pred.png
*_pred_th.png
*_annotated_regions.png
```

La estructura moderna es:

```text
tests/modern/<modelo>/<experimento>/<checkpoint>/<dataset>/test/SRC/...
```

`_gr.png` conserva tambien la convencion historica: se guarda la imagen fuente invertida, igual que hacia `util.py` despues de `normalize_image()`.

Las visualizaciones de train/validacion se guardan en:

```text
train/modern/<modelo>/<experimento>/<checkpoint>/<dataset>/train/SRC/...
```

## 7. Checkpoints

Se guardan separados por modelo y experimento:

```text
models/modern/<modelo>/<experimento>/<dataset>__pt...__seedN.pt
```

El dataset forma parte del nombre, por lo que ejecutar Dibco y Palm con los mismos parametros no sobrescribe checkpoints.

Tambien se crean:

```text
<checkpoint>.history.json
<checkpoint>.summary.json
```

## 8. Parametros habituales

```bash
MODEL=segformer_b0 \
SEED=1 \
PAGES_TRAIN=1 \
VAL_PAGES=1 \
ANNOTATED_PATCHES=1 \
NPATCHES=1024 \
PATCH=256 \
BATCH_SIZE=4 \
GRAD_ACCUM=8 \
AUG="random flipH flipV rot scale" \
bash scripts/run_modern_2x2_ablation.sh
```

Batch efectivo nominal por defecto: `4 x 8 = 32`.

## 9. Tests de codigo

Los tests automaticos ya no estan en `tests/`:

```text
pytest_tests/
```

`pytest.ini` hace que:

```bash
pytest -q
```

solo busque ahi. `tests/` queda libre para las imagenes experimentales.
