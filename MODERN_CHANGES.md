# Cambios de esta version integrada

- Base reconstruida directamente desde `antiguo.zip`.
- `main.py`, `util.py`, `utilIO.py`, `CNNmodel.py`, `Dockerfile` y el resto de ficheros legacy se conservan byte a byte.
- `LEGACY_SHA256.txt` contiene los hashes de los ficheros de `antiguo.zip`.
- Los tests automaticos se movieron de `tests/` a `pytest_tests/`.
- `pytest.ini` restringe la busqueda de pytest a `pytest_tests/`.
- `tests/` vuelve a quedar reservado a imagenes resultantes de los experimentos.
- La evaluacion moderna guarda `_gt.png`, `_gr.png`, `_pred.png`, `_pred_th.png` y `_annotated_regions.png` con la convencion historica.
- Se anadio `-res/--res` a `main_modern.py` y se reproduce el layout de columnas del resultado legacy para comparacion directa.
- Se anadieron IoU, specificity, TP, TN, FP y FN al resumen moderno y a la fila legacy.
- Se mide el tiempo de inferencia por pagina y su media.
- El threshold moderno usa la comparacion estricta `>` para coincidir con `util.run_test()`.
- Los checkpoints incluyen el dataset y se separan por modelo/experimento para evitar sobrescrituras entre datasets.
- Se anadio `scripts/run_modern_all_datasets_2x2.sh` para Dibco, Einsiedeln, Palm, PHI y Salzinnes.
- Se anadio `Dockerfile.modern` sin modificar el Dockerfile TensorFlow historico.
- Se mantienen las cuatro condiciones `baseline`, `oversampling`, `masking` y `full` con presupuesto fijo de crops.
- Se mantienen early stopping por F1 de pagina completa, checkpoint del mejor modelo, AMP y gradient accumulation.
