# Bitmap RQ: regresión completa del monitor

## Dependencia y alcance

La tanda se habilita después del PASS revisado y respaldado del
[banco C enfocado](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_FOCUS_R1.md):
42.910 filas y cuatro defectos reales de fuente. El cambio del bitmap
se integra en un monitor con nombre propio. Se conservan sin cambios
las referencias R3, numérica y de valores, el selector RC, las cinco
etapas y los oráculos originales.

## Validación exigida

Se ejecutan **183.058 filas nuevas** en 16 bancos; ninguna fila aprobada
de otra variante sustituye esta ejecución. Se comparan 447 bits públicos
antes y después de cada flanco, con 31 sondas de quietud y 62 muestras
entre flancos/restauraciones. Los seis bancos drenados deben terminar
sin cancelar tráfico original. Continúan los escenarios posteriores a
fallos y las entradas X/Z del corpus.

Se compilan las 26 unidades declaradas y seis controles negativos de
entrada. La revisión mantiene las funciones de contabilidad de la tanda
978 y rechaza 34 corrupciones de evidencia; además enlaza por hash el C
que aprobó el banco enfocado, sus fuentes y su paquete de resultados.
No se transfieren aprobaciones a código distinto.

## Ejecución y decisión

El coordinador `continue_symbolic_monitor_predecode_rq_bitmap_full_r1.py`
espera el cierre anterior, congela un snapshot nuevo y conserva cada
resultado en D: y NFS antes de revisar. El límite es 1.800 segundos y
dos hilos. Un fallo detiene la dependencia y conserva el diagnóstico;
no reinicia automáticamente la prueba.

Solo el PASS completo habilita síntesis propia, comparación de recursos
y timing con 979 y posterior replay del netlist propio. Esta preparación
no atribuye una mejora de frecuencia ni califica timing físico, placa
o inferencia. Los criterios restantes de V0–V9 permanecen abiertos según
su evidencia específica.
