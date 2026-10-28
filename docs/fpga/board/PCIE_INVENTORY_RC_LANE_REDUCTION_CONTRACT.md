# Contrato: reducción de bytes RC por carril

Preparar R12 desde el productor R8 y el selector R3, después de cerrar
R11 y verificar el punto de recuperación de R97. Conservar los candidatos
anteriores. La selección tardía de R11 no forma parte de este cambio.

## Cambio aislado

Sustituir solamente la acumulación de `rc_removed_span` por slot EOP:
guardar un valor de 13 bits por carril, inicialmente cero. Mantener las
condiciones procedurales existentes: EOP efectivo, propiedad y rama
terminal/normal. El valor sigue siendo `coriginal` o `cdelivered`.

Sumar los 16 valores mediante un árbol balanceado, con sumas sin signo
de 15 bits en cada nivel. No cambiar `nbytes`, rangos, tags, actualizaciones
de tablas, contador de completados, aritmética de prefijos, RQ, latencia,
reset, fallos ni reglas de vacío. No usar el nuevo arreglo para modificar
otros resultados. Un helper combinacional permite probar la reducción real.

## Correspondencia de eventos

El decodificador anterior escribe como máximo cuatro posiciones EOP.
Los punteros duplicados conservan el último slot declarado; cada slot
efectivo pertenece a un solo carril. La reducción nueva debe conservar
ese mismo conjunto de valores, incluso después de errores de protocolo.
No sumar dos veces un puntero duplicado ni habilitar escrituras X/Z.

`tools/check_ntrpu_pcie_rc_eop_slot_bijection.py` examinó 88.741
combinaciones de 0–4 punteros, usando 16 posiciones y una clase desconocida;
incluye 25.696 combinaciones con duplicados y dos controles negativos.
Es un modelo finito con supuestos explícitos, no una prueba de equivalencia
RTL. Con hasta cuatro valores de 13 bits, la suma máxima es 32.764.

## Criterios de aceptación

Comparar exactamente R12 contra R8 para limitar el cambio. Conservar por
hash el selector R3. Probar el helper real contra una suma secuencial
independiente: ceros, máximos, acarreos, truncamiento y X/Z en cada término.
Añadir casos integrados de EOP duplicados/desordenados, continuidad retenida,
terminal/normal, propiedad desconocida y estados posteriores a fallos.
Conservar íntegros todos los bancos anteriores, incluidos los 2.048 casos
adicionales de máscaras aritméticas de R11, mutantes y controles de evidencia.

Exigir RTL completo, síntesis VU5P/4 ns/`RuntimeOptimized` sin excepciones,
DCP acíclico y replay de la propia netlist con todos los suplementos.
Medir recursos, setup/hold y ruta crítica frente a R8. Respaldar resultados
y fallos localmente y en NFS antes de avanzar. El árbol escrito en RTL no
garantiza su mapeo. No acredita endpoint nativo, DDR/512 GiB ni diseño físico.
