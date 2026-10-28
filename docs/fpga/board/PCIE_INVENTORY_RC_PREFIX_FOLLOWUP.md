# Próximo experimento: admisión RQ por eventos efectivos

Propuesta dependiente del cierre del replay de la netlist R5. Conservar
[sus resultados](PCIE_INVENTORY_RC_PREFIX_RESULTS.md) y las variantes previas;
no modificar fuentes, bancos ni revisiones ya ejecutadas.

## Ruta que motiva el cambio

La síntesis R5 revisada del 2026-10-08 mide WNS **−8,806 ns**, frente a
−9,877 ns del original. El peor camino tiene **12,789 ns y 51 niveles**,
desde `producer/allocated_tags_reg[51]` hasta `remaining_bytes_reg[20]`.
Pasa por liberaciones RC, propiedad, capacidad/admisión `rq_prefix_valid`,
datos de la nueva reserva y acumulación de bytes. La lógica RQ mantiene
cuatro etapas por las cuatro posiciones posibles de encabezado, aunque
el decodificador sólo puede declarar dos SOP efectivos.

El mejor WNS no implica mejora global: TNS empeora de −109536,984 a
−127995,078 ns; setup y hold siguen fallando. Medir nuevamente recursos,
WNS, TNS y endpoints, sin convertir esta estimación en frecuencia física.

## Transformación acotada

Partir de R5. Mantener sus prefijos RC y el contador global original.
Predecodificar los cero, uno o dos SOP RQ efectivos y ordenar su admisión
por lane. Reducir la cadena de decisiones de capacidad a esos eventos;
conservar la exploración física de payload/EOP y sus fallos independientes.

Un puntero duplicado produce un único SOP; prevalece el último slot de
descriptor, como en `sop_map` y `rq_start_slot`. Punteros invertidos se
procesan por lane, sin confundir ese orden con el número del slot. Los
punteros malformados a DW4/DW12 conservan sus efectos posteriores al fallo.
Un puntero desconocido no puede convertirse en un encabezado conocido.

La primera admisión observa la tabla propia después de liberaciones RC;
la segunda ve la primera reserva efectiva. Conservar duplicados, capacidad,
reutilización en el mismo beat y la prioridad de las dos escrituras finales.
No suponer que `tag_count` está entre 0 y 256 después de errores: preservar
su rango almacenado de 9 bits, la resta de hasta cuatro liberaciones y
los incrementos con los anchos y signos originales. No limitar ni reparar
estados malformados para facilitar la transformación.

## Pruebas antes de sintetizar

1. Modelo independiente de eventos frente a la exploración por lane:
   todas las combinaciones de cantidad/punteros SOP, duplicados e inversión;
   slots y tags iguales/distintos, RQ read/write, propiedad y liberaciones RC.
2. Límites de capacidad: counts almacenados 0–511, cero a cuatro
   liberaciones y hasta dos admisiones; conservar resultados negativos y
   superiores a 256. Añadir X/Z en punteros, tags, kind y contexto propio.
3. Banco integrado intacto de 22.339 filas/68 escenarios, siete mutantes,
   todas las palabras internas y controles de evidencia. Mantener los
   suplementos R5 de selección, aritmética e integración RC; añadir casos
   RQ dirigidos desde entradas crudas y comparar también después de fallos.
4. Sintetizar el monitor completo con la misma parte, RuntimeOptimized,
   4 ns e I/O cero; auditar DCP y reproducir su netlist propia contra el
   banco binario y todos los nuevos casos binarios. Registrar regresiones.

Sin pausas, menor capacidad ni cambios de latencia pública, cancelación
cold reset, secuencia o vacío inmediato/eventual. No combinar inicialmente
este cambio con otra aritmética RC o contador registrado. Identidad/BDF,
CDC/reset nativo de 50 ms, DIMM exacto, cuatro DDR/512 GiB, pesos reales,
ECAD y cierre físico siguen siendo dependencias separadas de la
[cola completa](VIVADO_QUEUE.md).
