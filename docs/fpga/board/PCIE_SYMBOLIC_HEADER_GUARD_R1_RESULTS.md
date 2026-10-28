# Predicados de cabecera: temporal de 15 bits aprobado

Prueba focalizada del 2026-10-09: **1.085.445 casos comparados**.
La nueva unidad conserva `available` y `delivered` signed de 15 bits,
la comparación mixta original y los predicados de longitud/terminal.
Su referencia contiene literalmente el bloque aritmético de R7.

## Cobertura y controles

- 1.048.576 combinaciones de ocho extremos de remaining, cuatro
  alineaciones y los 2.048 counts; normal y terminal recorren cuatro estados.
- 36.864 inyecciones individuales X/Z en los 36 bits de remaining,
  dirección y count, con ocho bases y todas las combinaciones de cuatro
  estados de normal, propiedad y terminal.
- Cinco testigos iniciales detectan módulos realmente alterados:
  ignore_enable en paso 0, omit_terminal en 1, signed_compare en 2,
  count_upper en 3 y zero_extend en 4.

Se compararon 32 bits: ambos temporales y los dos predicados. El testigo
de zero_extend detecta la pérdida de los bits altos del temporal completo;
no demuestra por sí solo una diferencia de los predicados finales.
Ocho corrupciones de evidencia fueron rechazadas. Las dos fuentes SV
compiladas en cada caso coinciden con su manifiesto; solo aparecieron
las seis advertencias de entorno XSIM previstas.

## Integración preparada

El nuevo validador recibe el registro B capturado y usa cuatro instancias
de esta unidad. Conserva el recorrido de payload del validador anterior.
Un probe derivado de R7 agrega exclusivamente observaciones de fallos
tardíos y temporales completos; sus predicados y estado se conservan.
El banco mantiene las comparaciones R7/S1/C/E/D/A/B originales y añade
16 bits de fallos y 60 bits de cabeceras por origen, con cuatro flancos
de vaciado. **Esta integración todavía no está validada.**

La siguiente tanda recorrerá 178.185 filas de los catorce bancos existentes.
Cada mutación que silencia un bit tardío usa determinísticamente el primer
banco positivo con cobertura de ese bit; la ausencia de cobertura falla.
No se modifica ni filtra ninguna fila por sus resultados.

- Tanda: `20261009T165242Z-V4-ffn-supervised-pcie-symbolic-header-guard-r1`.
- Archivo D:/NFS: SHA-256 `263dae834433d50f09c2dc0054063bba862009ffbc5b24b42baf3d7b827ea296`.
- Revisión: `20261009T165242Z-V4-ffn-supervised-pcie-symbolic-header-guard-r1-independent-symbolic-header-guard-review.json`, SHA-256 `e1f2322516ffd985d7bf76ae5b650e921e2026f8db5d0870ea93c59e59674288`.

Faltan la validación completa, retiro y fallo persistente, quiet, síntesis
y replay propios de las nuevas etapas, monitor completo y cierre físico.
Se mantienen las dependencias de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
