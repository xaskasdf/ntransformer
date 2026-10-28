# Bandera del último evento de contexto

## Atribución de la ruta mapeada

El cono `delivery_stage/retirement_i/late_i/cp2379_in` de la tanda 966
contiene 150 nodos y termina en los 32 FF `B.rc_user[95:64]`. Se modelaron
sus LUT mediante los INIT archivados y su CARRY8 mediante la semántica
binaria `SINGLE_CY8` de la biblioteca instalada.

Z3 demostró que la salida equivale a **`last_eop >= last_sop` antes del
lane 13**, para todas las asignaciones binarias de esos 32 FF, sin asumir
tráfico válido. Ambos índices empiezan en −1. Tres controles —invertir,
forzar cero y forzar uno— produjeron contraejemplos. La prueba no atribuye
el segundo CARRY8 de la ruta ni demuestra equivalencia secuencial o X/Z.

## Cambio aislado

El RTL original usa `last_eop >= 0 && last_eop >= last_sop`. Se propone
sustituir solamente ese predicado por una bandera combinacional:

```systemverilog
context_last_is_eop=0;
// Recorrer únicamente los lanes anteriores al actual.
if (aligned_sop) context_last_is_eop=0;
if (eop)         context_last_is_eop=1;
```

EOP conserva la prioridad si ambos eventos coinciden. Se mantienen los
`if` procedurales, la selección de cabeceras, los contadores y todos los
predicados de fallo. No se añaden registros ni ciclos. Una segunda prueba
formal cubrió las 1.048.576 máscaras de eventos en los 17 límites de lane;
las 17 desigualdades fueron UNSAT y tres controles negativos fueron SAT.
Los eventos booleanos representan ramas efectivamente tomadas; falta
comprobar la implementación RTL con X/Z.

## Validación y dependencias

Después del replay aprobado de la tanda 967 se ejecuta el banco aislado:
47.734 muestras, 32 vectores de fallo por muestra antes del OR, 10.240 fronteras,
17.352 muestras de desconocidos y seis mutantes. Compara contra el padre
por prefijo aprobado y la salida agregada de la validación original.
Solo un resultado revisado y respaldado habilita la integración completa
de 183.058 filas; luego hacen falta síntesis y replay propios. No hay
mejora de timing medida para esta bandera. El borrador de límites
registrados en B permanece separado y sin calificar.

## Evidencia reproducible

En `build/cloud-setup/20261002-board/`:

- Cono extraído R3: `45113a749d7a9fc97964ebe8f6c67237880a438ac26571154eaf6afbb6c6ee84`.
- Prueba de atribución: `aa1ad899fa03fcafed4447594cde487143d1885f9ae6ba054576055cf50f3622`.
- Prueba de bandera: `258668c19e2c3ddd3d4b338abd5d678a56f9d19194022860e50f98a2540eeb19`.
- RTL aislado: `b20635a2410d5c277c4532409c4edef75937445f0027dd36a7c8563864cd5d9d`.

Z3 5.1.0 usa un entorno local aislado. Su wheel y procedencia quedaron
verificados en D: y NFS en `symbolic-monitor-predecode-z3-toolchain-r1-20261011T140152Z.zip`,
SHA-256 `8aee87f9b401ac0639ce6e7de5c82f389738cd37586ca4cb38c8ca20b5ee6807`.
