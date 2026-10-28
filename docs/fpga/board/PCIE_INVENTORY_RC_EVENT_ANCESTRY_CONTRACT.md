# Contrato: ascendencia RC por EOP efectivos

El candidato R8 conserva el productor R7 y cambia únicamente el selector
por R3. Requiere los resultados aprobados de R7, incluida su propia netlist.
El [seguimiento de R7](PCIE_INVENTORY_RC_OWNERSHIP_FOLLOWUP.md) y
la [cola general](VIVADO_QUEUE.md) siguen vigentes.

## Cambio y equivalencia

Cada EOP efectivo obtiene su contexto del último SOP anterior o simultáneo,
o del contexto retenido. Los punteros duplicados conservan el último slot.
Sólo el primer EOP de cada contexto puede escribir; propiedad, terminal y
normal mantienen su semántica procedural de cuatro estados.

Para cada encabezado se selecciona la última escritura anterior del mismo
tag. Su máscara reúne los orígenes de esas escrituras y su raíz retenida.
No se propagan máscaras entre escrituras. La selección de commit conserva
el último EOP válido por tag. No cambian capacidad, aritmética ni latencia.

## Puertas de validación

- Mantener el banco original de 155.168 entradas binarias y las 1.153.296
  comparaciones, incluidas 994.288 X/Z, contra el selector secuencial R1 y
  la referencia independiente de propiedad.
- Añadir 32.768 combinaciones de seis controles en ocho contextos,
  incluidas 32.256 X/Z: contexto retenido, varios EOP sin SOP intermedio,
  punteros EOP duplicados y discontinue. Comparar los 840 bits de salida.
- Conservar selector 89.632, RC 45.056/6.624 X/Z y 544/60, RQ
  615.536/5.712 X/Z y 8.850/240, propiedad integrada 46.356/3.872, banco
  completo 22.339/68 y sus siete mutantes. Revisar estados posteriores a fallos.
- Sintetizar VU5P con período de 4 ns y `RuntimeOptimized`; auditar el DCP
  y repetir 22.330/67 y los tres suplementos binarios con su propia netlist.
- Comparar área, WNS/TNS, endpoints y ruta crítica frente a R7 y anteriores.
  Ningún resultado OOC certifica cierre físico, DDR de 512 GiB o endpoint nativo.

## Ejecución y respaldo

Evaluar el inicio XSIM sin captura automática de ondas mediante Tcl propio,
conservando todos los casos, `run all`, logs y controles. Registrar su
ejecución real y el tamaño WDB; no atribuir causalmente una aceleración a
esa opción al comparar también un diseño diferente.

Usar copias inmutables, nuevas carpetas por ejecución, revisión independiente
y respaldo local/NFS por tanda. Conservar las ejecuciones ajenas y las fallidas.
