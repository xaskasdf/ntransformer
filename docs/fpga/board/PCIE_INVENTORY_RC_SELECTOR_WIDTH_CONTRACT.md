# Contrato: anchura del selector RC

R9 parte de R8 aprobado en RTL y en su propia netlist. El único cambio
funcional de código es declarar `newest` como `reg signed [4:0]` en el
selector R4, en lugar del `integer` de 32 bits del selector R3.
El productor R9 conserva R8, salvo un comentario de identificación.

## Invariante y alcance

`newest` se inicializa a −1 y sólo recibe posiciones EOP conocidas entre
0 y 15. `event_lane` se inicializa a cero y recibe el índice constante del
bucle de lanes cuando hay un EOP efectivo; punteros desconocidos no escriben
en el mapa procedural. La variable no necesita representar otros valores.

La comparación debe mantener el centinela negativo, la promoción de signos
y la prioridad de la última escritura. Este argumento delimita el cambio;
no sustituye las pruebas de cuatro estados y de la netlist.
No cambiar máscaras, propiedad, aritmética, RQ, capacidad ni latencia.

## Validación y métricas

- Conservar las 1.153.296 comparaciones heredadas, incluidas 994.288 X/Z,
  y las 32.768 adicionales, incluidas 32.256 X/Z. Comparar los 840 bits
  completos con el selector R1 y la referencia secuencial de propiedad.
- Ejecutar selector 89.632; RQ 615.536/5.712 X/Z y 8.850/240; aritmética
  RC 45.056/6.624 X/Z y 544/60; propiedad integrada 46.356/3.872;
  principal 22.339/68, siete mutantes y todos los controles de evidencia.
- Verificar tablas y estados posteriores a fallos. Exigir síntesis
  `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, 4 ns y DCP acíclico.
- Reproducir con la propia netlist las 22.330/67 filas binarias principales
  y los tres suplementos completos. No compilar RTL del DUT en ese replay.
- Comparar LUT, FF, CARRY8, WNS/TNS, endpoints y ruta crítica con R8 y R7.
  La reducción de anchura no garantiza menor área o mejor timing.

Mantener el inicio XSIM sin captura automática de ondas, todos los casos
y los logs. Registrar tiempos y tamaños reales; no atribuir diferencias
a una sola causa cuando cambie también el diseño.

## Conservación y dependencias

Guardar fuentes nuevas e inmutables, revisiones y respaldos local/NFS por
tanda, incluidos fallos. Conservar las variantes anteriores y las tareas
ajenas. El [seguimiento R8](PCIE_INVENTORY_RC_EVENT_ANCESTRY_FOLLOWUP.md)
separa los experimentos posteriores de máscaras y aritmética.

Continúan pendientes los criterios nativos, DDR de 512 GiB, inferencia
real y diseño físico/ECAD de la [cola completa](VIVADO_QUEUE.md).
