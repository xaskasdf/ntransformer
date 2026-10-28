# Monitor PCIe: medir el camino físico completo

Estado: **pendiente**, condicionado a aprobar la netlist propia del monitor
R1. La [síntesis conjunta](PCIE_SYMBOLIC_MONITOR_R1_SYNTHESIS.md) registra
104259 LUT, 26976 FF y WNS −3,584 ns a 4 ns. El peor camino va de
`delivery_stage/out_b_reg[origin][record][rc_user][64]` a
`retirement_i/fault_latched_reg[10]`: 31 niveles, 1,538 ns de lógica y
6,028 ns de interconexión **estimada**. Aún no demuestra ese retardo ruteado.

## Dependencias y tandas

1. Aprobar y respaldar la comparación de la netlist sintetizada: 174436
   filas binarias, 447 bits públicos, ambas fases, diez sondas de quiet y
   seis controles de entradas. Conservar la regresión RTL X/Z de 183058
   filas y los dieciséis mutantes del contenedor.
2. Abrir exactamente el DCP revisado. Crear una referencia optimizada y,
   desde el mismo DCP original, una variante con `CLOCK_BUFFER_TYPE BUFGCE`
   en la red de `clk`. Usar el flujo ya registrado en
   `tools/ntrpu_board_pcie_inventory_clock_preflight.tcl`, adaptado al top
   del monitor, sin asumir que su resultado anterior se transfiere.
3. Verificar los puertos, primitivas y registros de ambas variantes. El
   único BUFGCE debe recibir `clk`, tener CE constante y alimentar todos
   los pines de reloj. Exportar DCP, EDIF, netlist, XDC y reportes; revisar
   todas las diferencias y ejecutar la netlist optimizada propia antes
   de permitir colocación/ruteo.
4. Colocar y rutear esa variante con reloj interno real, siguiendo el
   precedente `ntrpu_board_pcie_inventory_clock_placement_r1.tcl`. Mantener
   4 ns, incertidumbre heredada, I/O cero, sin asignación artificial de
   `HD.CLK_SRC`, excepciones, degradaciones DRC ni pines de placa.
   Empezar con dos hilos y límite explícito por tanda; conservar también
   resultados parciales o fallidos.
5. Leer de nuevo el DCP exportado y contrastar inventario, conectividad,
   rutas, clocks, cobertura de restricciones, setup/hold, pulsos, DRC y
   metodología. Reproducir su netlist funcional propia; cualquier
   simulación SDF constituye una validación adicional separada.

## Decisión de arquitectura

Comparar rutas internas registradas y rutas de frontera por separado.
Si persiste la cadena de payload de 31 niveles, estudiar un prefijo
paralelo de estado por lane o una partición registrada explícita. No
inferir una mejora a partir del número de etapas ni de un módulo aislado.
Medir recursos, latencia y temporización de cada candidato.

Preservar dos RQ y cuatro RC por ciclo, los 256 tags, contexto retenido
por valor, RC antes de RQ, contabilidad completa y publicación alineada.
Una etapa añadida debe incorporarse a la historia del oráculo y a las
obligaciones pendientes de quiet. Solo el reset frío cancela trabajo.
No eliminar tráfico malformado binario ni relajar las pruebas X/Z.

El resultado será caracterización OOC del monitor. La integración con
el reloj/endpoint PCIe real, CDC, reset, recepción posted, cuatro DDR,
ECAD y pruebas físicas de la tarjeta seguirán requiriendo sus propios
criterios. Respaldar y verificar cada tanda en D: y NFS.
