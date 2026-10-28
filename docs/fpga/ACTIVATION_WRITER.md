# Escritura de registros de activación por AXI

Estado al 2026-10-03 UTC: **159/159 casos RTL y 159/159 sobre netlist**,
VU5P, Vivado 2026.1. `fpga/rtl/ntrpu_activation_writer.sv` escribe un
registro [ABI1](ACTIVATION_MEMORY.md) de 64 bytes. Es independiente del
escritor de resultados binary64; ambos conservan sus contratos.

## Comando y publicación

El productor entrega el registro completo de 512 bits junto con dirección,
base/longitud, 1–32 elementos, secuencia y época. Todos los campos quedan
registrados en `cmd_valid && cmd_ready`; el productor puede cambiarlos
después. Para conectar HLS, debe reunir sus ocho palabras y verificar
`ap_done/ap_return` antes de ofrecer el comando.

Antes de emitir AW/W, el módulo verifica alineación a 64 bytes, sumas de
65 bits, pertenencia del registro completo a la región y época esperada.
Comprueba la escala ABI1, los 24 bytes reservados y los INT8 posteriores
al número de elementos, que deben ser cero. Preserva todos los demás
bytes INT8, incluidos patrones que parecen NaN al interpretarlos como FP64.
Se aceptan escala cero negativa, subnormales y el límite `float_max/127`.

Emite un beat AXI de 512 bits, `AWLEN=0`, `AWSIZE=6`, `AWBURST=INCR`,
`AWID=0`, `AWCACHE=2`, `AWPROT=0`, `AWLOCK=0`, `WSTRB=all ones` y
`WLAST=1`. AW y W avanzan independientemente; la completion espera B
después de ambas transferencias. `done_bytes` cuenta bytes W, no persistencia.
Solo estado cero permite publicar el destino privado.

| Estado | Significado |
| --- | --- |
| 0 | Escritura terminada con B correcta |
| 1 / 2 | Comando/rango inválido / época incorrecta |
| 3 / 4 | BRESP/BID incorrectos / registro ABI1 inválido |
| 7 / 8 | Cancelación / watchdog |

La cancelación previa a la emisión evita toda escritura. Durante SEND,
cancelación y watchdog retienen AW/W hasta completar las transferencias y B;
pueden quedar bytes privados modificados. El primer error se conserva.
El watchdog predeterminado es de 65.536 ciclos sin progreso: no recupera
por sí solo un peer detenido. Resetear exige resetear o aislar también AXI
y descartar el destino; el job interrumpido no emite una completion.

## Evidencia y límites

Cada banco ejecuta 53 escenarios con semillas 1/7/42: formas parciales,
direcciones de 640 GiB y próximas al límite de 64 bits, descriptores
alterados tras aceptación, formatos inválidos, bytes INT8 crudos, AW/W
en ambos órdenes, errores B/BID, timeout, nueve cancelaciones y cinco
fases de reset por semilla. Se verifican registros completos y guardas;
Python recalcula los bytes independientemente y rechaza once alteraciones
del log. La netlist usa el watchdog real, sin SDF; RTL usa 32 ciclos.

- `20261003T074923Z-V7-activation-writer-unit`: PASS, 49,437 s.
- `20261003T074923Z-V7-activation-writer-synth`: 87,148 s, **385 LUT,
  932 FF**, cero DSP/BRAM. WNS estimado +3,422 ns, WHS +0,066 ns a 5 ns.
- `20261003T075418Z-V7-activation-writer-netlist`: PASS, 107,629 s.

Los avisos de salidas constantes corresponden a los atributos AXI fijos.
Se conserva el aviso de jerarquía plana; los dos módulos `glbl` del replay
son idénticos al eliminar comentarios/espacios. No hay endpoints internos
sin restricción; faltan retardos de 819 entradas/651 salidas y reloj padre.
No se afirma ruteo, conexión HLS/FFN ni funcionamiento DDR/PCIe.
Los tres respaldos locales incluyen fuentes, revisión y hashes verificados.

## Reproducción

Con Vivado cargado, desde la raíz y usando carpetas nuevas:

```bash
vivado -mode batch -source tools/ntrpu_board_activation_writer.tcl \
  -tclargs sim NEW_SIM
vivado -mode batch -source tools/ntrpu_board_activation_writer.tcl \
  -tclargs synth NEW_SYNTH
vivado -mode batch -source tools/ntrpu_board_activation_writer_netlist.tcl \
  -tclargs NEW_SYNTH NEW_NETLIST
```

El [adaptador SwiGLU](SWIGLU_MEMORY.md) tiene validación separada de sus
lecturas, HLS, registros escritos y recuperación; no hereda automáticamente
la aprobación del escritor. Véase el [registro](board/VIVADO_STATUS.md).
