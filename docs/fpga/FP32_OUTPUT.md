# Conversión de salida FP64 a FP32

Actualizado el 2026-10-03 UTC. `fpga/rtl/ntrpu_fp64_to_fp32.sv` es un
conversor escalar comprobado para la salida down de [la FFN](FFN_MEMORY.md).
La [integración al escritor AXI](FFN_FP32.md) ya pasa bancos numéricos y de
errores; allí se registra la revisión vigente. Las métricas siguientes
corresponden exclusivamente al conversor aislado.

## Contrato

- Entrada `s_data[63:0]`, salida `m_data[31:0]`, handshake valid/ready.
  Índice de 14 bits, last, sequence y epoch acompañan cada resultado.
- Redondeo al más cercano con empates al par; conserva ceros con signo
  y subnormales FP32. No usa operadores HLS que eliminen subnormales.
- Estado 0 correcto; 2 entrada NaN/infinita; 3 magnitud mayor que FLT_MAX.
  Rechaza incluso valores que una conversión ordinaria podría redondear
  hacia FLT_MAX. La salida de error es cero y no debe publicarse.
- Tres etapas, latencia de tres ciclos sin pausas y aceptación de un dato
  por ciclo. Cuando la salida se bloquea, se detienen todas las etapas;
  datos, estado y metadatos permanecen estables.
- Reset descarta los datos en vuelo. El propietario del vector debe
  invalidar su destino privado; el conversor no cancela ni confirma AXI.

## Evidencia

Vivado 2026.1, `xcvu5p-flva2104-2-i`, reloj supuesto de 5 ns. **51.928
vectores × tres semillas = 155.784 conversiones** pasan en RTL y otras
155.784 sobre netlist funcional de síntesis, sin SDF. Incluyen cada exponente
binary64, ambos signos, empates y vecinos, límites subnormales, aleatorios,
NaN/infinito, fuera de rango y las 768 salidas down reales de Brandon.
Cada semilla tiene 33.964 conversiones válidas, 20 no finitas y 17.944
fuera de rango; cuatro resets ejercitan cada etapa y una salida bloqueada.
Python `struct.pack` aporta el redondeo nativo independiente del RTL/HLS.
El revisor rechaza 13 alteraciones de evidencia.

Las semillas 1/7/42 terminan en 65.176/65.235/65.066 ciclos de banco,
con latencias observadas de 3–9/3–10/3–9 ciclos al introducir pausas.
La síntesis utiliza **376 LUT, 382 FF, cero DSP/BRAM**; WNS estimado
**+2,755 ns**, WHS **+0,083 ns**. Faltan retardos de 146 entradas y 115
salidas y el reloj padre. No se ha ruteado ni calificado en placa.
La eliminación de bits 52:25 del cociente desplazado solo recorta bits
no consumidos; la netlist conserva los resultados comprobados.
Los avisos de visor para arrays grandes no afectan las comparaciones.
La definición duplicada de `glbl` se comprobó idéntica.

Tandas en `artifacts/ntrpu-board-vivado/`:

| Run | Resultado |
| --- | --- |
| `20261003T100601Z-V7-fp32-converter-unit` | Fallo del banco al tratar EOF como registro; tiempo simulado cero. Preservado. |
| `20261003T100601Z-V7-fp32-converter-synth` | Síntesis, 85,272 s; checkpoint para replay. |
| `20261003T100710Z-V7-fp32-converter-unit-r2` | RTL PASS, 77,335 s; corrección solo del lector de fixture. |
| `20261003T101108Z-V7-fp32-converter-netlist` | Replay PASS, 93,216 s; mismo RTL sintetizado. |

SHA-256 del RTL: `8f44acd780d6ea5d6b64196de21ced4d87285d44a684340ae2dcb44b3fc60bee`.
Checkpoint: `95475649ca0a14ccf6b19aeb9fdb0e1ec55ebb6f91b4a566027b85f392408ddd`.

**Errata del fixture inicial:** Windows escribió CRLF mientras el manifiesto
calculó SHA-256 sobre LF. El archivo físico se verificó por separado con
`c71e5c3c00a07b7aa49cb34e88cc3448feb32f77d5101eca6dcef1bf77dbb4a1`.
La representación LF tiene
`eb04850b7dc1709b2a0ad837cb5771200a5e1bd4143113164cd269b183480272`.
El revisor admite exclusivamente esas dos identidades fijadas; el productor
vigente escribe bytes LF y se comprobó la igualdad de los 51.928 vectores.
Los respaldos originales no se reescriben.

## Reproducir

```bash
python tools/prepare_ntrpu_fp32_vectors.py NEW_FIXTURE --brandon "$FFN_FIXTURE"
vivado -mode batch -source tools/ntrpu_board_fp32_converter.tcl \
  -tclargs sim NEW_FIXTURE NEW_SIM
vivado -mode batch -source tools/ntrpu_board_fp32_converter.tcl \
  -tclargs synth NEW_FIXTURE NEW_SYNTH
vivado -mode batch -source tools/ntrpu_board_fp32_converter.tcl \
  -tclargs netlist NEW_SYNTH NEW_NETLIST
python tools/check_ntrpu_fp32_converter.py NEW_FIXTURE NEW_SIM/seed-1.log
```

## Contrato de integración

El escritor integrado empaqueta **16 FP32 por beat**, con máscaras de cuatro
bytes para la última fila, y valida el espacio físico redondeado a 64
bytes. La comprobación de finitud se hace sobre cada resultado original;
no interpreta parejas FP32 como un binary64. Gate/up conservan binary64.

El commit espera el último resultado convertido, aunque la completion
de cálculo llegue antes. Se captura el primer error de conversión, se drena
el productor y se prohíbe AW/W antes de validar todo el grupo. Solo el éxito
de las respuestas B y de la raíz permite publicar. Los conteos de bytes y
spans propagan el formato hasta el escritor; backpressure, cancelación,
reset y cruces de página están comprobados. La evidencia de integración
y de las cadenas completas se mantiene en [FFN_FP32](FFN_FP32.md).
