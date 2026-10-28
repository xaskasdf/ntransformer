# Control de la composición FFN → residual

Actualización: 2026-10-15, 18:17 UTC. **Wrapper R3 aprobado en simulación
nativa de protocolo**, con la aritmética sustituida por peers explícitos.
La composición aritmética completa está [encolada](LLAMA32_FFN_RESIDUAL_REAL_R1.md)
desde las 18:35 UTC y todavía no está cualificada.

## Contrato implementado

[`ntrpu_llama32_ffn_residual_r3.sv`](../../fpga/rtl/ntrpu_llama32_ffn_residual_r3.sv)
conecta el propietario FFN, el [controlador de traspaso](LLAMA32_FFN_RESIDUAL_HANDOFF_R1.md)
y el [residual real](LLAMA32_RESIDUAL_REAL_R1.md). Captura identidad de
64 bits, ocho regiones FFN, residual retenido y destino final. Antes de
iniciar FFN comprueba alineación, extremos de 65 bits y ausencia de
solapamiento del destino con las nueve asignaciones reservadas.

El controlador cede los cinco canales AXI al residual después del drenaje
FFN. Las respuestas R/B sólo se habilitan para su propietario. FFN y
residual mantienen sus respuestas hasta el ACK público final; la admisión
siguiente espera su liberación. Una cancelación o cambio de época revoca
el resultado inmediatamente y de forma persistente. El estado de respuesta
publicado permanece estable mientras espera ACK.

## Evidencia y límites

Banco: [`tb_ntrpu_llama32_ffn_residual_control_r9.sv`](../../fpga/tb/tb_ntrpu_llama32_ffn_residual_control_r9.sv).
Revisor: [`check_ntrpu_ffn_residual_control_r1.py`](../../tools/check_ntrpu_ffn_residual_control_r1.py).

| Medida | Resultado agregado, semillas 1/7/42 |
| --- | --- |
| Trabajos / respuestas / resets | 288 / 276 / 18 |
| Descriptores ofrecidos / aceptados sin revocación | 165 / 156 |
| Beats leídos / escritos confirmados por B | 761 / 372 |
| Alteraciones de trazas rechazadas | 186 |
| Mutantes RTL compilados y detectados por aserción | 20 |
| Ciclos de cada banco | 3.809 / 3.820 / 3.820 |

Se prueban los extremos de las nueve regiones, adyacencia válida,
desbordamientos, identidades altas, backpressure, cancelaciones, cambios
de época, errores de ambos peers, reset con transacciones pendientes,
revocación tardía y órdenes adicionales mientras el dispositivo está ocupado.
El revisor reconstruye los datos de prueba y exige que la lectura del
residual provenga de la escritura FFN confirmada por B.

Los peers ejecutan dos lecturas y una escritura de 64 bytes cada uno.
Sus descriptores de 8.192 bytes representan el contrato de control;
**estas cuentas no miden la aritmética, latencia ni throughput de inferencia**.
El archivo `ntrpu_ffn_residual_control_peers_r1.sv` debe quedar fuera del
paquete aritmético completo.

## Correcciones conservadas

- R1–R3: valores indefinidos en la captura. Cambiar sólo los registros
  internos no resolvió el problema; los drivers empaquetados y los aliases
  de tipo `wire` del banco R4 permitieron avanzar.
- R4 detectó un defecto del wrapper: una época revocada durante FFN
  llegaba al traspaso como cancelación. RTL R3 conserva la causa original
  y registra el estado público antes de exponer VALID.
- R5–R8 conservaron fallos del banco, incluido `starts=x`. Los contadores
  empaquetados por sí solos no bastaron. R9 utiliza sumas explícitas para
  los siete incrementos indexados; conserva las aserciones.

Todos los intentos están respaldados individualmente en D: y NFS. Un
timeout no cuenta como detección de mutante. Ninguna fuente ni tanda
anterior se modificó.

| Evidencia | Identificador / SHA-256 |
| --- | --- |
| Baseline | `20261015T181444Z-V7-ffn-residual-control-native-r9` |
| Archivo baseline | `746fb3affbf2bb7bf2369e076d8c31c813c947f20e5a879315c5b341ee9945f6` |
| Mutantes | `20261015T181617Z-V7-ffn-residual-control-mutants-r1` |
| Archivo mutantes | `a99d2fc525940b935ffa2ee39d28f9ba98b0d84f6e46f6a967e2a434cd7883b7` |
| Revisión independiente | `f83430c80ba82fe4a2874377e1bad9716934168b30317f9dbaf6632128194dd2` |
| RTL R3 | `cc2ae32db00bdba218d46cff43ebd9392e5049749b855900fde58b502d4f67c4` |

## Siguiente puerta de validación

Avance 2026-10-15 18:35 UTC: el [banco aritmético completo R1](LLAMA32_FFN_RESIDUAL_REAL_R1.md)
ya está preparado, revisado en sintaxis/interfaces y encolado con 68
fuentes reales detrás del traspaso R1. La simulación aritmética integrada
continúa pendiente; esta cualificación nativa conserva su alcance de control.

Extender el banco FFN completo manteniendo los 53.477.376 bytes Q8
originales y su entrada ABI1 post-O. Añadir el residual retenido y el
destino final; leer la salida FFN desde los bytes realmente escritos,
comparando con `job-001-layer-output-fp32.bin`. El tráfico adicional
esperado es 16.384 bytes leídos y 8.192 escritos. No precargar resultados
esperados en la memoria accesible por el DUT.

El paquete previsto contiene 68 fuentes aritméticas efectivas: las 51
del banco FFN, ocho módulos de integración/residual adicionales y nueve
Verilog HLS residuales; sustituye el TB anterior por el nuevo. Verificar
también los once archivos de simulación IP generados por Vivado.
Encolar síntesis OOC y XSim detrás del traspaso R1, después de completar
el banco, revisor y sellado. Exigir evidencia y respaldo entre etapas.

Auditor R39 a las 18:17: seis coordinadores vivos; FFN R4 sigue en XSim,
fase 1. No se reinició ni se interrumpió. Continúan pendientes productor
post-O vivo, inferencia completa, timing integrado, DIMM exacto y cierre
DDR4/PCIe físico.
