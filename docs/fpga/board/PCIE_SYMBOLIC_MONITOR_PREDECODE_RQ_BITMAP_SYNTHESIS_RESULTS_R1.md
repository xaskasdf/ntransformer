# Síntesis del bitmap RQ: mejora de setup estimado

La tanda 984 `20261011T180537Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-rq-bitmap-synth-r1`
terminó en **427,705 s**. Está revisada, comparada con el padre 979 y
respaldada en D: y NFS. La comparación comprobó iguales dispositivo,
reloj de 4 ns, retardos de E/S, estrategia de síntesis y builds de Vivado.

## Medidas comparables

| Medida | Padre RC 979 | Bitmap RQ 984 | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 91.660 | 92.913 | +1.253 |
| FF | 27.532 | 27.560 | +28 |
| CARRY8 | 568 | 568 | 0 |
| WNS, ns | −3,704 | −2,768 | +0,936 |
| TNS, ns | −1.931,080 | −1.472,519 | +458,561 |
| Endpoints con fallo de setup | 1.456 | 1.439 | −17 |
| WHS, ns | −0,075 | −0,075 | 0 |
| Endpoints con fallo de hold | 2.106 | 2.068 | −38 |

Los endpoints totales pasan de 70.907 a 71.157. El coste adicional es
aproximadamente 1,37 % de LUT. Los FF son recursos del mapeo; la variante
no añade una etapa RTL. Setup y hold siguen incumpliendo. Estas cifras
son estimaciones OOC, sin colocación/ruteo ni medida de frecuencia física.

## Ruta crítica y revisión

La peor ruta pasa de la realimentación C→C a B→R:
`delivery_stage/out_b_reg[origin][record][rc_user][64]/C` →
`retirement_i/fault_latched_reg[10]/D`. Tiene 27 niveles, incluidos tres
CARRY8, y 6,750 ns de datos: 1,560 ns lógicos y 5,190 ns de interconexión
estimada. La comparación por etapas medirá separadamente C→C y C→E.

El preflight comprobó 529 filas y 31 pruebas de quietud. Se verificaron
26 unidades de simulación, 14 fuentes sintetizadas, doce categorías de
cobertura de timing y el grafo final. No hay latches, BRAM, URAM ni DSP.
Los avisos emitidos quedaron revisados; dos categorías alcanzan el límite
de cien mensajes del proveedor y no califican los mensajes suprimidos.

## Decisión y respaldo

La mejora permite continuar el [replay de la netlist propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_MAPPED_R1.md).
No se promueve todavía el candidato ni se lanza otro ruteo. Después de
la prueba E encolada, la [comparación RQ/979 por etapas](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_STAGE_TIMING_R1.md)
ayudará a escoger la siguiente modificación sin confundir ambos cambios.

Archivo: **35.251.114 bytes, 184 archivos**, SHA-256
`68833a048627936007defeb81085104a097e17ad11cbe51cc3f2c1e209059747`.
Snapshot `source-20261011T180418Z.zip`:
`eb734605ee4cce97e51615dac35a9005630681dc6c4e96e749c0bcc8f1d53512`.
Revisión: `894faca9f771e685e599f9b186e17b5faea33ac8dc44b0a49514703ec7a5ace8`.
Comparación: `4a5a120c579f5ec132ecb2550899e968627640080ceb74df240ae4323f8e159b`.
Registros en `build/cloud-setup/20261002-board/`; certificados bajo
`artifacts/ntrpu-board-vivado/<tanda>/`.
