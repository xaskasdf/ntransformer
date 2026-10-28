# Estudio físico de la raíz de atención y O — R1

Estado 2026-10-15: **medición terminada y respaldada; timing interno fallido**.
Comenzó como `20261015T090048Z-V7-llama32-attention-projected-root-route-vivado-r1`
tras la revisión y respaldo de [post-O real](LLAMA32_POST_ATTENTION_REAL_R1.md).
No repite ni modifica la simulación funcional aprobada de la raíz.

## Resultado y revisión R2

Las 86.884 conexiones ruteables quedaron completas, sin errores de ruta.
A 5 ns: **WNS −0,079 ns**, WHS +0,010 ns y WPWS +1,958 ns.
Recursos: 48.175 LUT, 44.887 FF, 84,5 BRAM y 134 DSP. No hay cajas
negras, latches, lazos ni extremos internos sin restricción. Permanecen
3.409 entradas y 1.473 salidas externas sin presupuesto temporal de placa.
El aviso crítico de Vivado indica precisamente incumplimiento de timing.

Ruta crítica: `quant_owner/completion_epoch_reg[25]` →
`quant_owner/output_mem_reg/ENARDEN`, bajo `output_path/quantize_store/`.
Tiene 18 niveles lógicos y 4,633 ns de datos: 1,444 ns lógicos y 3,189 ns
de conexiones. Se conserva como punto de partida de optimización.

La revisión R1 midió correctamente ese resultado, pero su control negativo
de enable no reconoció un identificador Verilog escapado con un espacio
antes del salto de línea. R2 corrige únicamente esa búsqueda; conserva
las métricas y rechaza 19 controles. No se repitió Vivado ni se relajaron
criterios. El BUFG de datos adicional alimenta 1.031 pines CE de registros,
con enable constante y paso asíncrono, según la inspección estructural.

| Evidencia | SHA-256 |
| --- | --- |
| Archivo original D:/NFS, 49 miembros | `b341f6c7a0c6b92f6ec302e72d5190bcebaf0f86b3bcf040176123a600a979a7` |
| Revisión R2 | `3e33f29f8ee00005031503c0628d1fb7673f7ef56d479880a3bd60da4542bb1f` |
| Suplemento R2 D:/NFS | `99ee8b7a68bfe1bf517d9aa13ab132ff6e73310ac8a91fc94bca075af83bd033` |
| Checkpoint ruteado | `05c0cd78f012aa1ba811ff1f64a52bcef3fcc0ec98b7cee10f33d33a99e1da51` |

El siguiente estudio aplica `phys_opt_design -directive Explore` al
checkpoint ruteado, conservando reloj, ubicación de BUFG y restricciones.
Es una tanda separada, detrás de [la unión real con post-O](LLAMA32_ATTENTION_POST_REAL_R1.md).

## Motivo y checkpoint

La [raíz → cuantización → O](LLAMA32_PROJECTED_ROOT_REVIEW_R4.md) entregó
4.096 FP64 y 16.384 bytes FP32 exactos, con cancelación/recuperación.
Su síntesis a 5 ns tiene WNS −0,203 ns. El camino crítico observado va desde
`output_path/quantize_store/quant_owner/completion_epoch_reg[9]` hasta
`output_path/completion_status_reg[0]/CE`: 21 niveles, 5,099 ns de datos,
con 3,648 ns de conexiones estimadas antes de colocación. Atraviesa
identidad, cancelación y finalización entre controladores.

El estudio coloca y rutea ese mismo checkpoint antes de decidir cambios
en la cadena de control. Fuente funcional original:
`20261015T064819Z-V7-llama32-attention-projected-root-real-vivado-r3`.
Checkpoint SHA-256:
`658d6c66a2cb3eb59d7fe260e0ac0974d716a3e2c53f3e28503a55f58a8fa5dd`.
La revisión funcional R4 está en un suplemento verificado; se comprueban
tanto ese suplemento como el archivo original y el checkpoint extraído.

## Contexto físico y límites

Parte `xcvu5p-flva2104-2-i`, periodo 5 ns. Se inserta un BUFGCE de reloj
habilitado permanentemente en `BUFGCE_X0Y60`, comprobando que conserva
todos los consumidores. Su ubicación es provisional. No se agregan
excepciones temporales, cambios de severidad ni dispensas DRC.

Los puertos externos siguen sin presupuesto temporal de placa; el ensayo
evalúa caminos internos. No integra DDR4/PCIe, no aprueba pinout ni ECAD
y no sustituye una simulación posterior a ruteo. Los buffers de datos que
pueda insertar Vivado requieren revisión explícita: paso asíncrono con
habilitación constante, usos completos y cargas CE de registros.
Esta inspección estructural no prueba equivalencia lógica entre síntesis
y ruteo: cambiar el número de consumidores puede conservar una estructura
válida. La validación funcional posterior a implementación sigue pendiente.

## Ejecución y criterios

Secuencia: abrir checkpoint, comprobar reloj/fuentes, `opt_design`,
`place_design -directive Explore`, `phys_opt_design -directive Explore`,
`route_design -directive Explore`, guardar y reabrir checkpoint.
Límite de una hora, dos CPU y una sola tanda principal de Vivado.

Se revisan rutas completas y sin errores, setup/hold/pulsos, cobertura
interna, reloj, DRC, metodología, recursos y restricciones antes/después.
La revisión distingue **medición terminada** de **timing interno aprobado**;
un slack negativo se conserva como fallo medido. Los controles negativos
incluyen checkpoint, reloj, excepciones, rutas y semántica de buffers.
Cada resultado o diagnóstico debe respaldarse en D:/NFS antes de liberar
el turno. No hay criterio que permita declarar aprobada la placa aquí.

| Paquete sellado | SHA-256 |
| --- | --- |
| Fuentes/checkpoint D:/NFS | `fb712a87c4cf421698727ac5e79317519feaeed4b66e218d5bb6ccbdcfff8657` |
| Configuración R1 | `25c06f21cd24422bb9e4470c8b67e160dc03b5654da138669afbf01c49ab237e` |

El coordinador original conserva su diagnóstico. La revisión recuperada
está en `llama32-attention-projected-root-route-review-r2-coordinator.json`.
Las validaciones externas permanecen preservadas.
