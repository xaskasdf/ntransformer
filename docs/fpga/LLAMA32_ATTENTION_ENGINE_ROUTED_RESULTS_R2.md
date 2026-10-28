# Atención: medición física del motor y control de proyección (R2)

Estado 20261013T214349Z: **timing interno OOC aprobado en el checkpoint enrutado**.
No acredita timing de tarjeta ni simulación funcional posterior al enrutado.

## Resultado físico medido

Tanda `20261013T211707Z-V7-llama32-attention-engine-route-vivado-r3`. Se implementó el checkpoint de síntesis ya validado en XSim,
sin modificar RTL, a 5 ns sobre `xcvu5p-flva2104-2-i`, con dos CPU.
Los informes provienen del checkpoint cerrado y vuelto a abrir.

| Medición | Resultado |
|---|---:|
| WNS interno de setup | **+0,195 ns** |
| WHS interno de hold | **+0,015 ns** |
| WPWS | +1,958 ns |
| Redes enrutables / completamente enrutadas | 33.574 / 33.574 |
| Errores de enrutado / huecos explícitos | 0 / 0 |
| LUT / FF | 18.991 / 16.492 |
| BRAM36 equivalentes / DSP | 33,5 / 77 |

Se revisaron 270.950 segmentos de red. No hay cajas negras, latches,
relojes ausentes, extremos internos sin restricciones, bucles ni errores DRC
o advertencias críticas. Se conservaron las restricciones originales:
no se agregaron excepciones, waivers ni cambios de severidad.

El peor setup va de `compute_tokens_reg[5]/C` a `query_done_status_reg[3]/CE`:
18 niveles, retardo de datos 4,625 ns, del cual 2,892 ns es interconexión.
El peor hold está dentro del IP exponencial. El margen de hold de 15 ps
es pequeño y pertenece exclusivamente a este contexto implementado.

## Diagnóstico original y revisión R2

La revisión R1 esperaba exactamente un BUFG y emitió diagnóstico al encontrar
dos. El log de colocación registra que Vivado agregó uno para distribuir
una habilitación a 2.048 consumidores. La revisión R2 verifica el netlist
funcional completo, cuyo hash coincide con el checkpoint respaldado:

- El BUFGCE principal mantiene el reloj `clk` a 5 ns y sitio `BUFGCE_X0Y60`.
- `i___40_bufg_place` usa `CE_TYPE=ASYNC`, CE fijo a 1 y entrada desde LUT2
  `INIT=0x8`; no divide ni habilita dinámicamente otro reloj.
- Sus **2.048 salidas conectadas son exclusivamente FDRE/CE** de `query_mem`,
  cuyos pines C siguen conectados a `clk`. Se enumeran todos los registros
  y usos de la red; no se permite un consumidor adicional u oculto.

Los 14 controles de informes y nueve mutaciones de netlist fueron rechazados.
No se repitieron colocación, optimización, enrutado o simulación. Se conserva
el diagnóstico original; esta revisión es evidencia adicional.

## Control de proyección O

La tanda `20261013T213453Z-V7-llama32-attention-projection-control-vivado-r3` también terminó y se respaldó:
145 escenarios XSim, 83 corrupciones de traza rechazadas; síntesis con
204 LUT, 851 FF, sin BRAM/DSP, WNS +3,176 ns y WHS +0,066 ns.
Esta tanda usa un sustituto del motor matricial y no demuestra la proyección
aritmética completa. La conexión con el GEMV real sigue pendiente.

## Evidencia y límites

- Archivo de implementación: `ce2efec6950970141c7cc62d699e70d547a66514b8c8c53eacac40986c91aa81`.
- Checkpoint enrutado: `9edfc0e3515aa26feefce84c9df54d10de4127d9be161ce0fde9eca215685d11`.
- Netlist: `2452fe20c9f1ca568b35afa7fe166649aece1defdb6f272f68dcc3c0d2eed052`.
- Revisión física R2: `d34c5103e7dc475e00cb5c70120c3d190a7a375cbf234e2e8c8ae3d02b5f98f8`.
- Archivo del controlador O: `d42d32560b2a1786cc674b93e9f982f669eff0da8e56411057acc3ec849f0240`.

Quedan 4.270 entradas y 1.223 salidas sin presupuesto de E/S. El sitio del
reloj es provisional; faltan el contexto padre, DDR/PCIe, ubicación de tarjeta
y simulación funcional del netlist enrutado. El motor aislado cumple timing
interno en este ensayo; **la tarjeta completa a 200 MHz continúa pendiente**.
