# Validación DMA/ACK por contexto en paralelo

## Cambio y contrato

Experimento aislado del 2026-10-08, continuación del
[selector por secuencia](PEER_READ_SELECTOR.md). El camino colocado de esa
variante todavía atravesaba selección y autenticación en serie, con 22 niveles
lógicos y fanout 962 en el enable del resultado. Su ruteo no terminó dentro del
presupuesto; ese diagnóstico motiva una nueva comparación.

`ntrpu_peer_read_lane_parallel.sv` calcula un bit de aceptación DMA y otro ACK
para cada contexto. Cada bit incluye contexto activo, comando emitido, ausencia
de duplicado, identidad completa, destino, generación de ventana, estado cero,
longitud exacta y bytes enviados completos. Un evento válido sin ningún bit de
aceptación genera el mismo fallo global que antes. Las actualizaciones se
aplican directamente al contexto aceptado, sin un multiplexor ancho previo.

No agrega registros, ciclos, buffers ni excepciones de timing. Conserva el
orden de retiro y la retención ante fallos; no convierte DMA en confirmación
remota. La unicidad de secuencias activas, impuesta por admisión, impide que un
evento auténtico corresponda a dos reservas simultáneamente. La equivalencia
exhaustiva de todos los estados no está demostrada formalmente.

El módulo original, el selector previo y la cola principal permanecen intactos.
`PARALLEL_PROFILE.json` registra la derivación exacta desde el RTL de referencia.
El perfil nativo anterior se conserva byte por byte como procedencia, sin
trasladar su calificación al nuevo ledger.

## Pruebas funcionales

| Banco | Resultado revisado |
| --- | --- |
| Diferencial original/variante | 56 casos, 1.572 ciclos de comparación de salidas públicas, 112 retiros |
| Casos simultáneos nuevos | 9 ciclos con DMA y ACK juntos: mismo contexto, contextos distintos y mezcla de eventos válidos/incorrectos |
| Ledger existente | 21 casos, 29 rechazos, 18 escenarios de cuarentena, 98.816 bytes observados |
| Composición compartida | 6 casos, 27 retiros exitosos, 46.848 bytes exitosos, 4 reservas fallidas retenidas |
| Traza frente a referencia compacta | 6.663 líneas `SG`/`SF` idénticas, incluidos tiempos y datos |

El diferencial contiene siete casos exitosos y 49 de fallo. Comprueba que
reconocer un evento válido simultáneo no suprima el fallo del otro y que los
resultados demorados no liberen reservas. Incluye también las comprobaciones
anteriores de sesión, origen, destino, generaciones, estado, longitud,
notificaciones anticipadas y duplicadas. Los auditores detectaron cinco
alteraciones de la traza diferencial, seis del ledger y quince del banco completo.

Se detectaron los cuatro mutantes RTL: omitir autenticación DMA, omitir autenticación
ACK, desactivar la aceptación DMA del segundo contexto y suprimir el fallo ACK
cuando DMA también está presente. Este último pasó los primeros 51 casos y fue
detectado en el caso 52 (DMA válido + ACK incorrecto), demostrando la utilidad
de la ampliación del banco. Los resultados quedan en cada tanda. El criterio
exige el fallo observado; XSIM puede devolver código
cero aun ante `$fatal`.

## Medición física

La síntesis terminó en 687,692 segundos sobre `xcvu47p-fsvh2892-2-e`,
con periodos de 5,000/2,222 ns y las mismas restricciones de la referencia.

| Recursos sintetizados | Selector anterior | Validación paralela | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 22.845 | 22.624 | -221 (-0,97 %) |
| Registros | 21.710 | 21.709 | -1 |
| BRAM36 equivalentes | 0 | 0 | 0 |
| LUT como RAM distribuida (incluidas arriba) | 3.036 | 3.036 | 0 |
| CARRY8 | 1.405 | 1.493 | +88 |

El ahorro de LUT viene de lógica; la RAM distribuida permanece igual. Las
CARRY8 aumentan un 6,26 %: no hay una reducción uniforme de todos los recursos.

El auditor confirmó ausencia de latches, black boxes y extremos internos sin
restricciones. Conserva 13 entradas CDC-3 y 8 CDC-9 informativas revisadas.
Los puertos externos siguen sin calificación: 6.077 entradas y 5.684 salidas
sin delays, más dos entradas APB con cobertura parcial. Estos resultados no
cierran timing ni califican una interfaz de placa.

El ruteo alcanzó el límite de **1.200,116 segundos**, con ocho CPU, prioridad
baja y retorno `-15`. Conservó `optimized.dcp` y `placed.dcp`, pero no produjo
`routed.dcp`. Llegó a la iteración global 5; el último resumen intermedio de
setup fue WNS -0,203 ns y TNS -57,016 ns. Ese resumen parcial no reemplaza
informes finales ni permite concluir una mejora de frecuencia.

El informe de congestión de la iteración 120 señala ocho nets: offsets y
longitud del guard compacto, `lane_request`, `result_context`, un control de
emisión y lógica de admisión. Son localizaciones para investigar, no una
identificación exhaustiva de las causas. La inspección separada terminó en
81,976 segundos, exportó 80 caminos y conservó intacto el checkpoint.

| Slack colocado estimado (ns) | Selector anterior | Validación paralela |
| --- | ---: | ---: |
| Transporte, setup | +0,186 | -0,027 |
| Transporte, hold | -0,195 | -0,194 |
| Memoria, setup | +0,173 | +0,056 |
| Memoria, hold | -0,107 | -0,127 |

Los 20 peores caminos de setup de transporte exportados ahora parten de
`guard/lease.window_offset[0]` y terminan en enables de `guard/watermark`.
El peor tiene 19 niveles (10 CARRY8), 4,636 ns de datos: 1,295 ns de lógica y
3,341 ns estimados de interconectado. El camino DMA/ACK anterior no aparece
en esos 20; no se midió separadamente su nuevo margen. Los peores caminos
comparados tienen extremos diferentes y el guard no cambió de RTL. La nueva
colocación empeora el setup estimado global; menos niveles no implica un camino
más rápido. Ninguno de estos valores es timing final ruteado.

La aceptación requiere informes finales de ruteo, setup, hold, pulse width,
bus skew, cobertura interna, CDC y DRC. La síntesis o el timing colocado no
reemplazan esos informes. HBM nativa, localidad entre stacks, PCIe y placa
continúan requiriendo validación propia.

## Siguiente experimento propuesto

Investigar una variante aislada del guard que reduzca las restas dependientes
en las comprobaciones de ventana y región publicada. Una opción es precalcular
sus extremos al aceptar la configuración y comparar intervalos con aritmética
**de 65 bits**, conservando la comprobación del límite inferior. Para valores
sin signo, `offset >= base` y `bytes <= size - (offset - base)`, junto con
`offset - base <= size`, equivalen a `offset >= base` y
`offset + bytes <= base + size` cuando ambas sumas conservan el bit adicional.
Esta es una propuesta aritmética, todavía sin implementación ni medición física.

Primero debe demostrar equivalencia de admisión y prioridad de errores: rangos
vacíos, límites exactos, un byte fuera, desbordamiento de sumas de 64 bits,
revocación y reconfiguración. Después, repetir el diferencial del guard, el
banco compartido y el flujo físico con las mismas restricciones. Mantener
watermark, arbitraje, autenticación completa, retención y latencia. El guard de
referencia y la cola principal permanecen protegidos; una futura variante debe
usar archivos propios. El cierre nativo HBM y PCIe sigue siendo una etapa aparte.

## Evidencia y reproducción

Directorio: `D:/AI/artifacts/ntrpu-peer-parallel/20261008T035541080627Z`.
Los manifiestos `peer-parallel-*` usan el almacén incremental
`D:/AI/artifacts/ntrpu-content-store`, con restauración y repetición del auditor
por tanda. Las diez tandas (incluidos cuatro mutantes) fueron restauradas y
auditadas; `HANDOFF` reúne documentos, procedencia y recibos. No se retienen
ZIP duplicados; conservar `objects/` y `manifests/` juntos.

Herramientas: `prepare_ntrpu_peer_parallel.py`, `run_ntrpu_peer_parallel.py`,
`check_ntrpu_peer_parallel.py` y `fpga/tb/tb_ntrpu_peer_parallel.sv`.
La preparación genera una copia privada; el runner diferencial se ejecuta
en Linux con el entorno Vivado cargado. La composición y el flujo físico
reutilizan sus runners existentes, con un auditor de procedencia específico.
