# Etapa de escritura de contexto en las lanes

## Contrato y cambio

Experimento aislado del 2026-10-08, continuación de
[los límites de tamaño de 65 bits](PEER_READ_SIZE.md). Su checkpoint colocado
mostraba diecisiete de los veinte peores caminos de transporte entre el guard
y las lanes, con un enable de contexto de fanout 1.032.

`ntrpu_peer_read_lane_commit.sv` reserva el slot y evalúa todas las condiciones
de admisión en el handshake original. Guarda solicitud y direcciones; escribe
el contexto definitivo un ciclo después. La carga ancha del registro temporal
depende del handshake, y la del contexto definitivo depende de `stage_valid`.
La aritmética de admisión ya no controla directamente esa segunda carga.

La etapa utiliza un slot ya reservado: siguen existiendo dos reservas por lane
y cuatro globales. Puede escribir el contexto anterior y capturar la siguiente
solicitud en el mismo flanco. Añade un ciclo antes de ofrecer un comando nuevo
y 1.154 bits de estado declarado por lane; no añade buffers de payload. El
coste sintetizado debe medirse, porque puede diferir de los bits declarados.

Para una solicitud aceptada en E0, con el lector disponible:

| Flanco | Acción |
| --- | --- |
| E0 | Reservar el slot, actualizar watermark y capturar solicitud/direcciones. |
| E1 | Escribir contexto/direcciones definitivos; puede capturarse otra solicitud. |
| E2 | Primer handshake posible del comando de esa solicitud con el lector. |

La referencia podía hacer ese handshake en E1. Si el lector no está disponible,
el comando conserva sus datos y su validez hasta ser aceptado.

El watermark y la propiedad se actualizan al aceptar, como antes. La comprobación
de solapamiento ve también el contexto temporal. Configurar exige cero reservas;
comandos y resultados esperan a que su contexto esté escrito. `stop` y fallos
no descartan la escritura pendiente: sus resultados deben conservar la identidad
correcta y la reserva fallida. Una revocación posterior permite terminar trabajo
ya aceptado; una revocación en el handshake sigue rechazándolo. El reset frío
conserva su función de reiniciar todo el estado.

La lane original, el guard, los lectores y la cola principal permanecen
protegidos. Las variantes se seleccionan únicamente dentro de copias privadas.

## Validación funcional

El banco `tb_ntrpu_peer_commit.sv` compara la referencia y la variante mediante
flujos ordenados de transacciones públicas, permitiendo la latencia definida.
No exige igualdad de señales inválidas ni de ciclos entre implementaciones.
Comprueba estabilidad de comandos, payload y resultados bajo backpressure.

Pasaron 80 casos y 2.337 ciclos observados. Por variante se compararon
149 solicitudes, 110 comandos, 102 transferencias de datos, cuatro rechazos
y 142 resultados. Incluye 56 escenarios previos de autenticación de DMA/ACK,
dieciséis fallos con cada slot en la etapa y ocho escenarios de conflictos,
revocación, captura consecutiva, configuración, capacidad y reset frío.
El auditor rechazó seis alteraciones de evidencia.

También pasaron los quince casos del guard y los seis de la composición
compartida. Esta conserva 27 retiros exitosos, 46.848 bytes exitosos y cuatro
reservas fallidas retenidas, con pico de cuatro reservas globales. Se compararon
1.138 transacciones en 94 flujos contra el donante, conservando datos y orden
por canal y tipo, sin exigir timestamps iguales. El auditor conserva sus
quince controles negativos y verifica los mismos ajustes de los IP de lectura.

Se detectaron seis mutantes RTL: comando prematuro, uso de la solicitud viva al
escribir contexto, solapamiento que ignora la etapa, descarte con `stop`, slot
incorrecto y dirección viva al escribir. La aceptación depende del auditor,
además de las aserciones RTL: el mutante de descarte con `stop` llega a
`ALL_PASS` en XSIM, pero altera el resultado del caso 57 y el comparador lo
rechaza. No interpretar exit code cero o esa marca aislada como aprobación.

## Medición física

La síntesis terminó en 287,498 s y pasó la revisión de procedencia, cobertura
interna y CDC. Conserva las trece observaciones CDC-3 y ocho CDC-9, sin latches,
cajas negras ni endpoints internos sin restricciones.

| Recurso sintetizado | Donante de tamaño | Etapa de contexto | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 21.889 | 22.185 | +296 |
| Registros | 21.713 | 23.764 | +2.051 |
| CARRY8 | 1.405 | 1.405 | 0 |
| LUT de RAM distribuida | 3.036 | 3.036 | 0 |
| BRAM equivalentes a 36 Kb | 0 | 0 | 0 |

El ruteo terminó en 896,420 s, dentro del límite de veinte minutos.
Se conserva `xcvu47p-fsvh2892-2-e`, periodos de
5,000/2,222 ns y todas las restricciones de reloj y CDC originales.
Síntesis y ruteo usan ocho CPU, prioridad baja y límite de 1.200 s por tanda,
previa comprobación de capacidad en la VM compartida.

El auditor aprobó el timing interno: 36.184 redes completamente ruteadas, cero
errores de ruteo y cero endpoints fallidos de setup, hold y pulse width.
Los ocho controles de bus skew cumplen, con margen mínimo de 1,673 ns.
El READY de la interfaz del lector conserva fanin registrado. CDC final:
once CDC-3 y ocho CDC-9, sin clases nuevas; cobertura interna sin huecos.

| Slack final ruteado (ns) | Transporte | Memoria |
| --- | ---: | ---: |
| Setup | +0,006 | +0,028 |
| Hold | +0,011 | +0,007 |

El peor pulse width global es +0,579 ns. El margen mínimo de setup, 6 ps,
es estrecho: corresponde a este checkpoint y estas restricciones. No califica
otra integración ni demuestra ancho de banda en hardware. El camino crítico
de transporte va de `guard.compact[2].target_offset[7]` al enable de
`lane[1].sent[0][6]`: 18 niveles, nueve CARRY8 y 4,730 ns de datos, de los cuales
3,286 ns son interconectado ruteado. La carga ancha del contexto ya tiene su
etapa, pero la aritmética aún participa en la inicialización del contador.

El diseño final usa 21.873 LUT y 23.750 registros. No se compara este total
ruteado con la síntesis del donante para atribuir una reducción de área.
Las E/S conservan 6.077 entradas y
5.684 salidas sin retardos externos y dos entradas con retardo parcial.
PCIe, HBM nativa, E/S y placa siguen necesitando evidencia propia.

## Netlist y revisión DRC

El banco público pasó sobre el netlist ruteado sin editarlo, sin `force` y sin
SDF, en 131,251 s. Conservó los seis escenarios, 27 resultados exitosos,
46.848 bytes exitosos, cuatro resultados fallidos que retienen reserva y pico
de cuatro contextos. El auditor verificó 2.785 muestras y rechazó diez controles
negativos. Es simulación funcional del netlist; el cumplimiento temporal
procede de los informes STA anteriores.

La revisión DRC terminó en 188,079 s, sin modificar el checkpoint ni aplicar
waivers. Clasificó las siete advertencias: las seis PDCN-1569 corresponden a
entradas que no afectan la función de sus LUT según su INIT. La RTSTAT-10
enumera 3.404 redes: 3.302 son puertos externos del módulo fuera de contexto
sin carga interna, y 102 no tienen cargas lógicas hoja. No quedaron objetos
sin explicación. El reporte sigue teniendo advertencias (`drc_clean=false`);
la integración de E/S requiere su propia revisión.

## Siguiente paso propuesto

Conservar este checkpoint como referencia interna y conectar la variante a
un banco con el IP HBM nativo. Precargar una región de origen y comprobar su
lectura desde ambos canales, más patrones separados por pseudocanal para
verificar localidad. No equiparar puertos AXI distintos con bancos físicos
distintos sin comprobar las direcciones y los datos; ver
[localidad HBM](PEER_HBM_LOCALITY.md).

La integración debe repetir retención, backpressure, revocación, errores,
configuración y timing con sus restricciones reales. El margen de 6 ps no se
hereda a ese diseño. El endpoint PCIe, DMA/BAR y la placa continúan pendientes;
esta tanda cierra únicamente el bloque interno descrito aquí.

## Evidencia y reproducción

Directorio: `D:/AI/artifacts/ntrpu-peer-commit/20261008T065309077450Z`.
Cada tanda conserva fuentes congeladas, resultados, hashes y auditor. El almacén
incremental `D:/AI/artifacts/ntrpu-content-store` se restaura y el auditor se
repite por tanda; conservar juntos `objects/` y `manifests/`.
Las trece tandas fueron respaldadas, restauradas y revisadas. `HANDOFF` reúne
fuentes, documentos, informes resumidos y recibos; `OUTCOME.json` distingue
timing interno, simulación funcional del netlist y calificaciones pendientes.

Herramientas: `prepare_ntrpu_peer_commit.py`, `run_ntrpu_peer_commit.py` y
`check_ntrpu_peer_commit.py`. `COMMIT_PROFILE.json` registra la procedencia
del guard de tamaño y la lane con autenticación paralela, sin heredar calificación
física o de HBM. Los bancos existentes y el flujo físico permanecen sin cambios.
