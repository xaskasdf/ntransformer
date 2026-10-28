# Control privado y supervision del puente

Este candidato pertenece exclusivamente al experimento AXI Bridge. No cambia
el top, los contratos de reset ni la cola de simulacion principal. La prueba
nativa anterior sigue descrita en [MAPPING.md](MAPPING.md); este guard nuevo
tiene validacion RTL local y una integracion nativa acotada documentada en
[ISOLATION_NATIVE.md](ISOLATION_NATIVE.md): trafico host seleccionado y PERSTn
con un contexto pendiente. El [banco de interrupcion AXI](CLOCK_LOSS.md) paso
en Vivado/XSIM 2026.1 el 2026-10-11, deteniendo la salida publica en bajo y alto.
Conservo el contexto y el fallo tras recuperar reloj. La perdida fisica del
reloj PCIe y el cierre CDC/STA siguen pendientes.
La [revision OOC de CDC/STA](GUARD_CDC.md) del 2026-10-11 conserva un CDC-10
critico y dos LUTAR-1; el ensayo de restricciones no autoriza implementacion.
La [variante experimental V2](GUARD_V2.md) separa los controles asincronos y
registra sus propias regresiones; el archivo del guardian V1 se conserva.

## Frontera frente al host

`audit_host_boundary.py` revisa la copia restaurada del ejemplo nativo r3:
19 conexiones AXI-Lite pertenecen al mapper y 16 al escritor. El trafico
entrante `m_axib_*` termina en la BRAM de 128 KiB del ejemplo. Las direcciones
se reducen a 17 bits dentro de ese destino; no se convierten en direcciones
del control privado. El auditor fija hashes del wrapper, aplicacion y XCI revisados,
comprueba conexiones y rechaza cambios de perfil relevantes. No es un parser
general de HDL ni una prueba formal de no interferencia del nucleo opaco.

El perfil habilita solo BAR0 de PF0, deshabilita AXI-Lite master, bypass,
MSI-X y XVC VSEC externos. Los parametros inactivos de otras funciones no
acreditan por si mismos que existan esas funciones en el dispositivo generado.
AMD documenta el VSEC del puente como accesible por el control AXI-Lite;
el recorrido de capacidades desde PCIe omite ese VSEC.
[PG194: mapa del puente](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Bridge-Register-Memory-Map).
Esto respalda la separacion prevista, sin demostrar que todo acceso PCIe
malicioso sea inocuo para esta instancia.

La prueba nativa descrita en [ISOLATION_NATIVE.md](ISOLATION_NATIVE.md) paso
48 solicitudes host durante idle, mapping y ownership. Comprueba accesos BAR,
Configuration Write/Read Type 0, completions, palabras privadas y destinos TLP
mediante log y VCD. Cubre esos offsets y estados; no demuestra aislamiento
exhaustivo ni ausencia de toda alteracion transitoria. PERSTn, bus mastering
y administracion de recursos siguen bajo control del host: aislamiento de la
tabla no significa disponibilidad frente a un host adverso.

## Guardian de admision

`ntrpu_axi_bridge_guard.sv` usa `mgmt_clk` independiente y continuo. Un request
estable cruza a `axi_clk` y vuelve como echo; el siguiente request espera esa
respuesta. Asi evita el aliasing de un toggle libre. El timeout por defecto es
64 ciclos de gestion; debe superar la ida/vuelta CDC al reloj AXI mas lento
permitido. Detecta ausencia de progreso, no frecuencia exacta ni jitter.

Conexion propuesta, sin alterar el cliente existente:

| Salida del guardian | Entrada del cliente | Comportamiento |
| --- | --- | --- |
| `client_cold_rst_n` | `cold_rst_n` | Solo reset de configuracion; liberacion tras tres flancos AXI |
| `client_link_up` | `link_up` | Abre tras calificacion; cierra aun sin reloj AXI |
| `client_stop` | `stop` | Fallo retenido, sin falso stop al arrancar |
| `fault`, `fault_code` | Gestion local | Bit 0: disponibilidad nativa; bit 1: reloj sin progreso |

`native_link_up`, ambos resets AXI nativos y PERSTn participan en disponibilidad.
Una caida breve borra asincronamente su cadena de calificacion; la gestion
retiene el fallo despues del primer armado. Recuperar enlace/reloj no rearma.
No se genera reset operativo, se cancela VALID ni se fabrica B/R/ACK. El contexto
queda retenido incluso si respuestas reales llegan tarde. Un nuevo arranque de
configuracion puede borrarlo; no usar ese mecanismo como recuperacion ordinaria.

El supervisor de aplicacion existente fuerza resets y por eso no se conecta
directamente a estos registros retenidos. El guardian usa cadenas de tres FF
con `ASYNC_REG`, sin modificar aquel supervisor. La liberacion sincronizada
sigue la [guia UG949](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Synchronous-Reset-vs.-Asynchronous-Reset).
La composicion de disponibilidad asincrona, sus anchos minimos de pulso,
recovery/removal, CDC y colocacion requieren revision fisica. No se agregan
false paths generales ni se declara cierre de timing desde simulacion RTL.

## Validacion reproducible

```powershell
python -B fpga/experimental/axi_bridge_dma/run_guard_local.py NUEVA_SALIDA --toolchain TOOLCHAIN --example EJEMPLO_NATIVO_R3
```

El runner congela fuentes y la frontera auditada; usa Icarus local y guarda
log, VCD y revision independiente. El banco conecta el escritor real sin
cambios; mapper, lector, destino AXI y ACK son fixtures. No simula PCIe nativo.
Hay 17 escenarios en cuatro relaciones/fases de relojes: arranque, nominal,
ausencia inicial de reloj, pausas, reloj detenido alto/bajo, cuatro entradas
nativas, ofertas AW/W retenidas, burst parcialmente emitido, respuestas tardias,
retencion tras recuperacion y nuevo reset de configuracion.

La primera tanda `20261009-local-r0` paso 68 ejecuciones en 2,88 segundos,
con coincidencia log/VCD. El watchdog de prueba, de 32 ciclos, detecto las
paradas en 33-38 ciclos de gestion observados, dentro del limite de 44 que
incluye CDC y registro del fallo. Las relaciones AXI/gestion fueron 250/100,
125/71,43, 71,43/250 y 250/250 MHz, con fases distintas. Esto no convierte
32 ciclos en un parametro certificado para una placa.

Las mutaciones comprueban watchdog desactivado, PERST ignorado, rearmado
automatico y reset indebido del contexto; tambien se alteran conexiones host,
perfil de IP, log y payload VCD. Los resultados y respaldo se guardan en
`D:/AI/artifacts/ntrpu-axi-bridge-guard/`. La integracion nativa posterior tiene
su evidencia en `D:/AI/artifacts/ntrpu-axi-bridge-isolation/`. Pendientes:
perdida fisica del reloj PCIe, CDC y sintesis/timing, gestion fisica,
lectura DDR/HBM y protocolo de ACK entre tarjetas.
