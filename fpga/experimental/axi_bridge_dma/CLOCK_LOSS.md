# Ensayo nativo de interrupcion del reloj AXI

Estado: **PASS en Vivado/XSIM 2026.1 para reloj detenido en bajo y alto**,
tanda `native-r0-20261011T045953Z`. Ambas revisiones se repitieron sobre la
evidencia descargada y coinciden con la VM. No cambia la cola principal ni
acredita perdida fisica del reloj PCIe.

## Resultado nativo y respaldo

Veredicto: `NATIVE_AXI_BOUNDARY_CLOCK_INJECTION_PASS`.

| Medicion del VCD | Detenido bajo | Detenido alto |
| --- | --- | --- |
| Deteccion desde la inyeccion | 670,999 ns | 672,999 ns |
| Transiciones AXI durante pausa | 0 | 0 |
| Flancos descendentes de gestion durante pausa | 128 | 128 |
| Duracion entre pausa y liberacion | 1276 ns | 1278 ns |
| Flancos descendentes de gestion tras liberar | 64 | 64 |
| Flancos ascendentes AXI tras liberar | 160 | 160 |

En ambos casos permanecieron liberados PERSTn y los resets nativos, con link
nativo activo. El guardian cerro la admision con causa exclusiva de reloj
(`guard_code=2`), sin borrar ownership ni identidad del lector pendiente.
Los fallos sincronos del escritor y mapper permanecieron bajos mientras AXI
estuvo detenido; al volver el reloj quedaron activos, sin rearme ni exito falso.

La tanda uso dos CPU, nice 10 y 826,918 s: compilacion 3,242 s, elaboracion
367,205 s y simulaciones 218,668 / 236,175 s. Los 623 archivos de la semilla y
el binding original permanecieron intactos. No hubo avisos en la compilacion
del banco; los logs conservan avisos de elaboracion del entorno/modelo AMD y
de `unique case` del modelo XDMA durante simulacion.

Evidencia: `D:/AI/artifacts/ntrpu-axi-bridge-clock-loss/native-r0-20261011T045953Z/`.
El archivo remoto contiene 45 archivos verificados, ocupa 823580 bytes y tiene
SHA-256 `cdbb85395d0504274910c0cbfbed9a86aceecd50eb86d7cf668b03be4937f32a`.
`LOCAL_REVIEW_0.json` y `LOCAL_REVIEW_1.json` reproducen las revisiones remotas.
El paquete `20261011-clock-loss-bundle.zip` conserva evidencia, fuentes,
documentacion y referencias de recuperacion; su recibo queda en
`20261011-backup-receipt.json` bajo el mismo directorio de artefactos.

## Punto de inyeccion y alcance

La fuente PIPE generada usa osciladores temporizados internos. En la tanda de
[aislamiento](ISOLATION_NATIVE.md), PERSTn bajo dejo 500 flancos AXI durante
los casi 2 us finales. Ese caso verifica reset con reloj activo.

`clock_loss_scenario.svh` fuerza exclusivamente la salida publica
`board.AXI_PCIE_EP.axi_aclk_out`, ya estabilizada en bajo o alto. Cada nivel
requiere una simulacion nueva. Conserva PERSTn, ambos resets nativos y link
liberados. Una caida concurrente invalida el caso de perdida exclusiva del reloj.
No fuerza registros internos ni fabrica respuestas. El VCD debe mostrar el
mismo reloj detenido en la salida, el cliente y el guardian. El alcance es una
interrupcion inyectada en la frontera AXI; no demuestra que se hayan detenido
todos los relojes internos del nucleo ni modela un fallo del GT o de REFCLK.

La semantica de fuerza sustituye temporalmente el driver HDL; al liberar se
restaura su comportamiento. Ver [UG900: fuerzas de simulacion](https://docs.amd.com/r/en-US/ug900-vivado-logic-simulation/Using-Force-Commands).
Para hardware, AMD identifica `axi_aclk` como derivado de TXOUTCLK, sin garantia
de continuidad durante reset. La gestion requiere una fuente independiente.
[PG194: senales globales](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Global-Signals).

## Criterios

Se reutiliza el arranque del banco nativo, sus dos mappings y transferencias;
el caso nuevo reserva un tercer comando `{session=9, id=3}`, origen `0x1000`,
64 bytes, y mantiene pendiente al lector. El revisor nuevo valida la admision
e identidad de ese comando; no reconstruye los payloads de calentamiento ni
repite la certificacion de aislamiento host de la tanda anterior.

| Intervalo | Condicion exigida |
| --- | --- |
| Antes | AXI a 250 MHz, gestion a 100 MHz, guardian armado sin fallo |
| Pausa | Cero transiciones AXI; 128 flancos descendentes de gestion |
| Deteccion | `guard_code=2`, stop activo y admision cerrada antes de 76 periodos de gestion |
| Retencion | Cold reset liberado; tag, direccion, longitud y ownership estables |
| Durante pausa | Fallos sincronos de escritor/mapper aun bajos: no hubo reloj para registrarlos |
| Recuperacion | 64 ciclos de gestion; AXI vuelve, fallos locales se registran; no rearme ni exito |

Los 76 periodos son un criterio conservador de banco para `CLOCK_TIMEOUT=64`,
con margen para ida/vuelta y registro. No son un limite certificado de placa.
Solo se cubre un lector pendiente; no se califica recuperacion de TLPs en vuelo.

## Reproduccion

```powershell
python -B fpga/experimental/axi_bridge_dma/check_clock_loss_reviewer.py
python -B fpga/experimental/axi_bridge_dma/prepare_clock_loss_native.py NUEVA_CARPETA_LOCAL
```

El preparador fija los hashes del banco base y guardian, genera
`ntrpu_axi_bridge_clock_loss.sv` e incluye fuentes, revisor y runner. Nunca
conecta por SSH. Los controles del revisor son trazas sinteticas en memoria;
su exito no equivale a compilar RTL ni a ejecutar la IP AMD.

La preparacion local del 2026-10-09 paso dos casos sinteticos del revisor y
quince corrupciones rechazadas, incluyendo reloj que sigue oscilando, ausencia
de gestion, deteccion tardia, reset concurrente, cambio de tag, fallo sincrono
prematuro y rearme. Icarus acepta la sintaxis del banco generado y su include
seleccionando un top vacio: es una comprobacion de parser, no elaboracion del
cliente ni de la IP AMD. Los artefactos preparados y recibos se conservan en
`D:/AI/artifacts/ntrpu-axi-bridge-clock-loss/`.

Para repetir la prueba, copiar una carpeta preparada a un directorio nuevo
de la VM y ejecutar desde ella:

```bash
python3 -B run_clock_loss_native.py /home/ubuntu/vivado/ntrpu-side/clock-loss/NUEVA_TANDA \
  --seed SNAPSHOT_WORK_PRIVADO --example EJEMPLO_R3 \
  --settings /lambda/nfs/vivado/AMDDesignTools/2026.1/Vivado/settings64.sh --execute-native
```

Se requieren la semilla compilada y el ejemplo ya respaldados en
`D:/AI/artifacts/ntrpu-axi-bridge-native/`. El runner copia la semilla, compara
las fuentes del cliente, usa dos CPU/nice 10 y conserva hashes, comandos,
log, VCD y revision por nivel. Limites: compilacion 90 s; elaboracion 600 s;
cada simulacion 600 s. Solo termina grupos de procesos propios si agota tiempo.
No inicia SSH ni modifica la semilla. El archivo final excluye el nuevo cache
compilado. Exigir revision y respaldo local verificado antes de cerrar la tanda.

## Cruces de reloj pendientes de implementacion

La [revision posterior de sintesis y restricciones](GUARD_CDC.md) verifica
las siete cadenas y conserva sus rutas internas. Sigue abierto un CDC-10
critico en la admision y dos alertas por LUT en controles asincronos.

`request -> echo_a` y `echo_a[2] -> echo_m` forman el sondeo de ida/vuelta.
`native_ok -> healthy_m`, `allow -> admit_a` y `fault -> stop_a` usan controles
asincronos: revisar asercion, liberacion, pulsos minimos y recovery/removal.
Cold reset tiene una liberacion por dominio y no pertenece al reset operativo.
No agregar excepciones generales para ocultar estas rutas; conservar las
restricciones del proveedor y revisar CDC/STA en el padre implementado.
[UG949: CDC](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Clock-Domain-Crossing).

El lector DDR/HBM y el ACK siguen modelados. Si llegan desde otros dominios,
requieren transferencia coherente de datos e identidad, con inventario retenido
ante reset. Este ensayo no incorpora ese transporte ni selecciona oscilador,
pinout, alimentacion o componentes fisicos para la placa.
