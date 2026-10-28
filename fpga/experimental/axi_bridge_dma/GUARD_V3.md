# Reset compartido en la integracion VU47P

## Cambio y alcance

V3 aborda los dos CDC-11 encontrados en [el ruteo V2](GUARD_PHYSICAL.md):
los resets AXI y de control del puente AMD proceden de la misma fuente,
pero V2 los sincronizaba por separado. V2, escritor, mapper y maestro
AXI-Lite se conservan sin cambios; los archivos V3 son una variante aislada.

`SHARED_NATIVE_RESET=0` mantiene las cuatro fuentes independientes y es el
valor por defecto. El perfil `1` usa tres: enlace, reset nativo y PERSTn.
Cada una tiene una cadena de liberacion de tres etapas en gestion y otra
en AXI. La admision consume cada permiso una sola vez. Pasamos de 42 a 36
registros ASYNC_REG, y de 56 a 48 chequeos entre etapas.

La seleccion es un contrato de cableado, no una deduccion a partir de que
dos senales coincidan durante una prueba. La asercion de simulacion rechaza
entradas conocidas diferentes despues de estabilizar las asignaciones del
mismo instante. El modo independiente sigue observando ambos resets.

## Prueba de la fuente nativa

El perfil compartido se vincula explicitamente en `prepare_guard_v3.py`.
La auditoria abre el checkpoint V2 ruteado, donde ambos puertos tienen
cargas, y comprueba igualdad de sus drivers fisicos. V3 utiliza el mismo
XCI congelado; se verifica su SHA-256. Tras optimizar V3, el puerto de
control sin consumidores puede desaparecer: se registra como
`REMOVED_UNUSED_ALIAS`, manteniendo la prueba anterior y comprobando el
origen del reset AXI que sigue activo. No se agrega una carga artificial.

Las excepciones propias son tres max-delay a primeras etapas, nueve cortes
gestion -> CLR de liberacion AXI, seis cortes nativos -> CLR de gestion y
tres cortes fault -> PRE. Los 48 caminos entre etapas siguen cronometrados.
No se agregan clock groups generales, waivers ni retardos de E/S ficticios.

## Validacion

El banco ejercita ambos perfiles con dos relaciones de reloj, pulsos de
100 ps modelados, reloj detenido alto/bajo, recuperacion, cold reset y
transferencias pendientes. Comprueba que AW/W conservan sus ofertas y que
B/ACK tardios no liberan una transferencia fallida. Estos ACK son fixtures,
no confirmaciones fisicas de otra tarjeta.

Hay 100 casos de funcionamiento y cuatro de topologia: dos liberaciones
escalonadas con comparacion V2/V3 independiente y dos cableados incorrectos
que deben rechazarse. XSIM paso los 104 casos. La version portable final
paso otros 104 con Icarus y repitio los cuatro de topologia en XSIM.
Los 12 controles negativos del revisor rechazaron evidencia alterada,
incluidos CDC criticos ocultos, cortes entre etapas y fuentes distintas.

El ruteo `v3-route-20261011T082400Z` en Vivado 2026.1 termino en 850 s:

| Medida | Resultado |
| --- | --- |
| FPGA / PCIe / reloj AXI | VU47P FSVH2892 -2E / Gen3 x16 / 250 MHz |
| Redes ruteables / ruteadas / errores | 66 560 / 66 560 / 0 |
| CDC del supervisor | CDC-3: 3 Info; CDC-9: 6 Info; 0 Warning/Critical |
| ASYNC_REG / caminos entre etapas | 36 / 48, todos cronometrados |
| Margen minimo entre etapas setup / hold | +3,579 ns / +0,066 ns |
| WNS / WHS / WPWS global bajo restricciones actuales | +0,137 / +0,010 / 0,000 ns |
| Endpoints fallidos setup / hold / pulse width | 0 / 0 / 0 |

La colocacion no inserto el BUFG de cold reset observado en V2: identifico
un candidato y lo descarto por degradacion de timing. El reporte de buffers
del supervisor queda vacio; el BUFGCE de gestion permanece como contrato OOC.

La primera tanda XSIM agoto 30 s durante el arranque de un caso; se conserva
y se amplio a 90 s el limite de proceso. El watchdog del banco no cambio.
La comprobacion local detecto ademas que la asercion con `$isunknown` sobre
una concatenacion no rechazaba el caso incorrecto en Icarus. Se reemplazo
por comparaciones de cuatro estados, dejando estabilizar un delta antes de
evaluar aliases. El cuerpo sintetizable es identico antes y despues de ese
ajuste; se comprueba por separado del hash de archivo completo.

La primera sintesis V3 termino, pero su auditoria intento medir un puerto
ya eliminado. `recover_guard_v3.tcl` reutiliza su DCP de sintesis, aplica los
cortes exactos y vuelve a optimizar con la prueba de alias del V2 cargado.
Se conserva el intento fallido junto con la recuperacion.

## Reproduccion

Desde este directorio, con carpetas nuevas dentro del espacio privado:
los runners nativos esperan todas sus dependencias juntas, incluido
`ntrpu_xdma_local_master.sv` copiado desde `fpga/rtl/`. El directorio
`source/` del respaldo conserva esa distribucion; no modifica el maestro.

```bash
python3 -B run_guard_v3_rtl.py SALIDA_RTL --phase rtl \
  --settings VIVADO_SETTINGS --execute-native
python3 -B run_guard_v3_native.py SALIDA_NATIVE --example EJEMPLO_AMD \
  --reference-checkpoint REFERENCIA_V2_RUTEADA \
  --settings VIVADO_SETTINGS --execute-native
python3 -B run_guard_v3_route.py SALIDA_ROUTE \
  --checkpoint SALIDA_NATIVE/output/optimized.dcp \
  --reference-checkpoint REFERENCIA_V2_RUTEADA \
  --settings VIVADO_SETTINGS --execute-native
```

`review_guard_v3.py` revisa trazas, alias, cadenas y excepciones;
`check_guard_v3.py` intenta ocultar o alterar evidencia para comprobar que
el revisor la rechaza. Las tandas tienen limites, dos CPU, nice 10,
manifiestos y respaldo local independiente de la vida de la VM.

## Limites y siguiente integracion

Este trabajo es PCIe con BRAM de prueba. No integra HBM ni acredita un
PCB o tarjeta de 512 GiB. Siguen pendientes el origen fisico del reloj de
gestion y los presupuestos externos: 1589 bits de entrada y 511 de salida
carecen de retardos. La cobertura CDC del supervisor no cubre esos puertos.
El informe metodologico global conserva LUTAR-1 (7), TIMING-9 (1),
TIMING-16 (11), TIMING-18 (7) y XDCB-5 (42); DRC conserva RTSTAT-10 (1).
Los 11 TIMING-16 estan dentro del IP PCIe y se revisan mediante consulta
puntual del checkpoint: los mismos endpoints dan +0,749 a +0,945 ns en
STA, mientras metodologia informa -1,251 a -1,055 ns. Las rutas atraviesan
`CFGMGMTREADWRITEDONE` o `CFGFUNCTIONSTATUS[2]`; el XDC nativo conserva
multicycle setup-start de dos ciclos sobre las salidas `CFG*`. La diferencia
de 2 ns es compatible con esa excepcion existente; no constituye una
justificacion para silenciar los avisos. La consulta no cambia XDC ni DCP.

El gate local previo a HBM queda cumplido: cero CDC criticos en el
supervisor, rutas entre etapas conservadas y pruebas funcionales completas.
Falta integrar HBM y validar el contrato de reset/reloj de ese controlador,
ademas de los pendientes de tarjeta. No hay simulacion nativa post-ruteo.
La simulacion de pulsos breves no caracteriza anchos minimos fisicos ni MTBF.

## Evidencia y respaldo local

Raiz: `D:/AI/artifacts/ntrpu-axi-bridge-guard-integration/`.
El paquete `20261011-guard-v3-bundle.zip` conserva fuentes, trazas, reportes,
DCP de referencia/sintesis/optimizacion/ruteo, intentos fallidos y recibos
por tanda. Su restauracion independiente es `20261011-v3-bundle-restored/`.
`V3_BUNDLE_BACKUP.json` registra tamano y SHA-256; `V3_BUNDLE_REVIEW.json`
contiene el veredicto completo. Cada fase incluye su manifiesto SHA-256.

Para revisar el paquete restaurado sin Vivado:

```bash
python -B 20261011-v3-bundle-restored/source/review_guard_v3_bundle.py \
  20261011-v3-bundle-restored
```

`V3_PROCESS_COMPLETION.json` consulta solo los procesos privados de estas
tandas. La cola principal, el RTL V2 y el nodo DDR se conservan sin cambios.
