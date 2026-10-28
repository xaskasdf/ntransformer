# Receptor doble con reservas y cruce hacia HBM


Actualización del 2026-10-07: el [retorno sintetizable de lectura](HBM_READ_RETURN.md) tiene un bloque experimental y evidencias separadas. Las lecturas de comprobación descritas aquí siguen perteneciendo a este banco de escritura; no se reclasifican como pruebas del nuevo lector.

## Alcance

Trabajo aislado del 2026-10-07, continuación de la [preferencia de reservas locales](PEER_RX_RESERVE_LOCALITY.md). La nueva composición está en `fpga/experimental/ntrpu_dual_hbm_async.sv`; conserva sin cambios el receptor, el asignador y los bloques acreditados en `fpga/rtl/`. No reemplaza el transporte compartido ni modifica la cola del hilo principal.

Dos entradas de comandos y payload de 512 bits usan el receptor de ocho bancos y dos grupos físicos. Cada salida atraviesa la IP AMD `hbm_cdc`, el control de capacidad y los conversores de anchura/protocolo existentes. La frontera de memoria ofrece dos interfaces AXI3 de 256 bits. El perfil nominal usa 200 MHz de origen y aproximadamente 450 MHz de memoria, con fase independiente.

## Contrato y retención

- La publicación espera la disponibilidad sincronizada de ambos stacks. `stop`, pérdida de disponibilidad, error externo o de adaptador cierran nuevas admisiones; no se deshabilita el camino necesario para vaciar trabajo aceptado.
- Se rechazan direcciones fuera del espacio de 34 bits antes de asignar banco. El receptor conserva alineación de 64 bytes, máximo de 64 beats y límite de página de 4 KiB. Estos controles no sustituyen permisos, regiones registradas ni épocas del protocolo externo.
- Un banco retorna después del último W aceptado **por el CDC de origen**. No equivale a una escritura terminada en HBM. El inventario AW–B incluye lo almacenado dentro del CDC; cada canal mantiene hasta cuatro escrituras en vuelo.
- `drained` exige inventario vacío y retiro de completions. Un fallo puede coexistir con vaciado: el consumidor debe comprobar también el estado de error.
- Un fallo antes de completar una barrera la deja retenida hasta reset frío. Una barrera ya anunciada conserva VALID e identidad incluso si posteriormente aparece un fallo. Se añadió `fence_offered` para mantener ese contrato bajo backpressure.
- No hay rearmado automático ni reset caliente de las IP. Un BID nativo inválido queda en cuarentena; no se convierte en éxito ni se borra su obligación pendiente.

## Pruebas funcionales acreditadas

XSIM con los conversores AMD reales y un modelo ligero de almacenamiento pasó **11 escenarios por política**, `PREFER_LOCAL=0/1`. Cada ejecución contiene 68 comandos, 242.112 bytes de carga y 7.878 beats de lectura de 32 bytes; algunas lecturas repiten datos de escenarios anteriores. Un comando queda deliberadamente sin completion para comprobar cuarentena de BID. Las demás transferencias se completan y se comparan con patrones independientes.

Se ejercitan longitudes de 1/8/9/64 beats, dos oleadas sin reset ocupando los ocho bancos, liberación asimétrica, respuestas B y completions retenidas, barreras, pausa/cambio de reloj de memoria, stop a mitad de payload, fallo externo, dirección fuera de rango, BID inválido, pérdida de inicialización y BRESP SLVERR. La última prueba inyecta un fallo con una barrera ya ofrecida. Hay 1.670 ciclos de entrada simultánea por política en esta carga; no se observa diferencia de duración entre políticas aquí.

El checker reconstruye comandos, préstamos y generaciones, payload de origen, fragmentos nativos, respuestas, completions y lecturas. Rechaza 17 corrupciones de traza. No presupone que cada fragmento nativo use el máximo de 16 beats: exige límites, continuidad y cobertura completa. El primer intento falló por esa suposición incorrecta del banco y permanece como diagnóstico.

El perfil corto previo al modelo nativo también pasó: tres escenarios, 58 comandos, 36.224 bytes escritos y 1.460 beats leídos, ocho bancos ocupados y dos oleadas sin reset. Este preflight aún usa almacenamiento ligero; por sí solo no acredita el controlador HBM nativo.

## Simulación con controlador HBM nativo

El mismo RTL y banco corto pasaron con el modelo AMD HBM de 16 GB y puertos AXI 00/16: **58 comandos, 36.224 bytes y 1.460 beats de lectura**, comparados mediante el ledger independiente. Se observaron 226 ciclos de entrada simultánea, ocho bancos ocupados y dos oleadas sin reset. La ejecución terminó a 62,714459 µs simulados en 1.033,531 s de pared; los 17 controles de corrupción fueron rechazados.

La primera captura Tcl no encontró las señales porque XSIM escapó el nombre del top parametrizado. Su traza funcional se conserva intacta. Una ventana adicional, con **el mismo snapshot verificado por SHA-256**, observó las 17 señales hasta 49 µs: ambos stacks listos, resets liberados y `started=1` sin fallo. El arranque fue a 48,390200 µs. Sus primeros 490 eventos coinciden exactamente con la ejecución completa; ocho controles negativos verifican la revisión de esta observación.

`CATTRIP` permanece `X` en ambos stacks: **el estado térmico no está calificado**. No se forzaron señales de inicialización. La comprobación usa lecturas del banco; todavía no constituye un retorno de lectura sintetizable ni una prueba sobre hardware.

El script de futuras ejecuciones incorpora la búsqueda relativa de señales ya probada en esa ventana y falla antes de simular si falta alguna. La tanda completa conserva su script original; no se atribuye retrospectivamente a la versión corregida. Los cuatro sondeos de visibilidad y el intento de observación sin señales están archivados como diagnósticos.

## Implementación física y comparación del circuito

La implementación aislada en `xcvu47p-fsvh2892-2-e` pasó timing interno con relojes de 5 ns y 2,222 ns, incertidumbre de setup de 0,100 ns y hold de 0,050 ns. El primer ruteo tenía −0,079 ns en memoria; `phys_opt_design -directive Explore` y nuevo ruteo, sin cambiar RTL ni relajar restricciones, dieron:

| Dominio | Setup mínimo | Hold mínimo |
| --- | ---: | ---: |
| Origen, 200 MHz | +0,349 ns | +0,012 ns |
| Memoria, aproximadamente 450 MHz | +0,014 ns | +0,011 ns |

El margen mínimo de ancho de pulso es +0,579 ns. Ocupa **11.968 LUT, 16.613 registros y 15 BRAM36 equivalentes**. Hay 22.704 redes completamente ruteadas y cero errores de ruteo. El CDC registra 15 hallazgos informativos CDC-3 y 12 CDC-9, sin advertencias/críticos; se conservan 21 registros de sincronización de arranque y 12 restricciones de bus-skew, con margen mínimo de 1,583 ns.

Las **13 advertencias DRC permanecen**, sin waivers. La auditoría del checkpoint demostró que los 12 pines señalados por PDCN-1569 no afectan a sus tablas INIT. RTSTAT-10 enumera 1.048 redes: 862 conexiones externas del bloque aislado sin carga interna y 186 sin cargas lógicas. No quedan objetos sin clasificar; esto no convierte el informe en DRC limpio ni califica las E/S externas.

El netlist ruteado, conservado byte por byte en una biblioteca separada, coincide con la referencia RTL en los 11 escenarios: 6.795 muestras de origen, 13.988 de memoria y siete comprobaciones de reset frío. Las inyecciones PAYLOAD y READY fueron rechazadas, junto con los 17 controles de traza. Es simulación funcional acotada, sin SDF ni prueba formal.

El cierre es **del bloque aislado**. Mantiene 1.393 entradas y 911 salidas sin retardos externos; no incluye el hard block HBM, PCIe ni restricciones de placa. El margen de 14 ps requiere nueva revisión al integrarlo.

## Tandas y respaldo

Raíz local de evidencia: `D:/AI/artifacts/ntrpu-dual-hbm/20261007T054144808111Z/`.

| Tanda | Resultado |
| --- | --- |
| `20261007T054656583233Z-sim1` | Rechazada: el banco exigía un tamaño de fragmento innecesariamente estricto |
| `20261007T054911798824Z-sim1` | Diez escenarios exploratorios; sustituida por la revisión con retención de barrera |
| `20261007T055540059182Z-sim1` | Once escenarios y revisión independiente aprobados |
| `20261007T055540074707Z-sim0` | Mismo banco y RTL, política de referencia aprobada |
| `20261007T055737272816Z-native` | Preflight corto aprobado; el nombre de tanda no significa modelo HBM nativo |
| `20261007T055947031032Z-alias` | Mutación RTL rechazada: duplicar los datos de un canal causa `NATIVE_DATA` |
| `20261007T060521498403Z-fence_drop` | Mutación RTL rechazada: retirar una barrera ya anunciada causa `FENCE_STABLE` |
| `20261007T060011680854Z-hard_compile` | Modelo HBM nativo elaborado y snapshot revisado; 851,502 s |
| `20261007T061822261231Z-hard_traffic` | Tráfico nativo aprobado junto con la observación adicional; 1.033,531 s |
| `20261007T063328246431Z-observe` | Diagnóstico rechazado: búsqueda Tcl del scope, aunque el proceso devolvió cero |
| `20261007T064156087004Z-observe` | Arranque observado en el mismo snapshot, prefijo idéntico y revisión aprobada; 697,209 s |
| `20261007T060410178176Z-physical` | Intento inicial cancelado; se sustituyeron restricciones generales por las excepciones CDC específicas acreditadas |
| `20261007T060720666438Z-physical` | Primer ruteo revisado, timing de memoria abierto |
| `20261007T062657331756Z-refine` | Refinamiento con timing interno aprobado; 285,755 s |
| `20261007T063445364021Z-netlist` | Comparación funcional RTL/netlist y controles negativos aprobados; 142,595 s |
| `20261007T064304588268Z-drc` | Auditoría de todos los objetos DRC aprobada; 89,351 s |

Cada tanda revisada se copia con hashes y se incorpora al almacén incremental `D:/AI/artifacts/ntrpu-content-store`. Los ZIP de transferencia solo existen en memoria. Los informes y manifiestos distinguen ensayos rechazados, preflight y simulación nativa.

## Reproducción y siguientes dependencias

```bash
source /home/ubuntu/vivado/AMDDesignTools/2026.1/Vivado/settings64.sh
taskset -c 26,27 python3 -B tools/run_ntrpu_dual_hbm.py NUEVA_TANDA \
  --binding /ruta/acreditada/ntrpu_hbm_native_rw.sv \
  --stub /ruta/acreditada/ntrpu_hbm_native_stub.sv --prefer-local 1
python3 -B tools/check_ntrpu_dual_hbm.py NUEVA_TANDA
```

Los nombres concretos del stub de referencia quedan en los manifiestos; no se genera uno arbitrario para sustituir el modelo aceptado. `--native-profile 1` selecciona el banco corto, no el controlador nativo.

La revisión de las tandas nativas conservadas se reproduce con:

```bash
python3 -B tools/check_ntrpu_dual_hbm_native.py TANDA_HARD_TRAFFIC/output \
  --observation TANDA_OBSERVE/output
python3 -B tools/check_ntrpu_dual_hbm_physical.py TANDA_REFINE/output
python3 -B tools/check_ntrpu_dual_hbm_netlist.py TANDA_NETLIST/output
python3 -B tools/check_ntrpu_dual_hbm_drc.py TANDA_DRC/output
```

Antes de integrar en placa faltan el retorno de lectura CDC, el adaptador al protocolo peer completo, autorización de regiones, PCIe y cómputo en una composición común. La lectura del banco de pruebas no implementa esos caminos. El [plan de integración](DUAL_HBM_INTEGRATION_PLAN.md) establece contratos y pruebas R0–R2/P0–P1/C0; la [revisión de cierre de placa](BOARD_CLOSURE_REVIEW.md) distingue sus dependencias eléctricas.
