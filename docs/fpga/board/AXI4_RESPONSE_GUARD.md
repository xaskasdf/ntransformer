# Protector causal de respuestas AXI4

## Revisión posterior del índice inicial

La ampliación con fallos antes/durante el certificado encontró un índice
indeterminado al arrancar en la simulación RTL original. Se conserva la
evidencia previa con su alcance; no prueba todas las secuencias iniciales.
La revisión candidata RTL R2 y el banco R3 corrigen el índice y una espera
prematura del banco, y pasan la matriz ampliada y la regresión anterior.
La síntesis previa pertenece a los módulos originales. Ver
[diagnósticos, validación y siguientes pasos](AXI4_SECTION_FAULT_LIFETIMES.md).

## Contrato implementado

`fpga/rtl/ntrpu_axi4_response_guard.sv` observa AW/W/AR aceptadas y filtra
B/R antes del consumidor protegido. Usa 32 obligaciones de escritura y
ocho lecturas por cada uno de los 16 IDs. No contiene productor, Clock
Converter ni XDMA; su integración sigue pendiente.

B necesita AW y una ráfaga W completa de la longitud correcta, aceptadas
en flancos anteriores. W puede preceder a AW. R necesita una lectura de
su RID, en orden dentro de ese ID, con RLAST exactamente donde corresponde.
Se permite reordenar entre IDs. Un crédito válido no demuestra que los
datos o RESP sean semánticamente correctos.

Una respuesta sin crédito, con RLAST incorrecto o modificada/retirada bajo
bloqueo deja su canal en cuarentena permanente. No recibe READY y ninguna
solicitud posterior puede autorizarla retroactivamente. Cada canal tiene
un registro de salida: una respuesta sana ya capturada conserva VALID y
payload hasta que el consumidor la acepte, incluso ante otro fallo.

Las obligaciones siguen ocupando capacidad mientras la respuesta está en
ese registro. Un fallo del inventario congela la dirección afectada; las
salidas sanas ya registradas y la otra dirección pueden terminar. Solo
la inicialización de configuración borra estado. No hay reset operacional.
READY de retorno depende del crédito, ID/LAST y espacio disponible; su
camino combinacional requiere revisión temporal en el top integrado.

## Prueba aislada revisada

`20261005T221338Z` pasa **19 escenarios**, 36,871 s de Vivado y 240,63 µs
simulados. El revisor reconstruye cada ciclo con colas Python independientes,
compara inventarios, capacidad, permisos y cada respuesta protegida de
512 bits. También rechaza **32 alteraciones de evidencia**.

- Cuatro escenarios sanos: 224 escrituras, 224 lecturas, 5.651 transferencias
  W y 5.648 R. Incluyen vuelta repetida de las 32 escrituras, todas las
  128 lecturas, ráfagas de 1/2/4/16/64/256 transferencias, 31 W completas
  más una parcial antes de AW, B reordenadas y R intercaladas entre IDs.
- El flujo continuo acepta nuevas solicitudes y retira B/R simultáneamente;
  el registro de salida añade un ciclo y permite reemplazo en cada flanco.
  Esto mide el bloque aislado, no ancho de banda DDR/PCIe.
- Quince escenarios de fallo: respuestas huérfanas, IDs incorrectos, B antes
  de WLAST, RLAST prematuro/ausente, LEN de escritura incorrecto, overflow,
  WLAST ausente en la transferencia 256 y cambios/retiradas de B/R bloqueadas.
  Se verifica que las respuestas sanas retenidas permanecen o completan.

El banco observa ID/LEN/WLAST de solicitudes; no comprueba direcciones,
atributos de ráfaga, límites de página ni payload W. Esos campos requieren
los monitores existentes y la composición completa de la frontera.

## Controles negativos

`20261005T221604Z`, 79,384 s: tres copias RTL privadas, compiladas contra
el banco positivo intacto, producen exactamente las aserciones previstas.

| Defecto introducido | Detección |
| --- | --- |
| Reabrir R cuando llega un crédito posterior | Caso 7, 232.410 ns |
| Cortar B sana retenida al aparecer cualquier fallo | Caso 5, 231.876 ns |
| Aceptar R sin comprobar su RLAST | Caso 9, 232.886 ns |

## Evidencia y continuación

La síntesis `222016Z`, 133,501 s, mide **2.213 LUT / 2.588 FF**, incluyendo
24 LUT de memoria distribuida; sin BRAM/URAM/DSP, latches ni cajas negras.
Todos los registros tienen INIT=0. A 7 ns: WNS +3,768 ns, WHS +0,020 ns
y WPWS +2,968 ns. No hay endpoints internos sin reloj/restricción ni
excepciones temporales. Es un solo reloj: el informe CDC no valida un cruce.
Faltan retardos de 561 entradas/566 salidas y colocación/ruteo; no se aprueba
timing físico ni equivalencia funcional del netlist. Los dos avisos revisados
son granularidad de floorplanning y falta de `HD.CLK_SRC` en OOC.

Fuentes, compilaciones, trazas, revisores y manifiestos están respaldados
en `artifacts/ntrpu-board-vivado/<ID>/` y en el volumen NFS.

| Tanda, prefijo `20261005T` | SHA-256 del archivo |
| --- | --- |
| `221338Z-V4-ffn-supervised-axi4-response-guard` | `b4a7c30e22f943311937a6385b4bcfdda23be8c3a608211c1a57f8441ea9c629` |
| `221604Z-V4-ffn-supervised-axi4-response-guard-mutants` | `82239260eae7c8a32e316d195d83849c9cced21c6a7a74963f9349a47fe3a380` |
| `222016Z-V4-ffn-supervised-axi4-response-guard-synth` | `408497d53b878250cf4fa4885afad700be9812cbfeca7d8270dee9bbd9ebe8d6` |

Adenda de síntesis:
`82ed816e6a2c6810a34ca64fb74207e4540a09b9b3c10d3edb4cc9f25a5675d2`.

La [admisión retenida](AXI4_RETAINED_ADMISSION.md) ya incorpora este protector
con grupo finito AW/W/AR y rechazo local por ráfagas, validada de forma aislada.
El [contrato de integración](AXI4_ACTIVE_BOUNDARY_PLAN.md) conserva los pasos
pendientes con CDC real y certificados. La protección aislada no cierra `complete_drain`,
las colas internas XDMA, el reset global ni la implementación física.
