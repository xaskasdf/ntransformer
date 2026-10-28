# Inventario PCIe del endpoint XDMA

El [monitor RTL](PCIE_REQUEST_INVENTORY_RTL.md) R5 ya pasó 22.219 ciclos
en 53 escenarios, con 256 tags, 257 lecturas y 1.052.672 bytes pendientes.
R4 tiene netlist funcional aprobado; su síntesis falla timing a 250 MHz
(WNS −11,964 ns). R5 limita los anchos temporales: 46.359 LUT/9.524 FF;
el timing sigue fallando y su netlist pasó los 52 escenarios binarios.
El DCP R5 reabierto confirma cero excepciones temporales. R6 tiene las
mismas métricas fallidas y pasó 67 escenarios de netlist. R7 pasó
68 escenarios RTL y coincide con R6 en los 22.339 estados emitidos;
su netlist pasó 67 escenarios. La síntesis R7 sigue fallando a 4 ns:
46.617 LUT/9.531 FF, WNS −11,267 ns; el peor camino no mejora.
Incluyen reproducción de esta captura;
la integración nativa del monitor conserva sus criterios pendientes.
[Recursos y timing](PCIE_REQUEST_INVENTORY_SYNTHESIS.md).
La [sonda de la frontera de producción](PCIE_INVENTORY_TIMING_PLAN.md)
terminó como `20261007T151433Z` sobre una copia, con 756 archivos originales
sin cambios. Confirmó los buses de datos/TUSER RQ/RC y dejó pendientes
READY[1:3] y el origen del BDF. Las inspecciones posteriores encontraron
esas salidas superiores ligadas a cero y localizaron ocho `CFGBUSNUMBER`
con nets propias en la primitiva `PCIE40E4`. Las tandas `164306Z` y
`165658Z` conectaron ocho salidas y verificaron su persistencia en el DCP,
con celdas y transporte intactos. Faltan muestreo funcional, conexión del
monitor y procedencia del Requester ID completo.
[Revisión de la frontera](XDMA_OBSERVATION_BOUNDARY.md).
Los avisos intermedios de R5 fueron
investigados: el grafo del netlist final es acíclico y el replay funcional
pasó. La conexión de producción sigue sin calificar.

## Captura nativa revisada — 2026-10-06

La tanda `20261006T230834Z` terminó en 597,049 s y pasó revisión
independiente: 266 señales, 77 entradas compiladas, 129 archivos AMD
sin cambios y 139 avisos conocidos. Archivo D:/NFS de 98.996.003 bytes:
`e51350b13a4a29a026ea53c134bf00e80822ac1ec3b6b580066bc2ecef54c90b`.
Informe `*-independent-pcie-inventory-review.json`:
`1eb64197d3223f12f1716ea726836d997abed949dda312b1c76d2122e4663213`.
Rechazó 65 corrupciones heredadas y 14 específicas de PCIe.

| Tráfico observado | Bytes | Solicitud → payload completo |
| --- | ---: | ---: |
| Descriptor H2C, host `0x100` | 32 | 336 ns |
| Datos H2C, host `0x400` | 64 | 344 ns |
| Descriptor C2H, host `0x200` | 32 | 336 ns |
| Escritura C2H, host `0x600` | 64 | 172 ns desde aceptación PCIe hasta recepción host |

Se correlacionaron tres tags (0/1/2), cuatro notificaciones de secuencia,
cinco beats RQ y cuatro RC con CQ/CC del Root Port y todos los bytes.
En H2C, el descriptor terminal llegó a 105.456 ns y el payload terminó
a 105.460 ns: durante **4 ns** la tabla de tags estaba vacía, pero el
modelo conservó una obligación RC. El máximo nativo fue un tag/64 bytes;
no hubo stalls ni varios SOP en un beat. La saturación de 256 tags y
RQ2/RC4 está probada en software, RTL aislado y netlists binarios R2/R3;
todavía no tiene evidencia con esa carga en el endpoint nativo.

La traza expuso dos condiciones que se incorporaron al verificador:
bits desconocidos en posiciones RC fuera del paquete, y Requester ID
efectivo `0x01a0` aunque RQ presenta cero. El XCI compilado tiene
`ARI_CAP_ENABLE=FALSE`: la función ocupa tres bits; el ID completo de
la completion debe seguir coincidiendo. El lector VCD conserva máscaras
de bits conocidos; rechaza desconocidos en campos válidos y payload
habilitado. El conjunto local pasa **63 pruebas**, seis de ellas sobre
trazas de cuatro estados. Informe de pruebas:
`pcie-inventory-reference-tests-20261006T232320Z.json`, SHA-256
`051cc84f364c3381877d1facb7d6d6f019169b397896c9269d91dedcf4cc887f`.

Son tiempos de simulación y un modelo de observación, no rendimiento
medido en una placa. Permanecen pendientes el monitor sintetizable,
su conexión física, cierre real del productor y drenaje global.

## Perfil real que debe soportar el observador

La auditoría `xdma-pcie-generated-profile-review.json` revisa el XCI y
las declaraciones compiladas del respaldo `20261006T223045Z` con Vivado
2026.1. SHA-256 del informe:
`b22da53880b4ba4053d612ba093cec2a161fee0fd00e6d990ec41beef0d2ad7c`.

| Propiedad generada | Valor exigido |
| --- | --- |
| Datos RQ/RC | 512 bits |
| TUSER RQ/RC | 137/161 bits |
| Asignación de tags | Cliente; tags extendidos habilitados |
| RQ straddle | Habilitado; hasta dos paquetes por beat |
| RC straddle / 4TLP | Ambos habilitados; hasta cuatro paquetes por beat |
| Alineación RQ/RC | Dword |
| CQ/CC straddle | Deshabilitado |
| ARI en la captura nativa | Deshabilitado; función PCIe de tres bits |

Los puertos de asignación de tags del core están sin conectar en este
perfil. Asociar cada solicitud con el tag de su descriptor; no esperar
un pulso de asignación del core. La aplicación debe evitar tags duplicados
y exceso de solicitudes pendientes cuando los asigna ella misma.
[PG213: gestión de tags](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Tag-Management-for-Non-Posted-Transactions).

## Frontera de observación e integración física

El candidato de captura en simulación es
`/board/EP/ntrpu_xdma_i/inst/pcie4_ip_i/`. La auditoría estática identifica
esa instancia y sus conexiones; la tanda anterior confirmó su visibilidad
en XSIM. Se observaron los puertos declarados RQ/RC, reloj/reset/link y
secuencias de transmisión, sin modificar ni forzar código AMD.

Son puertos públicos del subcore PCIe **internos al wrapper XDMA**.
Una referencia jerárquica de simulación no demuestra una conexión
sintetizable para el monitor. Antes de autorizar drenaje, resolver una
frontera de integración soportada que exponga estos contratos al diseño.
Conservar este requisito explícito aunque la captura y el modelo pasen.

Una vía candidata es conectar entradas de observación en una copia del
netlist sintetizado: Vivado permite `connect_net -hierarchical`, creando
los pines intermedios necesarios y guardando el resultado en un DCP nuevo.
Esto es una hipótesis de integración, todavía no validada para este IP.
Primero comprobar disponibilidad de los nets/pines reales y restricciones
del DCP; después comparar conectividad original, ausencia de drivers
añadidos al transporte, simulación, timing y CDC. No reemplazar señales
del core ni modificar el checkpoint original.
[UG835 2026.1: connect_net](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/connect_net).

## Contabilidad requerida

Procesar únicamente handshakes y conservar todo beat ofrecido mientras
READY está bajo. RQ puede iniciar dos paquetes por ciclo, con SOP en
lanes 0/32. Reconstruir SOP/EOP desde TUSER, incluidos paquetes que
continúan en el siguiente beat. TLAST/TKEEP no delimitan los paquetes
cuando straddle está habilitado. Los números de secuencia acreditan
avance de transmisión, no recepción en memoria host.
[PG213: interfaz RQ](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Request-Interface),
[straddle RQ](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Straddle-Option-on-RQ-Interface).

RC admite hasta cuatro SOP/EOP por beat, con inicios en lanes
0/16/32/48. Contar bytes de payload habilitados y conservar cualquier
paquete parcial. Un descriptor RC tiene 12 bytes; cada split completion
trae su propio descriptor. Vincular tag, función, dirección y bytes
pendientes. `Request Completed` también puede indicar terminación con
error: no convertirlo en aprobación. Un timeout deja inválidos muchos
campos; conservar la identidad válida y el fallo.
[PG213: interfaz RC](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Completion-Interface),
[descriptor RC](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Completion-Descriptor-Format).

## Orden de tandas y aceptación

### Referencia ejecutable para el monitor RTL

`tools/ntrpu_pcie_straddle.py` reconstruye paquetes y
`tools/ntrpu_pcie_request_inventory.py` conserva solicitudes por tag.
Son modelos de revisión en software, todavía no proveedores sintetizados.
Ejecutar `python -m unittest discover -s tests -p 'test_ntrpu_pcie*.py'`.
Los vectores incluyen 256 tags simultáneos, RQ2/RC4, reordenamiento,
dos splits del mismo tag en un beat, 4 KiB en 65 beats, backpressure,
reutilización del tag con payload anterior pendiente y fallos retenidos.

Separar en RTL la tabla de tags del contexto de payload RC: un descriptor
terminal libera el tag, pero el paquete parcial conserva su obligación
hasta EOP. Procesar retiros RC anteriores a asignaciones RQ del mismo
ciclo sin asociar una completion desconocida a una solicitud recién creada.
El prototipo debe soportar dos asignaciones y cuatro descriptores RC por
ciclo; reducir esos anchos para conseguir un banco verde no cumple el perfil.
Conservar por solicitud identidad efectiva, dirección siguiente, bytes
pendientes y fallo; comparar primero con el modelo, después con el netlist.

El span de completion incluye los huecos entre byte enables: `first_be=0101`
en una lectura de un DWORD necesita Byte Count 3, aunque solo dos bytes
sean datos solicitados. Una lectura de longitud cero requiere el tratamiento
especial de un byte contado y un DWORD ficticio. La tabla 3-10 de
[PG213 v1.1, edición histórica](https://docs.amd.com/api/khub/documents/1GUqHE02aRV~1PIVnWXyTw/content)
explicita el cálculo; el
[descriptor RC actual](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Requester-Completion-Descriptor-Format)
define bytes restantes y la reutilización del tag.

El modelo recibe explícitamente el Requester ID efectivo, el modo ARI y la
máscara de atributos. La revisión nativa los contrasta con CQ del Root
Port; el ID crudo de RQ no contiene necesariamente el bus asignado por el core.
Vaciar esta tabla no demuestra entrega de escrituras posted ni cierre del
productor. Esas obligaciones y la frontera física permanecen separadas.

### Dependencias de ejecución

1. Cerrar y revisar C2H/DMA/BAR; conservar también su diagnóstico inicial.
2. Capturar la frontera PCIe nativa y correlacionarla con los tres fetches
   esperados, datos H2C, emisión C2H y bytes aceptados por el Root Port.
   Confirmar el tráfico observado; no inventar eventos ausentes.
3. Implementar decodificación y contabilidad sintetizables para el perfil
   completo: dos RQ/cuatro RC, continuación, splits, tags simultáneos,
   reordenamiento entre tags, saturación, backpressure y errores.
4. Comparar RTL/netlist y trazas nativas. Ejercitar los límites completos
   con estímulos adicionales aunque el banco corto use pocos paquetes.
5. Resolver la conexión de producción, el cierre real del productor y
   la recepción host; transferir identidad y revocación coherentes al
   join de siete contratos. Un inventario vacío aislado nunca da ACK.

Registrar máximo de obligaciones, bytes pendientes y tiempos entre
aceptación, completion, Done y recepción host. Cada tanda conserva fuentes
congeladas y respaldo D:/NFS. Mantener Completion Timeout de 50 ms.
DIMM exacto, cuatro DDR/512 GiB, pesos reales y cierre físico siguen en
la [cola completa](VIVADO_QUEUE.md).
