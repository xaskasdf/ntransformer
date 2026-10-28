# Alarmas exclusivas y controles de contención AXI-Lite

## Resultado

La composición R3 y el certificado R5 permanecen sin cambios. La tanda
`20261005T203519Z-V4-ffn-supervised-guard-faults` pasa **24 escenarios** en
150,153 s: seis casos por combinación de dispositivo 5/37 ns y host
activo/detenido. Usa manager, Clock Converter y XPM AMD reales.

Cuatro casos inyectan una B/R huérfana en la salida cruda del CDC, antes o
después de obtener el certificado. El protector es la única fuente de alarma:
admisión/H/D permanecen sanos. Otros dos inyectan una aceptación B/R inválida
en D y comprueban su paso por el CDC hasta la cuarentena. Estas inyecciones
deliberadas no implican que la IP emita respuestas huérfanas espontáneamente.

## Reloj detenido, latencia y conservación

En la variante detenida, el banco congela host **1 ps después de que suba
el registro real de fallo**. La revisión cuenta cero flancos host entre
esa detección, el reporte y la solicitud de una identidad nueva. Gestión
continúa a 10 ns; la alarma llega en **20,8–29,8 ns**. En los casos con
certificado previo, la revocación coincide con ese reporte.

Las trazas distinguen D, respuesta cruda y manager protegido. La oferta
huérfana conserva datos y respuesta con READY=0; no hay handshake huérfano
hacia el manager. Se reconstruye también el readback enmascarado de cada
caso, con una escritura/lectura legítima en U/G/H/D. El reset solo aparece
durante configuración. La alarma de gestión persiste al retirar los reportes
de fuente y no admite nuevos mailboxes ni reapertura.

Los archivos compilados, parámetros IP y 86 archivos de biblioteca se
comparan con la referencia sellada. Ocho avisos conocidos de entorno/glbl;
ningún aviso de ejecución. Se rechazan 36 alteraciones de evidencia.

## Controles negativos

| Mutación privada | Detección por banco positivo sin modificar |
| --- | --- |
| B evita el filtro | Caso 0 del banco con IP real, 2.790.200 ps: una respuesta huérfana no puede llegar al manager |
| Se elimina la cuarentena B | Caso 4 aislado, 84 ns: se perdió inventario pendiente; la traza muestra la B inválida aceptada a 31,5 ns con cuarentena ya registrada |
| Se corta R sana por un fallo B | Caso 1 aislado, 59.500 ps: falla la conservación de R retenida |

La tanda `203858Z-V4-ffn-supervised-guard-containment-mutants` conserva
código de salida 1: el segundo control falló antes, en otra aserción válida,
y el Tcl detuvo la tanda antes del tercero. La adenda aprueba **solo los dos
controles ejecutados**, sin convertir la tanda en una ejecución completa.
El tercero se ejecuta por separado en
`204343Z-V4-ffn-supervised-guard-healthy-r-mutant` y pasa su revisión.
No se repiten los controles ya cubiertos ni se alteran fuentes de producción.

## Respaldo y siguiente dependencia

Prefijo completo de los IDs: `20261005T`; ubicación local
`artifacts/ntrpu-board-vivado/<ID>/`, con réplica y verificación de miembros
en NFS.

| Tanda | SHA-256 del respaldo |
| --- | --- |
| `203519Z` | `a17a96719b8d012abe265695d2625eb780adc486c5160f0cfdbc8bf8938f1c34` |
| `203858Z` parcial | `5f512e516ff3f8ae8b51c18a9ef8ef1f8cfac8833ee7897c5d7baee1ff13d9f1` |
| `204343Z` | `bfb1669c58c4fe4564d9f372e656975dbd0bbcc4a85466cf8812d6b156e05eea` |

Adenda parcial: `7ebc16e51d8264124e5869622198f5c07315c03bcfad4845e3d38fb6fdd83b74`.

La etapa de fallos de esta sección queda cubierta por la evidencia acumulada;
sus recursos permanecen en 802 LUT/1.762 FF, según la síntesis anterior.
La siguiente dependencia es el [inventario retenido AXI4](AXI4_DRAIN_PLAN.md).
No se acredita drenaje completo XDMA, reset global, reutilización NTRD ni
implementación física de la placa.
