# Cierre del productor BAR: contrato del banco acotado

## Estado y contrato propuesto

**Banco nativo revisado y respaldado:** [resultados](HOST_BAR_CLOSURE.md).
No acredita un fence del host ni permite reset. El banco modela un
único productor software con generación
identificada, cierre de admisión y registro de operaciones pendientes.
Su certificado tendrá alcance de banco; trasladarlo al driver requiere
sincronización entre hilos y validación con el sistema operativo y hardware.

Todas las solicitudes del productor deben pasar por una única función de
admisión. El cierre fija la generación y el último número admitido de forma
atómica respecto de nuevas solicitudes. Las rechazadas no pueden llamar al
emisor PCIe. Las admitidas conservan su obligación hasta recibir evidencia
de terminación; cerrar envíos no las borra. No reutilizar generación o
memoria por desbordamiento de contadores ni por vencimiento de un timer.

## Banco nativo mínimo

1. Enumerar el ejemplo inmutable y registrar dirección, TC, atributos y
   Requester ID efectivos. Mantener detenidos ambos canales SGDMA.
2. Admitir una escritura BAR de un DWORD con datos distinguibles. Retener
   su B pública; reconstruir RQ, AW y W. Cerrar el productor mientras esa
   obligación sigue pendiente y comprobar que aún no puede certificarse.
3. Intentar una solicitud concurrente y otra posterior al cierre. Registrar
   su resultado y verificar en las señales nativas qué solicitudes fueron
   realmente emitidas. Fijar el orden de la carrera en flancos separados;
   cubrir también la admisión inmediatamente anterior al cierre.
4. Liberar B desde una rama independiente. El avance del drenaje no puede
   depender de que termine una lectura de gestión bloqueada por esa B.
5. Obtener un readback del mismo destino, con una política de ordenamiento
   explícita. Comparar el dato, RQ/RC, AXI y el registro de obligaciones.
   Exigir generación coincidente y cero operaciones pendientes antes del
   certificado. Observar después que no aparece tráfico nuevo.

El readback solo respaldará el caso y la ruta ensayados. Un registro de
control cualquiera, una marca del emisor o `Run=0` no constituyen un fence
universal. Probar por separado certificado atrasado, generación incorrecta
y respuesta incompleta, manteniendo la propiedad bloqueada.

## Ordenamiento y límites

PG213 permite determinadas superaciones entre clases de TLP y responsabiliza
a la aplicación del orden después de entregar el paquete a su interfaz.
El revisor debe conservar los atributos reales, incluido Relaxed Ordering.
[Orden de recepción](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Receive-Transaction-Ordering).

El retorno de un número de secuencia RQ identifica avance dentro de la
tubería transmisora; no demuestra por sí solo recepción en memoria host.
No sustituir evidencia de destino por ese evento.
[Orden de transmisión](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Maintaining-Transaction-Order).

La integración posterior debe unir el cierre de envíos, inventario XDMA y
certificados AXI de la misma generación. Inventario protegido, reset global,
CDC, driver real, DDR y placa conservan sus criterios pendientes.
