# Criterios para cerrar el inventario XDMA

Plan V4 posterior al [banco de parada e inventario](XDMA_STOP_INVENTORY.md).
Estos son criterios pendientes, no resultados. Se preservan la configuración
nativa y el Completion Timeout de 50 ms. Ninguna prueba de esta matriz
autoriza por sí sola reset global o reutilización de buffers.

La [primera observación de dos descriptores adyacentes](XDMA_CHAIN_STOP.md)
ya está revisada: dos descriptores entregados y 2 completados en este
ensayo. También está revisado el [retorno C2H de 64 bytes](XDMA_C2H_STOP.md),
con R retenida y recepción CQ comprobada. El [modelo BAR](HOST_BAR_CLOSURE.md)
comprobó cierre de admisión y obligaciones en dos generaciones. Siguen
pendientes cierre del driver real, inventario protegido e integración global.

El [coordinador de siete certificados](DRAIN_CERTIFICATE_JOIN.md) ya está
revisado de forma aislada y con mailboxes XPM. Las secciones reales, las
respuestas locales y los proveedores externos siguen la
[próxima integración](DRAIN_SECTION_INTEGRATION_PLAN.md). El resultado
no satisface todavía el inventario nativo completo.

## Fronteras que deben cerrarse

| Productor | Obligaciones observables | Evidencia exigida |
| --- | --- | --- |
| Software host | Nuevos descriptores, doorbells, accesos BAR | Cierre de envíos con generación identificada; no basta leer `Run=0` |
| H2C | Lecturas PCIe, descriptores y escrituras AXI | Contenido completo, respuestas pendientes y estado del canal |
| C2H | Lecturas AXI y escrituras PCIe | Todos los beats R y recepción de datos en el host; separar posted de non-posted |
| BAR bypass | AW/W/B, AR/R y completions PCIe | Inventario propio, independiente de los motores SGDMA |
| Secciones locales | Admisión, CDC y respuestas retenidas | Certificados AXI-Lite/AXI4 de la misma generación, sin alarmas |

Los contadores deben referirse al mismo intervalo de propiedad. El contador
de descriptores completados vuelve a cero cuando sube `Run`; no usarlo como
identificador permanente. Un descriptor precargado, una petición aceptada y
una transferencia terminada son hechos distintos.

El coordinador debe poder permitir el drenaje sin depender de una lectura
de estado bloqueada por ese mismo tráfico. El diagnóstico R5 observó esa
espera con una B de BAR retenida; el banco R6 la libera desde una rama
independiente. Trasladar esta ausencia de dependencia circular al diseño
de gestión requiere una prueba integrada; el temporizador del banco no es
un mecanismo de recuperación aprobado para la placa.

## Próximas tandas acotadas

1. **Cadena H2C de dos descriptores.** Datos y destinos distinguibles;
   primero no terminal y segundo terminal. Retener la B del primero,
   borrar `Run` por PCIe y verificar su lectura. Registrar cualquier
   fetch anticipado del segundo. Liberar B y reconstruir qué descriptor
   termina y qué inventario permanece. No confundir prefetch con escritura
   de datos ni dar por cancelado un descriptor sin evidencia.
2. **C2H con respuesta AXI retenida.** Cargar un descriptor nativo,
   aceptar AR y retener el último R con contenido estable. Leer control,
   estado y contador antes/después de parar y liberar. Comparar cada byte
   recibido en memoria del Root Port. El emisor H2C acotado actual no sirve
   automáticamente como receptor C2H: extenderlo y revisarlo por separado.
3. **Cierre del productor BAR.** Definir el protocolo software/hardware
   que impide nuevas operaciones y un punto de confirmación observable.
   Probar una escritura anterior al cierre y una petición concurrente.
   Una lectura cualquiera de un registro no se acepta como fence universal;
   documentar atributos, ordenamiento y alcance del mecanismo elegido.
4. **Integración de certificados.** Asociar cierre del host, canales XDMA
   y secciones AXI a una generación única. Inyectar pérdida de readiness,
   fallo retenido y certificado atrasado mediante interfaces públicas.
   Mantener propiedad bloqueada si falta una condición.

Cada tanda requiere carpeta nueva, límites de tiempo e hilos, fuentes
compiladas identificadas y revisión independiente de trazas, avisos y
contenido. Respaldar diagnósticos y positivos localmente y en NFS antes
de avanzar. No repetir compilaciones sin una hipótesis nueva verificable.

## Paso posterior a reset

Resolver en el [coordinador](ENDPOINT_RESET_COORDINATION.md) la liberación
del reset nativo y la posterior preparación de CDC sin dependencia circular.
La invalidación de propiedad, restauración de registros y tráfico posterior
deben demostrarse juntos. Una aserción corta de reset no sustituye la
secuencia completa ni su duración prescrita. DIMM exacto, cuatro canales,
restricciones físicas y ensayo en placa conservan sus dependencias V2–V9.
