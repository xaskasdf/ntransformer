# Siguiente tanda: fallos W reenviados y capacidad de lectura

La matriz de reenvío ya está ejecutada y revisada en
[resultados R2](AXI4_FORWARDED_FAULTS.md): ocho casos y un control negativo.
Se conserva abajo el contrato de la tanda. El análisis de alcanzabilidad
de capacidad defensiva continúa pendiente de demostración formal.

## Alcance y entradas

La matriz de [solicitudes locales](AXI4_SECTION_REQUEST_FAULTS.md) cubre el
estado `REJECT`. La siguiente ampliación debe observar fallos W mientras
la admisión aún reenvía en `RUN` o termina su prefijo en `FINISH`.
Usar los módulos candidatos R2 y un banco nuevo; conservar las fuentes y
revisiones anteriores. No reabrir admisión ni resetear inventarios ante error.

## Casos que faltan

1. Aceptar AW de cuatro beats y terminar W en el segundo. Probar AW antes
   de W y W antes de AW: el emparejamiento debe descubrir la diferencia
   en ambos órdenes, con una respuesta sana independiente retenida.
2. Aceptar AWLEN=255/AWSIZE=4, dirección alineada a 4 KiB, y 256 beats sin
   WLAST. Verificar los strobes que rotan por los cuatro grupos de 16 bytes.
   El beat causante debe aparecer en la traza; no confundir congelación
   del inventario con descarte de una transferencia aceptada.
3. Iniciar el cierre con W parcial y completar el prefijo con longitud
   incorrecta. No exigir `REJECT` ni inventario cero: un fallo puede dejar
   `FINISH` retenido indefinidamente. El certificado debe permanecer revocado.

La admisión puede bloquear nuevos beats al registrar el fallo, pero una
oferta ya presentada conserva VALID y payload hasta su aceptación.
El comprobador debe contabilizar ese beat adicional si existe. Examinar
por separado fuente S, entrada H y salida D del CDC: sus inventarios pueden
congelarse en flancos distintos. No imponer conservación de contadores
internos después de que se declaren desconocidos; sí reconstruir cada
transferencia externa y la causa de cada alarma.

Usar H/D=7/5 y 7/37 ns y gestión a 10 ns. Detener un reloj solo después
del registro de su alarma, preservar B/R sanas y probar una identidad nueva.
No forzar flags internos. Añadir un control negativo privado para el
mecanismo nuevo que se pretenda calificar, sin repetir controles ya cubiertos.

## Capacidad defensiva de lectura: análisis estático

En `ntrpu_axi4_response_guard_r2.sv`, `read_bad` requiere aceptar AR con
`read_pending[arid]==READ_DEPTH` sin retirar simultáneamente una R final
del mismo ID. La admisión conecta `ar_capacity` de ese mismo guard y ID.

La lectura del RTL sugiere el siguiente invariante, todavía sin prueba formal:

- Sin oferta retenida, `permit_ar` requiere capacidad menor que el límite.
- Una oferta retenida nació con espacio y ARREADY bajo. Mientras espera,
  ninguna otra AR atraviesa ese único canal; el contador de su ID solo
  puede mantenerse o disminuir por respuestas consumidas.
- `saved_ar` conserva el ID ofrecido aunque la fuente cambie indebidamente
  su payload. Esa infracción activa otra alarma, sin cambiar el descriptor
  ya presentado al guard.

Por ello no hay todavía un estímulo externo legítimo identificado que
active este desbordamiento dentro de la composición. No sustituir esa
carencia por una escritura jerárquica ni etiquetar el caso como cobertura
funcional. Para cerrar su alcanzabilidad, demostrar inductivamente el
invariante con la inicialización, los grants retenidos y las respuestas
simultáneas, o conservar un contraejemplo externo reproducible. Los tests
aislados del guard aceptan estímulos que la admisión puede impedir.

## Criterios de cierre de la tanda

Mantener descriptores dentro de una página de 4 KiB y datos/strobes completos.
Exigir diagnóstico causal, ninguna respuesta fabricada, ofertas sanas
retenidas, certificado rechazado y pendiente terminal explícito. Registrar
límites del modelo FIFO AMD, hashes de bibliotecas/fuentes y respaldos por ID.
Estos casos no cierran el inventario protegido XDMA, CDC-15, reset global,
DIMM exacto ni implementación física de la placa.
