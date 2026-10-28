# Cancelación de certificados AXI4 en tránsito

## Revisión posterior del índice inicial

La ampliación con fallos antes/durante el certificado encontró un índice
indeterminado al arrancar en la simulación RTL original. Se conserva la
evidencia previa con su alcance; no prueba todas las secuencias iniciales.
La revisión candidata RTL R2 y el banco R3 corrigen el índice y una espera
prematura del banco, y pasan la matriz ampliada y la regresión anterior.
La síntesis previa pertenece a los módulos originales. Ver
[diagnósticos, validación y siguientes pasos](AXI4_SECTION_FAULT_LIFETIMES.md).

## Resultado

`20261006T005042Z-V4-ffn-supervised-axi4-section-cancel` pasa cuatro
cancelaciones/reemplazos por perfil, con Clock Converter AMD real y relojes
H/D de 7/5 y 7/37 ns; gestión a 10 ns. El banco y comprobador son revisiones
nuevas. Los seis módulos RTL de producción conservan sus hashes sellados.

Cancelar una solicitud suprime su certificado; no retira una transferencia
ya entregada al mailbox. El mensaje antiguo debe terminar y liberar el
handshake antes de enviar otro. Una confirmación antigua nunca acredita
la identidad nueva, aunque el reloj del receptor se detenga.

## Casos comprobados

| Mailbox | Situación al cancelar | Reloj detenido | Espera medida después de cancelar |
| --- | --- | --- | ---: |
| Admisión | Prueba entregada; quedan dos beats de W y respuestas normales | H | 155 ns |
| Admisión | Nueva identidad emitida, todavía no entregada al receptor | H | 275 ns |
| Par host | Admisión certificada, mensaje del par pendiente | H | 155 ns |
| Par dispositivo | Admisión y par host certificados; mensaje D pendiente | D | 155 ns |

Son intervalos impuestos y observados en este banco, no tiempos máximos
de recuperación. Se mantienen la identidad antigua y `src_send` mientras
falta su confirmación. El reemplazo usa un tag nuevo; el primer incremento
modifica también sus 32 bits altos. Las cuatro finalizaciones obsoletas se cuentan como 2/1/1
en admisión/host/dispositivo y no producen certificados válidos.

Cada perfil conserva 24 escrituras y 24 lecturas normales, 1.045 beats W
y 1.045 R, IDs reutilizados, W anterior a AW, atributos completos, strobes,
backpressure y conservación de datos entre S/H/D. Después mantiene B y R
locales de una operación de 16 beats a través de las cancelaciones.
Siguen pasando tres certificados frescos, el reset solapado del CDC y la
contención de una R huérfana posterior al certificado.

El aviso simulado D→gestión tarda 21,9 ns en ambos perfiles; admisión→gestión,
27,0 y 26,5 ns. El solapamiento medido de reset es 450,1 ns y 2.370,4 ns.
Cada transición ocurre después de un flanco local observado y se conservan
al menos 64 períodos por dominio. Son medidas de simulación, sin homologación
de reset físico ni caudal real: el modelo FIFO AMD omite retardos de CDC de datos.

## Revisión y controles negativos

`check_ntrpu_axi4_section_cancel.py` conserva la reconstrucción independiente
de transacciones del banco R2 y añade observación de los tres mailboxes en
cada flanco de gestión. Comprueba identidad actual, estabilidad del mensaje
emitido, ocupación durante la detención y ausencia de certificados obsoletos.
La auditoría rechazó **92 alteraciones de evidencia**, incluyendo payload,
strobes, RLAST, resets, cancelaciones omitidas, identidad, `src_send`, ACK
obsoleto y muestras faltantes.

`20261006T005623Z-V4-ffn-supervised-axi4-section-cancel-mutants` ejecuta el
mismo banco sin modificarlo, con dos versiones privadas defectuosas del
mailbox, en D=5 ns:

- Retirar `src_send` al cancelar: falla la aserción de ocupación; la traza
  muestra la retirada sin confirmación en 15,915001 µs.
- Sobrescribir el mensaje pendiente al reemplazar: la traza demuestra el
  cambio sin confirmación en 15,995001 µs. La aserción final detecta además
  que no se registraron las cuatro finalizaciones obsoletas esperadas.

Ambos primeros contraejemplos aparecen en el mailbox de admisión. Esto no
añade una prueba negativa independiente de cada instancia del par.

## Evidencia y siguientes dependencias

Ejecuciones, fuentes y adenda están verificadas localmente y en NFS:

- Banco: `6d8b93afc267fb720ad5061f0d859f090713663d104432df59c963e4101d226b`.
- Adenda: `6a87ef53bdef3786a0a6a02714fced9a493a482ff5bdfa4d53e9904c5c5a1978`.
- Defectos RTL: `49ffeaa749b419b7f7dc1699b2186a8fa0bed16f4fb6534cf662919152bca0d4`.

Las tandas usan dos hilos y límite de 600 s. Se permitió coexistir con el
ruteo de otro estudio tras verificar 30 CPU, unos 215 GiB disponibles y su
límite de dos hilos. No se modificaron ni detuvieron sus procesos.

Faltan fallos externos reales anteriores/durante los certificados, otras
causas alcanzables de fallo de admisión y reemplazo de **generación** con
mailbox ocupado; aquí los reemplazos ocupados cambian el tag. La generación
sí cambia en la prueba posterior con mailboxes libres. Permanecen abiertos
CDC-15 nativo, inventario protegido XDMA, reset global y diseño físico.
Ver [síntesis y límites](AXI4_SECTION_SYNTHESIS.md) y [cola](VIVADO_QUEUE.md).
