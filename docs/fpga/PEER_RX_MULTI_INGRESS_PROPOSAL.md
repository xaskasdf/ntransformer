# Propuesta de dos entradas independientes de payload

Fecha: 2026-10-07. Estado: **propuesta con un prototipo de recepción separado**.
El [prototipo de doble entrada](PEER_RX_DUAL_INGRESS_PROTOTYPE.md) ya tiene
simulación comparativa y síntesis; la integración completa sigue pendiente.
Extiende el contrato estudiado en
[transporte HBM](PEER_RX_HBM_NATIVE_TRANSPORT.md), conservando su referencia.

## Objetivo y presupuesto de ancho de banda

Separar el llenado por destino para que dos productores puedan entregar datos
simultáneamente. Hoy `fetch`, `payload_data[511:0]`, el presupuesto de beats y
`fill_active` son compartidos; duplicar descriptores o destinos no elimina esa
serialización.

| Frontera | Configuración | Techo aritmético, por dirección |
|---|---|---:|
| Entrada actual | 1 × 512 bits, 200 MHz | 12,8 GB/s |
| Entrada propuesta | 2 × 512 bits, 200 MHz | 25,6 GB/s agregados |
| Dos puertos AXI de HBM | 2 × 256 bits, 450 MHz | 28,8 GB/s agregados |

Son techos de las interfaces, antes de pausas, arbitraje y latencia; no son
resultados medidos. El presupuesto PCIe depende del enlace negociado y de
su camino DMA. Dos colas alimentadas por un único puerto físico de 12,8 GB/s
siguen compartiendo ese límite.

HBM ofrece puertos AXI3 de 256 bits y dominios de reloj independientes.
La primera integración conservaría AXI00/AXI16, procurando destinos locales
a cada stack; añadir puertos requiere revisar distribución, arbitraje y
congestión. [AMD PG276: arquitectura](https://docs.amd.com/r/en-US/pg276-axi-hbm/Core-Overview).

## Contrato propuesto

Mantener dos destinos, inicialmente fijos por entrada, y el pool de ocho
bancos: dos propios por destino y cuatro reservas compartidas. Cada entrada
tendría `fetch_valid/ready/desc`, `payload_valid/ready/data[511:0]`, contador de
beats y estado de llenado propios. Solo se acepta payload perteneciente al
descriptor reservado por esa entrada; no se intercala otro marco en ella.

El arbitraje de **nuevas reservas** puede seguir centralizado y conceder un
banco por ciclo; ambas entradas continúan llenando bancos ya concedidos
en paralelo. Ninguna concesión puede duplicar banco ni generación. Con
marcos de 4 KiB, el techo agregado exige una reserva cada 32 ciclos de
200 MHz en promedio; los marcos cortos y las pausas de devolución necesitan
medición específica.

Separar tres momentos: último payload aceptado, último W interno aceptado
y finalización externa. El último W permite devolver el banco cuando las
colas siguientes poseen todos sus datos; B y la finalización siguen vivos
en sus respectivos registros. Una generación antigua nunca puede liberar
una reserva nueva. Conservar el corte de devolución validado y su cierre
de concesiones durante captura/validación.

El orden queda definido por entrada. Un fence global cierra ambas admisiones,
deja drenar los dos llenados y espera las finalizaciones externas anteriores
antes de responder. `stop` conserva el trabajo aceptado; un fallo no autoriza
reciclar bancos que aún tengan dueño. No se añade reinicio en caliente.

## Memoria de buffers: decisión que hay que medir

La referencia compacta replica un pool para dos lectores y difunde una
única escritura. Agregar un segundo escritor a ese array no produce
automáticamente RAM inferible con el caudal deseado.

Prototipo propuesto: bancos físicamente separados, cada uno con propietario
exclusivo. Dos entradas pueden escribir en bancos distintos; por banco se
selecciona un único escritor y un único destino lector. Registrar selección
y datos para evitar un multiplexor global en el camino crítico. Mantener
los bancos fijos por destino cerca de su salida y medir el coste de las
cuatro reservas, que deben poder servir a cualquiera.

Comparar esta organización con dos subpools y reservas transferibles solo
estando libres. Esa alternativa reduce conexiones, pero puede dejar capacidad
libre inaccesible bajo carga desigual. La elección necesita síntesis, BRAM/LUT/FF,
congestión, temporización y pruebas de equidad; no se adopta por intuición.

Los ocho bancos suman **32 KiB lógicos**; las cuatro reservas, 16 KiB.
La memoria física puede ser mayor por anchura, profundidad mínima, replicación
y registros. Si ambos productores llegan a 25,6 GB/s y se detiene toda la
salida, esas cuatro reservas vacías absorben como máximo aproximadamente
**0,64 µs** de datos adicionales. Fragmentación y reservas ya ocupadas reducen
esa ventana. Dimensionar por la pausa medida y el caudal neto acumulado,
no por consumir toda la BRAM disponible.

## PCIe, BAR y múltiples NTRPU

El controlador de apoyo publicaría regiones por dispositivo y repartiría
pesos por rangos o bloques. Las transferencias se vincularían a
`{dispositivo, región, offset, longitud, época, tag}`; comprobar límites y
permisos antes de reservar recursos. BAR expone ventanas de dirección:
la unificación de capacidad pertenece al runtime y al protocolo de
transferencias, y no acredita coherencia automática entre tarjetas.

Propuesta inicial: regiones de pesos inmutables durante una época, copia
o partición explícita, publicación de disponibilidad tras una finalización
definida, e invalidación coordinada antes de remapear o retirar una región.
La finalización AXI local no basta para certificar visibilidad en otro
endpoint: ese contrato requiere pruebas de orden del camino PCIe completo.

P2P necesita validar topología y encaminamiento ACS, mapeo DMA y vida útil de
los recursos entre ambos drivers. El kernel evalúa caminos compatibles y
restringe cruces de jerarquías no acreditados; no asumir que cualquier par
de ranuras sirve. [Documentación Linux P2PDMA](https://cdn.kernel.org/doc/html/latest/driver-api/pci/p2pdma.html).

## Criterios para avanzar

1. Completar [presión sobre reservas](PEER_RX_HBM_RESERVE_PRESSURE.md),
   distinguiendo sustituto de memoria y controlador HBM nativo.
2. Implementar un prototipo separado con dos llenados y RAM inferible;
   conservar la referencia de una entrada para comparación.
3. Probar simultaneidad de aceptación, pausas asimétricas, agotamiento de
   reservas, equidad, marcos cortos/largos, stop, fence y generaciones obsoletas.
   Con un destino detenido, el otro debe progresar mientras conserve recursos.
4. Medir bytes aceptados y completados por ventana, stalls por causa,
   ocupación por canal, espera de reserva y latencia de finalización. Comparar
   contra una entrada con el mismo reloj, volumen, patrón y presupuesto de RAM.
5. Pasar síntesis y temporización; repetir con CDC/conversores y HBM nativos.
   Integrar PCIe/P2P solo después, con lectura de retorno y evidencias propias.

Esta propuesta no cambia `board/VIVADO_QUEUE.md`, las pruebas del agente
principal ni el RTL ya calificado.
