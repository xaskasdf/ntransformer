# Inventario PCIe: validación y publicación alineadas

Esta revisión integra el productor canónico inicial con los chequeos restantes
de RQ/RC. Es un candidato funcional pendiente de ejecución, síntesis y replay de
su propia netlist. No habilita conexión al endpoint ni acredita timing físico.

## Datos y orden

`ntrpu_pcie_inventory_join.sv` recibe únicamente señales públicas RQ/RC,
configuración y secuencia diagnóstica. Instancia el productor inicial
`ntrpu_pcie_inventory_scheduler.sv`; no utiliza la variante de timing regresivo.
El productor conserva la tabla de 256 tags, contadores y contextos parciales.
No recibe decisiones ni estado del validador o de la referencia del banco.

En el flanco N se capturan el beat, la configuración retenida y los contextos
anteriores del propio productor. Sus respuestas canónicas registradas contienen
propiedad, dirección, bytes restantes, atributos y bytes entregados. El validador
comprueba identidad, longitud, terminal, enables y payload usando esos datos.
Se conserva el orden RC antes de RQ, incluida la reutilización del tag mientras
el payload terminal anterior sigue abierto.

En N+1 se publican conjuntamente el estado posterior a N y todos sus fallos
acumulados. La secuencia de 32 bits identifica el ciclo publicado; su vuelta a
cero no cambia propiedad. También se publican ciclos inactivos y de armado.
Cold reset cancela la publicación pendiente y limpia ambos caminos. Los resets
del enlace conservan las obligaciones y producen fallo.

## Vacío y aceptación

La actividad entrante retira `inventory_empty` antes del siguiente flanco.
También lo bloquean estado pendiente, validación activa, fallos, cambios de
configuración y handshakes inválidos. Los ciclos quietos de diagnóstico no
mantienen ocupado el validador indefinidamente. No existe ACK de drenaje global.

El banco debe comparar todas las filas, incluso posteriores a errores, contra
la referencia archivada: estado, bitmap, fuentes completas de fallo, secuencia,
reset y recuperación de vacío. Debe conservar RQ2/RC4, 257 lecturas y
1.052.672 bytes máximos, sin pausas del observador. Exigir mutantes reales y
controles de corrupción de evidencia antes de autorizar síntesis.
