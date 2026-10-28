# Contrato: destinos constantes de la máscara RC

R10 es un candidato preparado, todavía sin simulación ni síntesis.
Parte de R8, la variante con mejor WNS OOC revisado. El productor R10
conserva íntegramente R8 salvo un comentario; el selector R5 sólo cambia
la escritura de la máscara de prefijos. Mantener `newest` como `integer`
de 32 bits para aislar este experimento del cambio de anchura de R9.

## Cambio y fundamento

Reemplazar el índice dinámico por destinos literales: bit 4 para el
header 1; bits 8–9 para el header 2; bits 12–14 para el header 3.
Conservar la selección del origen retenido, las condiciones de escritura,
la prioridad, la propiedad, la aritmética, RQ, capacidad y latencia.

El origen decodificado es −1 o el último SOP no posterior al EOP.
Un evento que termina antes del header destino sólo puede provenir de
un header anterior. El comprobador de posiciones agotó 1.024 casos
contra el selector R3 y cotejó la máscara `0x7310` con las 155.168 salidas
binarias de referencia. Ese modelo finito no califica la nueva RTL.

## Dependencias y aceptación

Antes de ejecutar, cerrar las cuatro tandas de R9 y sellar la cola R95.
Crear fuentes congeladas, proyectos y revisiones nuevos. Verificar por
comparación de texto que sólo cambió el bloque descrito.

Conservar todas las pruebas heredadas: 1.153.296 comprobaciones y
32.768 conjuntas de ascendencia, incluidos X/Z y los 840 bits completos;
selector, RQ, aritmética, suplementos de integración, banco principal,
mutantes y controles de evidencia. No recortar casos ni cambiar referencias.

Exigir síntesis VU5P FLVA2104 -2-i, `RuntimeOptimized`, 4 ns sin excepciones,
auditoría acíclica y replay de la propia netlist con los tres suplementos,
sin compilar RTL del DUT. Medir LUT/FF/CARRY8, setup/hold y ruta crítica
contra R7, R8 y R9 antes de elegir una variante. Mantener el inicio XSIM
verificado y respaldar cada tanda, incluidos fallos, localmente y en NFS.

Los criterios nativos, DDR de 512 GiB, inferencia real y diseño físico
de la [cola completa](VIVADO_QUEUE.md) permanecen abiertos.
