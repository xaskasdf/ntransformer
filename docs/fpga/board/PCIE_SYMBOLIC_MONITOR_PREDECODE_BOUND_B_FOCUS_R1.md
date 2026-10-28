# Límites registrados de payload: banco de B real

## Motivo y alcance

La [síntesis RQ 984](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_SYNTHESIS_RESULTS_R1.md)
sitúa la peor ruta en B→R: −2,768 ns estimados y 27 niveles lógicos.
El [borrador de límites registrados](PCIE_SYMBOLIC_MONITOR_PREDECODE_REGISTERED_BOUND_DRAFT_R1.md)
adelanta cinco sumas `address[1:0] + delivered` al mismo flanco que B,
sin añadir una etapa. Este banco califica únicamente ese productor.

Instanciar el B original y el borrador existente, con entradas A públicas.
Comparar los 2.892 bits de B y calcular los cinco límites esperados mediante
sumas de 32 bits sobre la salida capturada de la referencia. Los 75 bits
adicionales deben permanecer estables entre flancos y corresponder al
mismo registro. No escribir estado interno de los módulos.

## Cobertura prevista

Exigir **211.606 filas**, comparando 2.967 bits antes/después del flanco:

- 131.072 escenarios aritméticos: los 2.048 valores de words, cuatro
  alineaciones y dieciséis valores de remaining próximos a fronteras.
- 65.536 combinaciones de siete controles 0/1/X/Z y cuatro perfiles de
  operandos, incluidos unknowns activos y enmascarados.
- 10.892 variaciones: cada bit de A en 0/1/X/Z; 4.096 registros
  pseudoaleatorios deterministas, cinco filas dirigidas y cinco de cierre.

Conservar codificaciones malformadas y probar el límite 8.194; no recortar
el dominio a tráfico válido. Comprobar reset frío en 0/1/X/Z. Seis fuentes
defectuosas deben producir testigos exactos: usar B anterior, rotar queries,
omitir offset, truncar a trece bits, borrar X y omitir reset. El revisor
debe rechazar diecisiete corrupciones de evidencia.

## Cola, respaldo y siguiente paso

Ejecutar después de la comparación RQ/979 por etapas ya encolada. El
coordinador conserva las validaciones previas; 1.800 segundos y dos hilos
por tanda, con archivo de todo desenlace en D: y NFS. La cápsula R10 añade
el banco, sus mutantes y snapshots a la cobertura R9.

Antes del primer snapshot se corrigió el inventario esperado: seis fuentes
SV compiladas. El paquete de efectos sigue archivado, pero estos módulos
no lo importan y Vivado lo omite, como acredita la tanda 973. El cambio
del preparador aún no ejecutado conserva su hash anterior y una auditoría
R2; no modifica RTL, ejecutor, revisor ni el orden de la cola.

Un PASS permite preparar la comparación del consumidor tardío, incluidas
sus 32 contribuciones de fallo por lane. Todavía faltan esa selección,
integración completa, síntesis propia, replay y medidas A→B/B→R. El banco
aislado no acredita mejora de timing ni funcionamiento físico.
