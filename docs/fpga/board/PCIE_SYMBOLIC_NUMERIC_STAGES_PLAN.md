# C/E/D: aplicación numérica y retiro pendientes

Diseño propuesto el 2026-10-09; **todavía no implementado ni calificado**.
Concreta S2/S3 del [contrato inicial](PCIE_SYMBOLIC_PIPELINE_R1_CONTRACT.md)
después de la partición C/E/D. Su integración queda detrás del replay propio
de la revisión de frontend seleccionada; R12 sigue siendo la referencia del
monitor completo.

## Fronteras candidatas

| Flanco del origen | Etapa | Responsabilidad |
| --- | --- | --- |
| N | C | Control, propiedad y operaciones crudas |
| N+1 | E | Composición de comandos por valor |
| N+2 | D | Tabla, consultas y contexto descriptivo retenido |
| N+3 | A | Evaluar nueve descripciones en copias numéricas |
| N+4 | B | Calcular delivered y contribuciones EOP |
| N+5 | R | Validar, reducir deltas y publicar estado/fallos |

Esta partición añade dos fronteras numéricas explícitas para separar la
evaluación de efectos de la aritmética de entrega. Su temporización debe
medirse. Conserva un origen por ciclo, RQ2/RC4, 256 tags, 257 lecturas y
1.052.672 bytes; no introduce espera, pérdida de eventos ni cola finita
que pueda desbordarse bajo tráfico sostenido.

## Datos y semántica de A/B

A recibe el `s1_record_t` completo y los 391 bits de control correspondientes
a D. En la primera implementación conservará ambos por valor, junto con
nueve resultados `{remaining[12:0], address[11:0]}`: cuatro consultas,
cuatro EOP y un contexto previo. Son 225 bits numéricos adicionales;
no se eliminan campos de diagnóstico antes de probar esta frontera.

La evaluación se basa en `apply_effect` del paquete ya revisado. La semilla
de la descripción guarda dirección y remaining en orden distinto al
resultado numérico: debe conservarse la transformación exacta de
`symbolic_value` del modelo. TC y atributos viajan separados.

B usa esas copias, las palabras y `delivery_enabled` capturados para obtener
delivered de 13 bits. Debe conservar la expresión signed/unsigned original,
el truncamiento y el `if` procedural: un enable desconocido no se convierte
en una entrega. Un EOP terminal contribuye original; uno no terminal,
delivered. Conservar el operador condicional de selección y su semántica
de cuatro estados. Las consultas/EOP no válidas tienen salida canónica cero;
el contexto previo se evalúa incluso cuando ya no posee un tag.

A/B nunca consultan el bitmap actual, recargan una fila por tag ni escriben
de vuelta la tabla D. Una petición nueva con el mismo tag no cambia el
payload anterior. No se usa la secuencia como dirección o autorización.

## Retiro, fallos y cancelación

R mantiene su contador de bytes y procesa deltas en orden con guardas de
rango y anchuras explícitas. La reducción de cuatro contribuciones EOP
debe cotejarse con las dieciséis lanes de R7, incluidos alias/punteros
malformados. El estado público, los fallos tardíos y el fallo sticky de
rango deben corresponder al mismo origen, sin desplazarlo otro ciclo.

Reset frío borra válidos, copias y estado de retiro en el mismo flanco.
Reset de usuario y pérdida de link no cancelan obligaciones. La actividad
actual revoca quiet inmediatamente; los efectos capturados mantienen
pending hasta retiro. A/B por sí solas no certifican vacío.

## Evidencia requerida

1. Probar A y B contra observaciones numéricas de R7/probe y el modelo,
   usando todos los bancos crudos, con vaciado y dirigidos de C/E/D. La
   referencia no recibe estados ni resultados del candidato.
2. Probar cada etapa ocupada al reset, wrap, payload retenido con tag
   reutilizado, contadores fuera de rango, X/Z y filas posteriores al fallo.
   Contabilizar cada origen retirado o cancelado por reset frío.
3. Rechazar mutaciones de orden de campos, recarga por tag, omisión/doble
   consumo, pérdida de enable capturado, alineación de control/fallos y
   quiet prematuro; comprobar salida numérica completa y deltas por lane.
4. Medir A/B/R y el monitor integrado con el criterio de 4 ns, I/O cero y
   sin excepciones; auditar DCP y ejecutar su propia netlist. Pasar pruebas
   aisladas no cierra inventario nativo, CDC/reset, DDR, pesos o placa.
