# Conteo temprano de admisiones RQ — banco focalizado R1

## Motivación medida

El [reporte por etapas RQ](PCIE_SYMBOLIC_MONITOR_PREDECODE_RQ_BITMAP_STAGE_RESULTS_R1.md)
sitúa C→C en −2,758 ns: `allocated_tags[51]` llega a
`read_range_high[0]` a través de ownership, admisión RQ y escrituras de
metadatos. El camino tiene 28 niveles y dos CARRY8. Esta atribución procede
del DCP 984; no predice el resultado de un cambio nuevo.

El candidato conserva ese C y cambia una sola asignación: cuenta los
cuatro bits de admisión ya calculados para formar `allocate_count`, en
lugar de reconstruirlo desde las dos escrituras de metadatos. Conserva
la resta de completions, rango firmado, prioridad RC→RQ y latencia.

## Prueba preparada

Se comparan **51.110 filas nuevas** contra C RQ: registro de 3.574 bits,
pending, bitmap siguiente de 256 bits y un comparador adicional de
13 bits para admisiones/contador siguiente, antes y después del flanco.
El banco conserva 8.850 filas RQ revisadas y añade campos kind y ready/valid
completos en cuatro estados. No escribe ni fuerza estado interno.

Cuatro fuentes defectuosas deben fallar en testigos fijados previamente:
omitir lane 3, colapsar dos admisiones, contar solicitudes bloqueadas y
sumar completions. El revisor exige cobertura exacta y rechaza 21
alteraciones de evidencia.

Un modelo local comprobó 4.296 selecciones sobre 4.096 combinaciones
de SOP/punteros, condicionado a que las admisiones pertenezcan a las
posiciones SOP conocidas. Conserva la prioridad del segundo slot al
duplicar puntero. Ese modelo no sustituye la simulación del C real.

## Cola y límites

Espera el cierre revisado y respaldado del replay B-bound/winner.
Usa seis fuentes, dos hilos y hasta 1.800 s. R15 preserva los snapshots
pendientes y cada resultado terminal se guarda en D: y NFS.
Un PASS focalizado habilita integrar y probar el monitor completo;
síntesis, netlist propia y timing físico siguen pendientes.
