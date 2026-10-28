# Experimento de contexto por prefijo de lanes

## Propósito y estado

La validación tardía propaga contexto por dieciséis lanes en un solo ciclo.
La ruta estimada anterior contiene once LUT AND consecutivas, además de la
acumulación de fallos. Esta variante reconstruye el contexto al entrar en
cada lane y conserva su cuerpo original SOP → payload → EOP.
El ensayo aislado aprobó **47.734 muestras** y seis mutantes reales en la
tanda 964. La integración completa está en curso; todavía no hay medidas
de síntesis ni mejora de timing demostrada.
[Resultado aislado y evidencia](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_RESULTS_R1.md).

## Regla de entrada

Para el lane `i`, `s` es el último SOP alineado estrictamente anterior y `e`
el último EOP anterior. Un índice ausente vale −1:

| Condición | Activo | Contador de entrada |
| --- | --- | --- |
| `s > e` | 1 | `i − s` |
| `e ≥ 0` y `e ≥ s` | 0 | 0 |
| Ningún evento anterior | valor inicial | inicial + `i` solo si activo es 1 |

EOP gana cuando coincide con un SOP anterior porque ocurre después del
payload. Los campos de cabecera sobreviven a EOP; la propiedad RC se borra.
Se conserva el módulo de decodificación de eventos. El contador mantiene
once bits y no se suma cero en el lane inicial, para preservar su valor X/Z.
La aritmética completa de cuatro estados requiere comprobación en RTL.

## Evidencia del modelo

`prove_pcie_symbolic_late_fault_leaves_prefix_model_r1.py` comparó el recorrido
secuencial y la fórmula en **4.194.304 combinaciones**, equivalentes a
**71.303.168 fronteras de lane**. Incluye todos los mapas de cuatro SOP
alineados y dieciséis EOP, con activo inicial 0/1/X/Z, y 34.816 comprobaciones
del contador binario. Nueve alteraciones deliberadas fueron rechazadas.
Duración local: 69,818 s. Prueba SHA-256
`e45d521afc321d4c3e63c984034a0387a983103d59744a3ad80782553847a7b6`.
Esto valida el modelo abstracto, no el RTL, payload ni síntesis.

## Candidato y validación preparada

`ntrpu_pcie_symbolic_late_fault_leaves_prefix_r1.sv` conserva los trece sitios
originales que activan fallos y el árbol OR. Las sustituciones reconstruyen
exactamente el padre al deshacerse. No añade ciclos ni estado con reloj.

El banco preparado compara **47.734 muestras** con la referencia original
y el padre por hojas ya aprobado. Añade 10.240 casos SOP/EOP, contadores y
activo 0/1/X/Z a las 37.494 muestras previas. Compara además 32 vectores de
fallo por lane, antes del OR, para detectar errores ocultos por otros fallos.
Se prepararon seis mutantes de contexto y un Tcl de siete proyectos.

El lanzador, congelador y revisor están preparados. El revisor exige la
primera discrepancia prevista para cada mutante, comprueba fuentes y
archivos realmente compilados y rechaza quince corrupciones de evidencia.
Su comprobación local con logs sintéticos valida el lector, no el RTL.

`continue_symbolic_late_fault_leaves_prefix_r1.py` espera el replay aprobado
de la integración por hojas antes de congelar y respaldar este experimento.
Reserva 900 s y dos hilos para siete proyectos XSIM. Todo resultado terminal,
incluidos fallos y timeout, se archiva; una discrepancia detiene la promoción.
La cápsula de trabajo R6 incorpora la nueva copia de fuentes antes del lanzamiento.

La [integración completa](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_INTEGRATION_R1.md)
también quedó preparada, condicionada al PASS aislado y al turno libre.
Repite las 183.058 filas; no reutiliza resultados de otras variantes.
También quedaron encolados la [síntesis propia](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_SYNTH_R1.md)
y el [replay de su netlist](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_MAPPED_R1.md),
condicionados a las revisiones anteriores. La integración de 183.058 filas
comenzó en la tanda `20261011T130303Z`; quedan su revisión y las medidas
de esta variante. No se ha programado su ruteo.
No se selecciona esta arquitectura por el modelo abstracto ni se modifica
la cadena de pruebas ya activa.
