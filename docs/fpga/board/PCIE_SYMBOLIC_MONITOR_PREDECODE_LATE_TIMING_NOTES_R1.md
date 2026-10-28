# Validación tardía: siguiente hipótesis de timing

## Evidencia disponible

La [síntesis predecode](PCIE_SYMBOLIC_MONITOR_PREDECODE_SYNTHESIS_R1.md)
repite la peor ruta de la referencia: `rc_user[64]` capturado en B hasta
`fault_latched[10]` en R, 31 niveles y 7,566 ns estimados. El reporte contiene
selección de contexto, cálculo de enables y acumulación de fallos. Los nombres
de nets por sí solos no asignan cada LUT a una sentencia RTL.

Fuente inspeccionada: `fpga/rtl/ntrpu_pcie_symbolic_late_validation_r1.sv`,
SHA-256 `2d7dd1619b62bdccbbcf9cd510bb0df169847ec27bee620396e783c6ff6b1894`.
Sus dos recorridos de dieciséis lanes propagan contexto mediante asignaciones
bloqueantes. `faults` parte de cero y sus trece sitios de escritura únicamente
ponen bits a uno; ninguna lectura de `faults` realimenta el contexto.

El [análisis de los INIT mapeados](PCIE_SYMBOLIC_MONITOR_PREDECODE_CRITICAL_LUTS_R1.md)
identificó once LUT consecutivas que acumulan AND sobre datos seleccionados
y nueve que acumulan OR. Verificó conexiones e índices con la biblioteca
instalada. Es evidencia binaria de una ruta estimada; no asigna las LUT
a sentencias RTL ni demuestra mejora física de una reescritura.

## Experimento mínimo propuesto

Separar las contribuciones de fallo por lane y dirección, conservando los
`if` originales, y reunirlas mediante una reducción OR. Mantener primero
intacto el recorrido de contexto. Esto permite medir el efecto de expresar
la reducción por separado antes de intentar un prefijo paralelo de contexto.
Vivado puede ya optimizar parte de esa lógica: no se presupone mejora.

La segunda variante calcula el contexto de cada lane desde los SOP/EOP
anteriores. Requiere demostrar equivalencia para tráfico malformado,
no solo paquetes válidos. La primera variante tiene un
[experimento aislado aprobado](PCIE_SYMBOLIC_LATE_FAULT_LEAVES_R1.md),
con 37.494 muestras y seis mutantes. Su integración aprobó 183.058 filas y
su síntesis quedó revisada: 105.147 LUT y WNS estimado −3,746 ns, sin mejora
del peor setup. Su replay propio aprobó 174.436 filas binarias, seis controles
negativos y 34 corrupciones de evidencia, con respaldo en D: y NFS.
El [análisis actualizado](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_CRITICAL_LUTS_R1.md)
identificó cadenas directas de diez factores AND y diez OR en la nueva ruta.
El [contexto por prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_R1.md)
aprobó 47.734 muestras aisladas y seis mutantes, además de la integración
completa de 183.058 filas nuevas. Su síntesis mide 105.840 LUT y WNS estimado
−2,546 ns: mejora frente al padre, pero sigue incumplida. Su replay propio
aprobó 174.436 filas binarias y quedó respaldado en D: y NFS.
[Medidas del prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_SYNTHESIS_RESULTS_R1.md).

## Casos que deben conservarse

- Los `if` con condición X/Z no toman la rama. No sustituirlos por muxes
  ternarios que propaguen desconocidos de forma diferente.
- SOP ocurre antes del payload del mismo lane y EOP después de incrementar
  el contador. Un EOP también valida longitud cuando el contexto está inactivo.
- `cs` y `rs` tienen once bits y desbordan con ese ancho. `cn` y `cexp`
  sobreviven a EOP; no borrarlos implícitamente.
- Conservar envenenado, enables, payload X/Z, fronteras 1027/1028, resets y
  comparación pública en ambas fases, sin añadir ciclos ni descartar filas.

El siguiente candidato usa una [bandera del último evento](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_R1.md)
para el comparador de contexto identificado formalmente. Cada variante requiere banco directo contra la fuente anterior,
integración completa, síntesis y replay propios; esta nota no concede un PASS.

## Cálculo de límite que se puede separar de la selección

En el candidato por prefijo, el predicado de payload compara
`payload_index < caddress[1:0] + cdelivered`. Tanto la dirección como los
bytes entregados se seleccionan desde el contexto inicial o la misma
consulta de cabecera. La ruta mapeada previa presenta acarreo sobre los bits
seleccionados de `delivered`; eso sugiere evaluar el límite por cada origen
antes de seleccionarlo. La atribución exacta a esta suma sigue sin probarse.

Si las medidas del prefijo mantienen este problema, el siguiente ensayo
mínimo puede precomputar los cinco límites combinacionales —contexto y
cuatro consultas— y seleccionarlos con los mismos `if` existentes. Debe
preservar sus anchos y tratamiento X/Z; no debe introducir un mux ternario
ni mover registros sin una prueba separada. Guardar el resultado en trece
bits sería insuficiente para entradas arbitrarias: `8191 + 3 = 8194`.

La prueba deberá contrastar la expresión original en Vivado con una versión
de ancho explícito, incluir contadores 0/2/3/1027/2047, direcciones bajas
0–3, extremos de `delivered`, desconocidos por bit y selección de cabeceras
superpuestas. También debe verificar cada vector de fallo antes del OR.
Hay un [borrador RTL que registra cinco límites junto a B](PCIE_SYMBOLIC_MONITOR_PREDECODE_REGISTERED_BOUND_DRAFT_R1.md),
derivado de la integración aprobada. No está compilado ni calificado y no
tiene trabajo Vivado encolado. La nueva peor ruta ya cambió; se prioriza
el comparador de contexto identificado en lugar de este traslado de suma.

Una [comprobación binaria independiente del intervalo](PCIE_SYMBOLIC_MONITOR_PREDECODE_PAYLOAD_INTERVAL_MODEL_R1.md)
comparó 67.108.864 combinaciones contra una formulación con máscaras de
primera/última palabra. Pasó todo el dominio y seis controles negativos;
todavía no acredita una reescritura RTL ni semántica X/Z. La factorización
de enables y el traslado de la suma antes de la selección son experimentos
separados, pendientes de selección y validación RTL propia.

El [análisis mapeado del prefijo](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_PREFIX_CRITICAL_LUTS_R1.md)
verificó 22 LUT y dos tramos CARRY8 en la nueva peor ruta, desde `rc_user[79]`.
Su mayor cadena directa de OR es de tres LUT; la de AND, una. La recurrencia
sobre bits seleccionados de `delivered` demostrada para el padre no se
puede atribuir automáticamente a esta ruta. Una prueba posterior atribuyó
el primer cono a `last_eop >= last_sop` antes del lane 13 sobre sus 32 FF
binarios sin restricciones. El [segundo cono](PCIE_SYMBOLIC_MONITOR_PREDECODE_PREFIX_PAYLOAD_CONE_R2.md)
se atribuyó a la comparación del primer byte del lane 13 con el límite
superior de payload. Ambas atribuciones corresponden al padre 966;
la síntesis de la bandera ya permite examinar su nueva ruta por separado.

La [síntesis 970](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_SYNTHESIS_RESULTS_R1.md)
reduce 322 LUT, pero empeora el WNS 0,222 ns respecto a 966. Su ruta desde
`rc_user[64]` tiene 24 LUT y tres CARRY8, incluidos dos arcos S→O; reaparece
una cadena directa de seis factores AND. El prefijo 966 sigue siendo la
referencia comparable. Las atribuciones formales del padre no se trasladan
automáticamente a esos tres conos. Se completa el replay propio y la lectura
por fronteras antes de seleccionar otro cambio RTL; no hay nuevo ruteo.
