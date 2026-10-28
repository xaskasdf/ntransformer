# Comparación de timing por fronteras del monitor

## Motivo

El padre por prefijo 966 tiene peor setup sintetizado B→R de −2,546 ns.
El ruteo Quick anterior 955 estuvo dominado por E→D, con −13,498 ns y
99 % de interconexión en su peor ruta. Son variantes y estados de diseño
distintos; no se compararán como si fueran una mejora del mismo camino.
Hace falta conocer qué limita cada frontera antes del siguiente cambio.

## Tanda encolada

`continue_symbolic_monitor_predecode_stage_timing_r1.py` espera el replay
propio aprobado y respaldado de la bandera. Luego abre copias verificadas
de los DCP sintetizados del padre 966 y del candidato. No ejecuta
optimización, colocación, ruteo ni cambios de restricciones.

Clasifica todos los FDRE en siete grupos disjuntos por jerarquía:
`c_stage`, `e_stage`, `d_stage`, `apply_stage`, `delivery_stage`,
`retirement_i` y registros del top. Exporta inventarios exactos y setup/hold
para los 49 pares dirigidos, incluidos autorretornos. Guarda hasta veinte
peores caminos por pareja y registra explícitamente las parejas sin camino.
Estos grupos incluyen estado local y registros de salida de cada módulo;
no se presentan como particiones funcionales obtenidas solo del nombre.

Además conserva timing global, utilización jerárquica, fanout, reloj,
restricciones y comprobación temporal. Las rutas de E/S permanecen en el
informe global separado. Límite: 900 s y dos hilos; todos los resultados
terminales se respaldan en D: y NFS.

## Revisión

El revisor enlaza ambos DCP con sus síntesis, coteja los registros con los
inventarios archivados, verifica endpoints, sumas de retardos y peor camino
contra las propiedades temporales. Incluye ocho controles de evidencia.
El parser preparatorio pasó un fixture sintético basado en un camino real
archivado y tres controles; eso aún no aporta un informe nuevo de Vivado.

Los resultados son estimaciones de síntesis. Los avisos quedan registrados
para inspección; esta tanda no permite promoción de arquitectura ni
afirma cierre físico. Si hace falta cambiar formato o latencia, se debe
definir una revisión explícita que mantenga un beat RQ2/RC4 por ciclo,
256 tags y el resto del contrato de inventario.
