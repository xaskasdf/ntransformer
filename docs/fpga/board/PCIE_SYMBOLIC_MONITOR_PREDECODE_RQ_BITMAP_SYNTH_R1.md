# Síntesis propia del candidato bitmap RQ

## Condiciones de entrada

La ejecución depende del PASS de las 42.910 filas del C real, sus cuatro
fuentes defectuosas y la regresión completa de **183.058 filas** del
monitor. Cada resultado debe estar revisado y respaldado en D: y NFS.
El coordinador conserva esos trabajos y se detiene ante un diagnóstico.

Se congela un snapshot nuevo. Los hashes de la integración aprobada,
las 26 unidades de simulación y las 14 fuentes sintetizadas se enlazan
al archivo de configuración. Las referencias originales R3 permanecen
en la simulación; síntesis usa únicamente el C bitmap RQ, el selector RC
de prioridad y el monitor nuevo.

## Medición comparable

El filtro previo repite 529 filas de los bancos `monitor` y `range_return`,
con 31 sondas de quietud. Después se sintetiza el monitor en
`xcvu5p-flva2104-2-i`, con reloj de 4 ns, retardos de E/S de 0 ns,
`RuntimeOptimized` y jerarquía `rebuilt`. El límite es 1.800 segundos y
dos hilos. No se coloca ni rutea en esta tanda.

Se conservan DCP, EDIF, netlist funcional, XDC, inventario de primitivas
y reportes de utilización, timing, cobertura de restricciones, DRC,
metodología y excepciones. Una auditoría de conectividad debe descartar
ciclos combinacionales. La revisión inspecciona las advertencias reales
y rechaza cualquier clase no contemplada.

## Comparación y siguiente paso

La referencia es la **síntesis 979**, ya revisada y con replay propio
aprobado: 91.660 LUT, 27.532 FF, 568 CARRY8 y WNS estimado −3,704 ns.
La comparación verifica iguales versiones, builds y restricciones antes
de calcular diferencias. No se selecciona otra referencia por presentar
un resultado más favorable.

La decisión depende de las mediciones obtenidas, incluida la posible
penalización de área o fanout por decodificar cuatro tags. El resultado
de síntesis no sustituye el replay del netlist propio ni el cierre físico.
No se atribuye todavía ganancia de frecuencia, inferencia o capacidad
de placa a este candidato.
