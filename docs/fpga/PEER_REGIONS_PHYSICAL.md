# Implementación física del mapa de regiones

## Alcance y procedencia

Tanda aislada del 8 de octubre de 2026. Implementa `ntrpu_peer_regions` con
dos regiones autorizadas y dos controles de dominio, usando exactamente el
RTL que pasó la prueba HBM nativa de [PEER_REGIONS.md](PEER_REGIONS.md).
El árbol privado conserva todos los hashes del donante nativo; los scripts
nuevos se añaden sin sustituir las fuentes de referencia.

- Donante: `peer-regions-20261008T130010261412Z-traffic1`.
- Resultado nativo SHA-256: `eac5bd19b66464b99626c568a9bb85c7f3c89a799514853a9915b3abd654e3f1`.
- Evidencia local: `D:/AI/artifacts/ntrpu-peer-regions-physical/20261008T135935229195Z`.
- Dispositivo: `xcvu47p-fsvh2892-2-e`, Vivado 2026.1.
- Clocks: transporte 5,000 ns y memoria 2,222 ns; incertidumbre de setup
  0,100 ns y hold 0,050 ns. Se conservan las restricciones CDC anteriores.

Es un bloque fuera de contexto (OOC). El controlador HBM nativo, PCIe, los
retardos de interfaces externas y la placa completa quedan fuera de esta
implementación. Una prueba funcional nativa anterior no demuestra timing
físico de la integración completa.

Los clocks de entrada OOC no tienen `HD.CLK_SRC`; Vivado advierte que falta
calificar su origen físico. Hay 6.527 entradas y 5.686 salidas sin retardos
externos, además de dos entradas con restricciones parciales. Esas interfaces
deben cerrarse al integrar el bloque; la auditoría actual exige cobertura
completa de los endpoints internos.

## Coste de síntesis

| Recurso | Referencia anterior | Mapa de dos regiones | Diferencia |
|---|---:|---:|---:|
| LUT | 22.185 | 25.153 | +2.968 (+13,38 %) |
| Registros | 23.764 | 24.358 | +594 (+2,50 %) |
| LUT como memoria | 3.036 | 3.036 | 0 |
| CARRY8 | 1.405 | 1.741 | +336 |
| BRAM | 0 | 0 | 0 |

El bloque sintetizado ocupa el 1,93 % de las LUT y el 0,93 % de los registros
del dispositivo según Vivado. Estas cifras no representan el motor completo
de inferencia ni justifican aumentar el número de buffers sin medir el
tráfico. La utilización total tampoco descarta congestión local.

La síntesis tardó 372,684 s. Se comprobaron ausencia de latches, cajas negras,
bucles y endpoints internos sin restricciones, junto con los sincronizadores
de arranque y las configuraciones de IP. El auditor rechaza nueve alteraciones
de cobertura, CDC y clocks. El checkpoint sintetizado tiene SHA-256
`0da630f555b58530b0488ac78bd0c4594d72eaad614c46faebd44fc50faec6cc`.

## Estado de implementación

**Ruteo completo, timing interno aún abierto.** El primer intento alcanzó el límite
de 1.200 s durante la segunda iteración global del ruteador. Se archivó como
fallido, sin aprobación de timing. Su última solución intermedia sin
solapamientos tenía WNS −0,783 ns; no es un informe final de ruteo.

Se verificó el checkpoint de colocación y la continuación terminó en
1.213,393 s, dentro de su límite independiente de 1.800 s. Conservó el RTL,
los clocks y los criterios de aceptación. Los hashes enlazan síntesis,
intento interrumpido, colocación y continuación.

| Métrica final | Resultado |
|---|---:|
| LUT / registros | 24.771 / 24.344 |
| Setup transporte | +0,035 ns |
| Setup memoria | **−0,019 ns** |
| Endpoints que fallan setup / TNS | 57 / −0,622 ns |
| Hold transporte / memoria | +0,018 / +0,007 ns |
| Margen mínimo de bus skew | +1,661 ns |
| Nets completamente ruteadas | 38.636 |
| Errores de ruteo | 0 |

El checkpoint seleccionado es
`731d3a2081b3a134bd3b44d2117bb0bf7810fdcf2a0bd9fedf4097178fee2fa2`.
Su estado es `REGIONS_ROUTED_INTERNAL_TIMING_OPEN`, no una aprobación física.
CDC conserva únicamente las clases revisadas CDC-3/CDC-9; todos los endpoints
internos tienen restricciones y READY nativo sigue partiendo de registros.

El peor camino de memoria va desde `validate_ptr_reg[0]` hasta el CE de
`validated_cut/slot0_reg[185]`, dentro del tracker del lector del canal 0.
Tiene siete niveles, 1,963 ns de datos (0,728 ns de lógica y 1,235 ns de rutas).
La última señal de habilitación llega a 259 registros. El mapa de regiones
en transporte no es el camino que incumple la frecuencia en esta corrida.

Una pasada posterior `phys_opt_design -directive Explore` mantuvo WNS, TNS y
los 57 endpoints sin mejorar. Su exportación falló porque el script había
copiado `routed.v` al directorio de salida antes de generarlo. La tanda se
archivó como fallida y no se seleccionó. El script corregido copia sólo
metadatos de entrada; esa corrección aún no se ejecutó en Vivado. Se evitó
repetir la misma optimización sin una mejora observada. `REFINEMENT_NOTE.json`
registra ambos hashes del script y el diagnóstico.

## Advertencias del checkpoint seleccionado

La auditoría individual terminó en 201,431 s y clasificó las siete advertencias
sin aplicar waivers. Las seis PDCN-1569 corresponden a entradas de LUT que no
afectan a su función, comprobado mediante `INIT`. RTSTAT-10 enumera 3.405 nets:
3.303 puertos OOC sin carga interna y 102 nets sin cargas lógicas hoja.
No quedaron objetos sin resolver. El estado es `REGIONS_DRC_OBJECTS_REVIEWED`;
las advertencias siguen presentes y esta revisión no elimina el déficit de timing.

## Banco funcional por puertos públicos

El nuevo banco pasó primero contra RTL en 69,048 s. Reutiliza los seis escenarios
de propiedad compartida y añade lecturas mapeadas a PC0/PC16, cambio de generación,
modificación del mapa mientras hay reservas y rechazo fuera de ambos rangos.
El patrón de datos incorpora los bits altos de la dirección.

Se observaron 37 solicitudes, 31 admisiones, seis rechazos y 27 resultados
correctos con 46.848 bytes. Cuatro resultados fallidos retuvieron sus reservas.
Los 118 comandos AXI se distribuyeron en 69 hacia PC0 y 49 hacia PC16.
Doce controles negativos detectaron eventos faltantes, payload alterado y
truncamiento del bit del stack.

Este banco usa un respondedor AXI conductual en orden. Su aprobación no es una
simulación del netlist ni una nueva prueba del controlador HBM nativo. La
reproducción sobre netlist exige antes una revisión física aprobada; inspecciona
sólo puertos públicos, sin `force`, sondas internas ni modificación del netlist.
El banco para netlist está preparado, pero no se ejecutó al seguir abierto el
timing. No se hereda la aprobación del netlist de la variante anterior.

## Reproducir y revisar

Preparar desde la carpeta de una tanda nativa revisada, siempre en un árbol privado:

```text
python -B tools/prepare_ntrpu_peer_regions_physical.py <tanda-nativa> <repo-privado>
python3 -B <repo-privado>/tools/run_ntrpu_peer_regions_physical.py <salida> --stage synth --prior <salida-nativa> --review <revision-nativa> --cpus 8 --timeout 1200
python3 -B <repo-privado>/tools/check_ntrpu_peer_regions_physical.py <salida>
```

La etapa `route` necesita el resultado y la revisión de síntesis. Ejecutar
los runners en Linux con `settings64.sh` cargado. La continuación requiere una
procedencia explícita de la colocación; no convertir un timeout en aprobación.
El wrapper y los informes generados forman parte de `FILES.json`.

Cada tanda se exporta localmente, se verifica por SHA-256 y se almacena en
`D:/AI/artifacts/ntrpu-content-store`. La restauración en una carpeta nueva
debe reproducir la revisión del auditor. No se conservan ZIP de transferencia.
Esta entrega conserva seis tandas, incluidas las dos fallidas, y su índice es
`peer-regions-physical-20261008T135935229195Z-handoff`. El archivo `OUTCOME.json`
en la carpeta de evidencia enlaza las revisiones, hashes y respaldo de cada una.

## Criterio para avanzar

Exigir setup, hold y bus skew no negativos, cobertura interna completa y
revisión de las advertencias del checkpoint. El siguiente cambio debe estudiar
el recorrido desde la lectura de longitud del tracker hasta la habilitación
del slot: precalcular o registrar esa decisión y preservar reservas, orden,
errores y estabilidad bajo backpressure. Toda modificación RTL exige repetir
las pruebas afectadas antes de otra implementación física.

Tras aprobar timing, reproducir el
banco público en el netlist y conectar carga de pesos/publicación al mismo
mapa. El rendimiento sostenido, las interfaces PCIe/BAR y la calificación
eléctrica, térmica y de PCB siguen siendo hitos posteriores.
