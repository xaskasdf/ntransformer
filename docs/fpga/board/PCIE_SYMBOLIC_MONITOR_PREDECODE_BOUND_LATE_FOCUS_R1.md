# Límite de payload: banco del consumidor tardío

## Alcance

Después del [productor B registrado](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_FOCUS_R1.md),
probar el consumidor del límite `address[1:0] + delivered`. Se reutiliza
el borrador de consumidor por prefijo, con entradas B públicas y límites
consistentes calculados combinacionalmente. No se conecta aún el registro B.

Comparar el vector de fallos con el validador original sin modificar y
las 32 contribuciones RC/RQ por lane con la referencia por bandera de
último evento. Comprobarlas antes del OR impide que otra lane oculte un
error. Ambas referencias conservan los hashes calificados en la tanda 973.

## Cobertura prevista

El banco deriva exactamente el de 973 y ejecutará **309.884 muestras
nuevas**, sin reutilizar resultados:

- Las 47.734 muestras anteriores: 10.240 fronteras SOP/EOP, 17.352
  walking X/Z, 20.000 registros aleatorios y casos dirigidos.
- 262.144 combinaciones de los 8.192 valores de `delivered`, cuatro
  alineaciones y ocho valores de `seen`, incluidos desbordamientos.
- Seis casos dirigidos adicionales para contexto, extremo exclusivo,
  límite 8.194, selección de query, extremo desconocido y limpieza.

La matriz adicional genera máscaras correctas mediante aritmética entera
y compara el fallo esperado del contador. Seis fuentes defectuosas prueban
omisión del contexto, extremo inclusivo, truncamiento a trece bits,
rotación de query, X tratado como máximo y pérdida del límite inferior.
Sus primeras divergencias deben ser exactas; exigir además 21 controles
de corrupción de evidencia.

## Ejecución y decisión

La cola espera el PASS y respaldo del productor B, conservando las tandas
RQ y E previas. Usar 1.800 segundos y dos hilos. Verificar las ocho fuentes
SV realmente compiladas por caso, fuentes, revisores y archivos de cada
desenlace en D: y NFS. El paquete de efectos no utilizado sigue archivado.
La cápsula R11 añade los snapshots del consumidor a la cobertura anterior.

El resultado permitirá integrar productor y consumidor, preservando
latencia, prioridad, capacidad, quietud y cancelación del monitor completo.
Todavía se exigirán regresión completa, síntesis comparable, replay propio
y medidas A→B/B→R. Este banco no acredita timing ni placa física.
