# Siguiente candidato: ascendencia RC por eventos efectivos

Partir del productor R7 y selector R2 **sólo después de aprobar su propia
netlist**. La síntesis R7 revisada mide WNS −8,301 ns, TNS −73934,203 ns,
42.362 LUT y 10.729 FF. Mejora WNS 0,602 ns frente a R6 y 0,505 ns frente
a R5, pero setup/hold siguen fallando. El checkpoint es acíclico.

## Diagnóstico y cambio separado

La peor ruta OOC va de `allocated_tags_reg[243]` a `remaining_bytes_reg[20]`:
12,284 ns/49 niveles. Alcanza `selected_prefix_mask[14]` a 5,259 ns y luego
atraviesa estado previo, dirección, bytes entregados y contador global.
Los nombres orientan el análisis; no prueban por sí solos una causalidad RTL.

El selector R2 todavía explora DWORDs y transporta raíces/máscaras desde
actualizaciones previas. La hipótesis siguiente es derivar esos controles
desde una lista paralela de **EOP efectivos**, sin propagar máscaras entre
escrituras. Conservar inicialmente `rc_ownership`, `rc_prefix_state`, admisión
RQ y aritmética de bytes; medir un solo cambio de diseño a la vez.

El premodelo `tools/check_ntrpu_pcie_rc_event_ancestry_model.py` ya coincide
en **155.168 entradas binarias/840 bits de salida** con el modelo secuencial.
Registro: `build/cloud-setup/20261002-board/pcie-inventory-rc-event-ancestry-model.json`.
Este filtro sólo valida el algoritmo binario de eventos; todavía no existe
el candidato RTL ni acredita cuatro estados, síntesis o netlist.

Para cada EOP efectivo, determinar el último SOP anterior o coincidente,
o el contexto retenido cuando no exista SOP. Sólo el primer EOP de ese
contexto conserva propiedad; los siguientes encuentran `owned=0`. Un SOP
nuevo sustituye el contexto aun cuando el framing sea erróneo.

Para cada encabezado, seleccionar la última escritura anterior del mismo
tag. Investigar si su máscara se obtiene mediante OR de los encabezados con
escritura efectiva anterior y su raíz retenida mediante una escritura válida
del contexto inicial. Demostrar esta equivalencia: no asumir que cualquier
SOP anterior contribuye ni omitir descartes por terminal, propiedad o error.
La escritura final por tag conserva el último EOP efectivo y el slot físico
con la prioridad original en punteros duplicados.

## Cuatro estados y criterios

- Conservar `if (owned)`: X/Z no admite una escritura. En `if (terminal)
  ... else if (normal)`, terminal X/Z entra en la alternativa; no convertir
  esta regla en `!terminal` sin probar su semántica. Normal X/Z no escribe.
- Mantener comparación procedural de tags, termómetros malformados,
  punteros desconocidos, SOP/EOP simultáneos, discontinue, contexto retenido,
  ausencia de SOP y varios EOP posteriores. No aprender identidad desde RC.
- Conservar las 89.632 entradas del selector y el suplemento ampliado de
  155.168, con sus salidas completas. Reejecutar 1.153.296 comparaciones,
  incluidas 994.288 X/Z; ampliar casos si la nueva descomposición lo requiere.
- Conservar el banco principal 22.339/68 y siete mutantes, RQ 615.536/5.712
  X/Z más 8.850/240, RC 45.056/6.624 X/Z más 544/60, y propiedad integrada
  46.356/3.872. Comparar tablas y todos los estados posteriores a fallos.
- Exigir síntesis VU5P/4 ns/RuntimeOptimized, DCP auditado y replay de netlist
  propia frente a 22.330/67 y los tres suplementos binarios completos.
  Comparar área, WNS/TNS, endpoints y ruta con R7, R6, R5 y original.

Si el control queda reducido pero domina la aritmética, estudiar después
prefijos constantes calculados en paralelo y selección tardía, conservando
la recurrencia exacta incluso en cero, desbordamientos y X/Z.

Para ahorrar tiempo de VM, evaluar el [inicio XSIM sin captura automática
de ondas](XSIM_OWNERSHIP_RUNTIME_NOTE.md) conservando todos los casos y logs.
Su ahorro requiere medición. No modificar tandas ya congeladas.

Siguen abiertos conexión nativa, BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, inferencia real y diseño físico/ECAD. Mantener íntegra
la [cola y sus dependencias](VIVADO_QUEUE.md).
