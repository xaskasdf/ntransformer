# Conos mapeados de la ruta crítica de la bandera

## Resultado local

Se extrajeron los tres conos que alimentan las salidas CARRY8 de la peor
ruta de la síntesis 970. Cada recorrido termina únicamente en FF reales
de B o constantes; no introduce entradas libres para redes sin resolver.
La netlist y el informe se vinculan al paquete y la revisión de
[síntesis](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_WINNER_SYNTHESIS_RESULTS_R1.md).

| Salida en `delivery_stage` | Nodos / FF de frontera | Función binaria comprobada |
| --- | ---: | --- |
| `fault_latched_reg[10]_i_4077_n_13` | 22 / 12 | Constante cero |
| `fault_latched_reg[10]_i_3629_n_11` | 27 / 12 | Constante cero |
| `retirement_i/late_i/expected_enable2215_in` | 243 / 91 | Límite superior del byte 3 al entrar al lane 6 |

Las primeras dos funciones se evaluaron sobre las **4.096 asignaciones**
de `rc_user[75:64]`, incluidos recuentos malformados. Se detectaron tres
alteraciones seleccionadas por cono. Un control inicial no se propagó al
segundo cono; ese diagnóstico se conserva y no se presenta como detección.

El tercer cono se comparó mediante Z3 con 56 predicados explícitos.
Únicamente se obtuvo UNSAT para:

```text
unsigned15((seen6 - 3) * 4 + 3) < unsigned15(offset6 + delivered6)
```

La reconstrucción usa el contador de once bits, SOP antes del incremento
y EOP después. No necesita guard de payload, tráfico legal ni restricción
de alcanzabilidad de sus 91 FF. Invertir la salida o forzarla a cero o uno
produce contraejemplos. Esta atribución corresponde a 970; el comparador
del padre 966 correspondía a otro lane y byte.

## Consecuencia para el diseño

El STA atraviesa dos salidas que resultan constantes en el dominio binario.
Esto identifica redundancia funcional; **no autoriza nuevas excepciones de
timing** ni sustituir señales por cero en RTL. No se ha demostrado su
comportamiento X/Z, con retardos ni su eliminación por Vivado.

Se preparó un [experimento de distancias constantes](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_R1.md).
El replay propio y la lectura por etapas siguen siendo dependencias antes
de decidir su ejecución. El cierre físico E→D permanece abierto.

## Evidencia

Archivos bajo `build/cloud-setup/20261002-board/`:

- `pcie-symbolic-monitor-predecode-fault-leaves-winner-critical-cones-r1.json`:
  SHA-256 `def2dd1afe5dbfff8ccc071daa3d64dbd565872bc32ba2cc199a20aa00980e5d`.
- `pcie-symbolic-monitor-predecode-winner-constant-cones-r2-proof.json`:
  SHA-256 `aa94d6a05d5a9b52c0e55e1f66c623692e2b7f53c9cc750b80d465da7a18272e`.
- `pcie-symbolic-monitor-predecode-winner-payload-cone-r1-proof.json`:
  SHA-256 `6840ab84d9c1b6fd2e805fb31e6128c28947063a80df93b6ed26380fb19074a4`.

Los resultados fijan hashes del extractor, evaluadores, INIT, conexiones,
bibliotecas instaladas y entorno Z3. Son pruebas combinacionales binarias
de nodos concretos, no equivalencia secuencial del monitor ni timing físico.
