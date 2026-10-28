# Banco de metadatos PCIe: implementación y mediciones

## Alcance

`fpga/rtl/ntrpu_pcie_tag_metadata.sv` implementa un banco de **256 entradas
de 30 bits**, cuatro consultas registradas y seis actualizaciones ordenadas
por ciclo. No introduce backpressure ni reduce los puertos cuando coinciden
los tags. Es una pieza de la arquitectura propuesta; todavía no sustituye
la consulta activa de R7 ni implementa el monitor segmentado completo.

| Bits | Campo |
| --- | --- |
| 12:0 | Bytes restantes |
| 24:13 | Dirección siguiente |
| 27:25 | Clase de tráfico |
| 29:28 | Atributos |

Una escritura parcial cambia los primeros 25 bits; una completa también
cambia clase y atributos. Para cada campo gana la última escritura válida
al mismo tag. Cada consulta puede observar la tabla anterior y un prefijo
de cero a seis escrituras del ciclo. El prefijo 7 produce error y resultado
inválido. Las consultas inactivas devuelven cero. El reset frío sincrónico
limpia tabla y resultados y domina todas las escrituras.

La interfaz exige controles y tags binarios ya clasificados. No contiene
bitmap de asignación, generación de tag, contexto del payload antiguo,
certificado de vacío ni ACK. Las escrituras llegan **ya calculadas**: su
producción a tiempo dentro del monitor es una dependencia todavía abierta.

## Regresión RTL

La tanda `20261007T171521Z` terminó en **34,745 s** y pasó la revisión de
**28.912 ciclos**, en nueve grupos. Incluye todas las direcciones y prefijos,
las 4.096 combinaciones de máscaras de validez/escritura completa, 4.096
ciclos consecutivos de contención al mismo tag y accesos distintos con los
mismos bits bajos de banco. La referencia usa snapshots de una tabla de
software después de cada escritura ordenada.

Se observaron 13.551 ciclos con cuatro consultas, 8.559 con seis escrituras,
4.163 con las seis al mismo tag, 4.094 consultas inválidas y 19 resets.
El banco de prueba verifica que los resultados no cambian antes del flanco.
La revisión rechazó ocho corrupciones de evidencia.

## Síntesis del banco aislado

Vivado 2026.1, `xcvu5p-flva2104-2-i`, período 4 ns y retardos I/O de 0 ns,
aplicados antes de síntesis. La tanda `20261007T171832Z` tomó **367,594 s**.

| Medida OOC, sin colocar ni rutear | Resultado |
| --- | ---: |
| LUT de CLB | 33.763 |
| Registros | 7.808 |
| Latches / BRAM / URAM / DSP / CARRY8 | 0 |
| WNS / TNS de setup | +1,924 / 0 ns |
| WHS / THS de hold | −0,042 / −88,802 ns |
| Endpoints que fallan setup / hold | 0 / 7.808 |
| Peor camino de setup | 10 niveles / 2,058 ns |

El EDIF final es acíclico: 51.190 primitivas y 51.521 nets. Se rechazaron
cuatro corrupciones en su auditoría. Los doce controles `check_timing`
dan cero y no hay excepciones válidas ni ignoradas. El DRC OOC no reporta
checks, pero avisa que la cobertura de conectividad es incompleta.

**Hold sigue fallando.** `HD.CLK_SRC` no está definido y metodología reporta
417 avisos por retardos I/O mínimos y máximos iguales. Estas restricciones
caracterizan el bloque; no representan el presupuesto temporal del top.
La cifra positiva de setup no aprueba la placa ni el monitor R7. Tampoco
debe sumarse o restarse el área de este bloque a R7 como si fueran diseños
equivalentes.

## Reproducción del netlist

La tanda `20261007T173158Z` terminó en **294,511 s**. Compiló el netlist
funcional de esa síntesis y el mismo testbench, sin el DUT RTL. Sus 28.912
filas coinciden con la referencia de software y con todas las filas RTL.
Se conservaron los ocho controles de corrupción, 200 ns de arranque GSR
y muestreo 1 ns después del flanco. No se aplicó SDF.

El aviso de redefinición de `glbl` se revisó comparando los cuerpos del
módulo incluido en el netlist y del compilado por Vivado: son idénticos
tras quitar comentarios y espacios. No se ignoraron definiciones distintas.
Esta reproducción funcional no resuelve el fallo de hold.

## Comparación con eventos de R7

La tanda `20261007T172938Z` terminó en **69,446 s**. El banco recibe una
copia de los eventos calculados por R7: cuatro RC ordenados por lane y dos
RQ posteriores. RC modifica solo dirección/bytes; RQ puede cambiar también
clase y atributos. El orden por lane conserva el tratamiento de EOP
inválidos de la referencia.

Se compararon las **256 palabras en cada uno de 22.339 ciclos**:
5.718.784 comparaciones de palabras y 89.356 posiciones de consulta.
Pasaron los 68 escenarios, incluidos los estados posteriores a fallo.
Todos los estados públicos emitidos por R7 permanecen idénticos a la tanda
R15 revisada. Dos mutaciones ejecutadas se rechazaron: omitir una escritura
RQ falla en la fila 4; invertir el orden falla en la consulta de la fila
211. La revisión rechazó otras cinco corrupciones de evidencia.

El primer intento `20261007T172533Z` falló antes de compilar por el manejo
Tcl de un valor que empezaba por `-testplusarg`. Se conserva como diagnóstico.
La revisión Tcl R1 usa `set_property -dict`; no cambió RTL ni testbench.
Quedan registrados tres avisos del testbench por declarar `index` después
de sus tareas y tres avisos del entorno de compilación XSIM. Las secuencias
comparadas y las filas de rechazo confirman el contador usado.

## Dependencia pendiente: producir las actualizaciones

El shadow conserva R7 como productor y consumidor de sus metadatos activos.
Por tanto, **no prueba** que consultas y transiciones puedan encadenarse a
un beat por ciclo con la nueva latencia. Una actualización que dependa de
una lectura aún no disponible no se arregla añadiendo una FIFO.

El [experimento de propuestas RC](PCIE_RC_PROPOSAL.md) ya separa el cálculo
aritmético del descriptor: pasó 69.876 ciclos RTL/netlist y coincide con
los eventos sanos de los 68 escenarios R7. Conserva tres diferencias en
tráfico inválido. Falta integrar el parser y la comparación contra metadatos
independientes, manteniendo inválido todo certificado si hay transiciones
especulativas, obligaciones sin retirar o fallos latentes. La identidad
esperada del requester seguirá teniendo origen independiente.

Los criterios de latencia, forwarding, reutilización del tag y revocación
de vacío están en el [contrato de etapas](PCIE_INVENTORY_PIPELINE_CONTRACT.md).
Fuente/host, siete contratos, reset nativo, memoria exacta/512 GiB,
inferencia y entrega física siguen en [VIVADO_QUEUE.md](VIVADO_QUEUE.md).

## Evidencia y respaldo

Los archivos siguientes están verificados en D:/NFS. Las revisiones,
configuraciones y fuentes se conservan en el checkpoint de la cola.

| Tanda | SHA-256 del archivo |
| --- | --- |
| RTL `171521Z` | `ea23788de9a885442f67eb76c6a428646b05cacf22a8b19639fd217052ce86b5` |
| Síntesis `171832Z` | `8df801b7e73400472af0a98336e16e9bbc62fa3bc9861c59b75690490bf7abfe` |
| Diagnóstico Tcl `172533Z` | `0aedba4e7af94cfac64771597d2ec5a39329c59ceb8e9ab364a085f7b5b516cb` |
| Shadow y mutaciones `172938Z` | `6ef8bb009b610788d013b925e0566d401bbe2f29aa6b8c3ced420c400c9dda94` |
| Netlist `173158Z` | `5abd774605acab2409ef4729691002201e41312c5065293db76d3994c9f62125` |
