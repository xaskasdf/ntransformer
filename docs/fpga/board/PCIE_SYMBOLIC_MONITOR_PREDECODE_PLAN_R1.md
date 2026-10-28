# Monitor PCIe: predecodificación de admisiones R1

## Motivo y alcance

El [ruteo de referencia](PCIE_SYMBOLIC_MONITOR_R1_PHYSICAL_RESULTS.md) mostró
un camino E→D de 15,613 ns: 15,236 ns correspondían a interconexión.
La selección de una fila termina en el enable de sus registros.
Fue un ruteo `Quick`, sin optimización temporal; orienta este experimento,
pero no demuestra que esa sea la limitación de un ruteo optimizado.
Su [replay funcional](PCIE_SYMBOLIC_MONITOR_R1_ROUTED_MAPPED_RESULTS.md)
está aprobado y respaldado.

## Cambio aislado

`ntrpu_pcie_symbolic_effects_predecode_r1.sv` calcula dos vectores de
selección de 256 filas y los registra junto con el registro E existente.
`ntrpu_pcie_symbolic_descriptions_predecode_r1.sv` consume esa selección
en D. No añade un ciclo ni modifica el orden RC→admisión 0→admisión 1.
Las comparaciones desconocidas mantienen la semántica condicional original.

Se conservan 256 tags, dos RQ y cuatro RC por ciclo, cinco clocks C→R,
contexto por valor y cancelación únicamente por reset frío. La referencia
R1 y sus oráculos permanecen intactos. La comparación pública cubre
447 bits con `!==` antes y después del flanco.

## Validación y dependencias

1. **Filtro inicial:** 9.948 filas originales de cinco bancos; 31 sondas
   de quiet y sus restauraciones; 16 mutantes de guards, cinco de selección
   y seis de entrada. Cada mutante debe fallar en el comparador previsto.
2. **Regresión restante:** solo tras revisión y respaldo del filtro,
   once bancos con 173.110 filas. En conjunto: 183.058 filas, incluidos X/Z.
   Los seis bancos drenados deben cancelar cero filas de tráfico original.
3. **Medida del candidato:** síntesis propia, recursos, netlist propia,
   comparación funcional y nueva implementación con revisión de constraints.

Cada tanda usa fuente congelada, dos threads y un máximo de 1.800 segundos;
cualquier resultado terminal se conserva en D: y NFS. El coordinador
`continue_symbolic_monitor_predecode_r1.py` detiene la secuencia ante un
fallo de ejecución, revisión o respaldo.

La variante está en validación. Todavía no hay medidas de mejora temporal,
calificación PCIe nativa ni validación física de la placa.
