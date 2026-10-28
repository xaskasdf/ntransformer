# Validación tardía: experimento de fallos por lane R1

## Estado

La tanda aislada `20261011T104053Z` terminó en 150,527 s y aprobó la revisión
independiente R4. Están respaldados en D: y NFS tanto la simulación como la
revisión. **Sin síntesis propia ni aprobación de timing todavía**. No
sustituye el monitor validado; habilita una prueba de integración completa.

La [ruta estimada](PCIE_SYMBOLIC_MONITOR_PREDECODE_CRITICAL_LUTS_R1.md) incluye
nueve LUT consecutivas que acumulan OR. Se medirá si expresar los fallos
por lane permite mejorar esa estructura; Vivado podría generar el mismo
circuito, y la cadena aritmética anterior seguiría presente.

## Cambio de RTL

`fpga/rtl/ntrpu_pcie_symbolic_late_fault_leaves_r1.sv` deriva del bloque
`ntrpu_pcie_symbolic_late_validation_r1.sv`. Sus diez escrituras RC y tres
RQ se distribuyen entre 32 vectores de fallos de 16 bits, inicializados a
cero en cada evaluación. Un árbol OR de cinco niveles **en RTL** reúne
las contribuciones. El mapeo físico y su profundidad siguen sin medir.

El generador reconstruyó exactamente la fuente anterior al revertir estas
transformaciones. Conserva predicados, anchos, orden de contexto, SOP antes
del payload y EOP después del incremento, desbordamientos y comportamiento
procedural X/Z. No añade estado, reloj, ciclos ni cambios de capacidad.
La reconstrucción textual verifica el alcance de la edición; no reemplaza
la prueba funcional.

## Banco directo ejecutado

El banco instancia ambos RTL sobre el mismo registro B de 2.892 bits y
compara los 16 bits de fallo después de estabilizar la lógica combinacional.
Aprobó 37.494 muestras —20.397 sin fallo y 17.097 con fallo—:

- 64 casos de payload X/Z: cada lane RC y RQ produce un fallo comprobado
  también contra una expectativa explícita, seguido de un caso de limpieza.
- Cabeceras correctas y fallos aislados de requester, valor, terminal y enables.
- Inactividad, controles desconocidos, aborto y envenenado.
- 17.352 inyecciones X/Z, una por cada bit de B en tres contextos activos.
- 20.000 registros aleatorios deterministas, incluido tráfico malformado;
  estado final esperado del generador: `d8abb5e0`.

Seis copias RTL defectuosas omiten las contribuciones RC/RQ de los lanes
0/15, cambian el OR final por AND o intercambian los bits 7/8. El revisor
exige la primera discrepancia prevista de cada copia, fuentes compiladas
exactas, cobertura de los cinco bits de fallo y controles de corrupción
del registro de resultados. Los seis mutantes fueron detectados y doce
corrupciones de evidencia fueron rechazadas. La cobertura de bits 4/7/8/9/10
fue 2.370/95/109/59/14.709 muestras; los otros bits permanecieron en cero.

## Correcciones del revisor y respaldo

R1 exigía erróneamente ausencia de WDB; XSIM produjo siete archivos que
suman 343.587 bytes. R2 esperaba que se compilara `metadata_effect_pkg`,
aunque ningún módulo seleccionado lo referencia y Vivado lo omitió.
R3 buscaba el mensaje Tcl de arranque en `simulate.log`; aparece en la
consola, dentro de cada caso. Se conservaron todas las versiones fallidas.
R4 verifica exactamente siete SV más `glbl` por caso, fuentes y archivos
WDB contra el archivo certificado, y un mensaje de arranque por segmento
de consola. No interpreta el contenido WDB ni afirma que esté vacío.
Los controles funcionales, estímulos y RTL no cambiaron; no se repitió Vivado.

Archivo: 1.594.442 bytes, 349 miembros, SHA-256
`104d1b5b7fc34fd3c89e1dd01ba161002bf52ab4883e1e3cbb99f8b1fa0b274e`.
Revisión `...-independent-symbolic-late-fault-leaves-review-r4.json`, SHA-256
`164c959a18226b67b6e3e3b6b977566bcdd458e93ae803421abd415724da31ab`.

## Cola y límites

El coordinador original queda detenido con su fallo de revisión preservado.
`continue_symbolic_late_fault_leaves_review_r4.py` completó la revisión local
y su respaldo sobre la misma tanda. El siguiente ensayo es la
[integración de 183.058 filas](PCIE_SYMBOLIC_MONITOR_PREDECODE_FAULT_LEAVES_R1.md),
después de la cola física vigente.

Un resultado positivo solo permitiría estudiar la integración. Seguirían
pendientes la regresión completa del monitor, invariantes de cinco ciclos,
quiet, cancelación, capacidad, síntesis y replay propios, y timing físico.
No se concede aprobación de PCIe nativo, DDR ni placa.
