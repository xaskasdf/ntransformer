# Revisión y comparación física independiente RQ/E/D

Fecha: 2026-10-12 UTC. Preparada después de la
[implementación y reapertura comparables](PCIE_SYMBOLIC_MONITOR_PREDECODE_MATCHED_PHYSICAL_R1.md).
No aporta todavía resultados físicos de la pareja.

## Política conservada

Se reutilizan sin cambios el comprobador físico R2 y el comprobador de
subconjuntos temporales. Sus verificaciones archivadas rechazaron 22 y 18
corrupciones, respectivamente. Los wrappers nuevos solo cambian identidades
de diseño, marcadores, propietarios y nombres de evidencia; conservan las
funciones de revisión originales.

Para cada miembro se comprobarán archivos, DCP de entrada y ruteado,
restricciones guardadas, biblioteca BUFGCE, inventarios, reloj, DRC,
metodología, avisos y cobertura temporal. Las diez comparaciones entre
implementación y reapertura incluyen ocho grupos de frontera y E→D setup/hold.
Se conservan endpoints, aritmética de slack y separación lógica/interconexión.

## Comparación de los dos resultados

Después de ambas revisiones, el comparador exige la misma versión completa
de Vivado, biblioteca, comandos de implementación y restricciones temporales.
Reporta recursos y diferencias de setup/hold global y de los diez grupos,
incluidas las regresiones. Las posiciones de buffers y réplicas se registran,
sin tratarlas como prueba de equivalencia.

R2 corrige dos enlaces de formato detectados antes de ejecutar R1: los dos
grupos E→D están separados de los ocho grupos de frontera, y las claves de
los XDC guardados contienen el nombre del top. Se normaliza exclusivamente
ese prefijo en los nombres de tres archivos; sus comandos quedan intactos.
La verificación con un resultado histórico detecta cambios de período y
rechaza archivos faltantes, adicionales o de otro diseño. R1 queda conservado
como preparación no ejecutada.

## Cola y alcance

`continue_symbolic_monitor_predecode_matched_physical_review_r2.py` espera
ambas reaperturas revisadas. Ejecuta únicamente comprobaciones locales y
respalda revisión y comparación en D: y NFS. Un fallo exige inspección.

El resultado habilita los replays propios de las dos netlists ruteadas;
esos replays siguen siendo obligatorios. No se convierte una salida 0,
coincidencia de informes o slack negativo en timing aprobado. Permanecen
los límites OOC, ausencia de SDF y falta de validación PCIe/DDR/placa.
