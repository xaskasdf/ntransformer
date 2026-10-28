# Coste del banco de propiedad RC

Observación del 2026-10-08, tanda
`20261008T145423Z-V4-ffn-supervised-pcie-inventory-join-rc-ownership-rtl`.
El banco completo terminó en 645,496 s. El suplemento de propiedad consumió
354 s de `run all`, frente a 45,280 s de CPU comunicados por XSIM, y produjo
un WDB de 119.024.214 bytes. Pasaron sus 1.153.296 comparaciones.

Los procesos avanzaban y el WDB crecía en el volumen NFS. Es razonable
investigar captura de ondas y E/S, pero estas observaciones no aíslan su
coste ni demuestran que desactivarlas acelere la ejecución.

## Experimento para la siguiente tanda

Usar un script de inicio propio que sólo cargue la simulación (`run 0ns`),
sin `add_wave` ni `log_wave`, mediante `xsim.simulate.custom_tcl`. Conservar
`run all`, los mismos archivos RTL/TB y vectores, todos los contadores,
aserciones, logs y controles de evidencia. No cambiar la tanda R7 congelada.

AMD documenta la sustitución del Tcl inicial y las opciones de registro en
[UG900 2026.1: opciones de simulación](https://docs.amd.com/r/en-US/ug900-vivado-logic-simulation/Vivado-Simulator-Simulation-Options).
`log_wave` habilita el registro de los objetos seleccionados en WDB;
[UG900: registro de ondas](https://docs.amd.com/r/en-US/ug900-vivado-logic-simulation/Using-the-log_wave-Tcl-Command).

Verificar primero las propiedades reales de Vivado instalado y que el
script sustituto se ejecutó. Comparar contadores, casos X/Z y resultado
contra R7; medir tiempo de pared, CPU, tamaño WDB y respaldo. Guardar la
captura detallada de R7 como diagnóstico. Un ahorro de tiempo no permite
omitir el banco completo ni promover el diseño sin su propia netlist.
