# Captura de drenaje R2: diagnóstico y siguiente revisión

La tanda `20261006T064531Z-V4-ffn-supervised-drain-sections-r2` terminó por
el límite externo de 600 segundos: código 124, duración 600,009 s. Quedó
respaldada localmente y en NFS, con SHA-256
`58b157d1e3a1edeb66dc335057a711d90a84f7166de3fc5c9bd61327814c2ddd`.

El primer perfil produjo un VCD de 4.560.201 bytes. En `1-5000`, XSIM
informó `Missing VCD objects` al consultar `/tb_ntrpu_drain_sections/*`.
El archivo de ese perfil quedó en cero bytes. El error Tcl no terminó el
intérprete de XSIM; el límite externo detuvo la tanda. No se ejecutaron los
dos perfiles de 37 ns. Esta tanda es diagnóstica, no una validación completa.

`tools/ntrpu_board_drain_sections_r3.tcl` está preparado para:

- Crear proyecto, elaboración y proceso XSIM independientes por perfil.
- Enumerar la jerarquía real mediante `get_scopes`, registrar el scope
  seleccionado y exigir exactamente un banco de drenaje.
- Exigir señales en los cinco scopes de captura, conservar los umbrales
  del VCD y salir de XSIM inmediatamente ante errores de instrumentación.
- Conservar RTL, parámetros, estímulos y criterios HDL.

La enumeración usa la interfaz documentada en
[AMD UG835, get_scopes](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_scopes).
No está demostrado todavía si el nombre de la jerarquía o la reutilización
del proyecto explica el fallo original.

El intento de lanzar R3 fue rechazado por el preflight, antes de crear una
tanda: estaba activa una ruta externa en
`ntrpu-side/overlap-axi-bankplan/20261006T064514896795Z-route`.
Esa ejecución se conserva. R3 queda pendiente de ejecución cuando el
preflight permita usar Vivado, y luego de revisión independiente de las
cuatro trazas. Los 12 CDC-11 de la síntesis anterior siguen abiertos.
