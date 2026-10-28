# Traspaso R1: diagnóstico de potencia del ejemplo DDR

Fecha: 2026-10-15. Estado actualizado: **ejecutado**; R1 se detuvo en
la comparación NAME/TOP. El sucesor Tcl R2 exportó el diagnóstico con
confianza baja. Consultar [resultados y evidencia](POWER_PROBE_RESULTS_R1.md).
Los tres intentos están respaldados y restaurados; la cola se reanudó
a las 21:07:39 UTC sin alterar sus dependencias.
Las instrucciones y hashes originales siguientes se conservan como registro
del traspaso; la ejecución usa una tanda independiente por intento.
Continúa la [revisión eléctrica y de memoria](MEMORY_POWER_REVIEW_R1.md).

## Alcance y dependencia

Abrir un DCP enrutado de un canal DDR4 genérico y exportar su estimación
de potencia con la actividad heredada y los valores por defecto. El diseño
es `example_top`, parte `xcvu5p-flva2104-2-i`, con 154 puertos. No incluye
cuatro canales, PCIe ni cómputo NTRPU; tampoco califica el DIMM F1.

La ejecución corresponde al hilo principal, cuando disponga del único
slot de la [cola Vivado](VIVADO_QUEUE.md). No insertar un coordinador
independiente ni interrumpir validaciones. La entrada de cola consultada,
fechada 20:06 UTC, registra FFN R4 activa; comprobar el estado vigente al
programar. Este documento no cambia dependencias ni reserva recursos.

## Entradas verificadas localmente

Rutas relativas a la raíz del repositorio:

| Entrada | Ruta |
| --- | --- |
| Script | `build/board-electrical-study-20261015-r2/power_probe_r1.tcl` |
| DCP | `artifacts/ntrpu-board-vivado/20261002T125230Z-V3-ddr-route/20261002T125230Z-V3-ddr-route/output/route_attempt.dcp` |
| Respaldo documental | `artifacts/ntrpu-board-electrical-20261015-r2/electrical-inputs-r2.zip` |

SHA-256 y tamaños comprobados nuevamente antes de escribir este traspaso:

```text
script (1944 bytes):
1bdc28a229bf0b32a7cde294d6abe85e11ceae73e0dbd5795fdf0f6505e3f7b1
DCP (26756743 bytes):
161d111ce97d471d461328e615c96d32232b6e6c101673de210cb75b282f205c
ZIP (1180122 bytes):
1eec4da1be761973bac22f41d0cb63a497f2cb760132650b04398b54e14b77fa
```

El ZIP contiene el script bajo su misma ruta relativa, pero no el DCP.
La restauración previa figura en [BACKUP.json](../../../artifacts/ntrpu-board-electrical-20261015-r2/BACKUP.json).
Transportar script y DCP y verificar ambos hashes en destino. Si se necesita
restaurar, usar una carpeta nueva; conservar el paquete sellado.

## Ejecución propuesta en Linux

Usar una sesión nueva de Vivado 2026.1 con su entorno y licencia disponibles.
Sustituir las tres rutas del ejemplo. `parent` debe existir; `reports` debe
permanecer ausente hasta que el script lo cree. Ejecutar desde un shell
dedicado tras obtener el turno de la cola.

```bash
set -euo pipefail
probe=/ruta/absoluta/power_probe_r1.tcl
dcp=/ruta/absoluta/route_attempt.dcp
parent=/ruta/absoluta/tandas
printf '%s  %s\n' \
  1bdc28a229bf0b32a7cde294d6abe85e11ceae73e0dbd5795fdf0f6505e3f7b1 "$probe" \
  161d111ce97d471d461328e615c96d32232b6e6c101673de210cb75b282f205c "$dcp" \
  | sha256sum -c -
batch=$(mktemp -d "$parent/power-probe-r1-XXXXXXXX")
sha256sum "$probe" "$dcp" > "$batch/inputs.before.sha256"
cd "$batch"
rc=0
timeout --signal=TERM --kill-after=60s 20m \
  vivado -mode batch -source "$probe" \
  -log "$batch/vivado.log" -journal "$batch/vivado.jou" \
  -tclargs "$dcp" "$batch/reports" > "$batch/console.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$batch/exit_code.txt"
sha256sum "$probe" "$dcp" > "$batch/inputs.after.sha256"
sha256sum -c "$batch/inputs.before.sha256" > "$batch/input-integrity.txt" 2>&1
printf 'Tanda: %s; salida Vivado/timeout: %s\n' "$batch" "$rc"
exit "$rc"
```

El Tcl solicita `general.maxThreads=2`; no impone afinidad de CPU al
proceso completo. El límite inicial de 20 minutos es operativo, no una
predicción de duración. Un timeout, fallo de licencia o fallo de apertura
requiere revisión; no habilita reintentos automáticos. Este bloque y el
script estaban **sin ejecución nativa validada al preparar el traspaso**.
El resultado posterior y la corrección de identidad constan en la ficha
enlazada al inicio; no se modificó el script R1 archivado.

## Revisión y respaldo de la tanda

Aceptar únicamente como **diagnóstico exportado** cuando:

1. La salida es cero y aparece `NTRPU_POWER_DIAGNOSTIC_EXPORTED`.
2. Los hashes de entrada permanecen intactos; parte, top y puertos cumplen
   el contrato y `scope.txt` conserva el alcance genérico.
3. Existen, con contenido, `clocks.rpt`, `utilization.rpt`, `power.rpt` y
   `power.xml`; el XML se puede analizar. Registrar advertencias y errores.
4. La revisión transcribe versión, actividad y confianza, condiciones
   térmicas, potencia estática/dinámica y demanda por rail. Usar `null` si
   una magnitud no está disponible, sin completar valores por inferencia.
5. Se archivan logs, informes, revisión e identidad de las entradas en una
   tanda nueva; se verifica SHA-256 y una restauración antes de considerarla
   respaldada. Conservar también las tandas fallidas y su causa.

Mantener `board_power_qualified=false` y `exact_dimm_qualified=false` incluso
si la exportación termina correctamente. No multiplicar los vatios por
cuatro ni sumarlos a la campaña HBM para dimensionar fuente o refrigeración.

## Revisión previa de actividad

La inspección local del 2026-10-15 encontró una diferencia entre el banco
funcional y la entrada implementada. El [registro de revisión](../../../build/board-power-activity-preflight-20261015-r1/REVIEW.json)
conserva ocho hashes, parámetros XCI y la cadena de entradas declarada en
las invocaciones síntesis → colocación → ruteo. No es una reapertura nativa.

| Aspecto | Banco de capacidad | Fuente del ejemplo implementado |
| --- | --- | --- |
| Jerarquía RTL | `ntrpu_ddr_capacity_tb/dut`, MIG directo | `example_top/u_ntrpu_ddr_geometry`, más lógica del ejemplo |
| `System_Clock` del XCI | `Differential` | `No_Buffer`, con buffers externos |
| Reset | `sys_rst` directo al MIG | Secuenciador `ntrpu_startup` |
| Tráfico | Tareas del banco | Generador `u_ddr4_v2_2_30_axi_tg_top` |

Ambos XCI comparten IP DDR4 2.2 rev30, período DDR de 1250 ps,
referencia de 4000 ps, CL12, AXI de 37/512 bits, ECC y dos slots. La única
diferencia de valores en `component_parameters` es `System_Clock`; esto
no demuestra equivalencia del diseño completo ni de nombres tras síntesis.

Además, [`read_words`](../../../fpga/tb/ntrpu_ddr_capacity_tb.sv) espera
dos flancos descendentes de UI antes de activar `rready` para cada beat
y lo desactiva después. El banco recorre direcciones dispersas, máscaras
y ráfagas cortas. Su PASS funcional no caracteriza lectura sostenida.
El lanzador actual usa `runtime all` y no registra SAIF por ventanas.

Decisión: conservar el diagnóstico inicial sin SAIF. Antes de añadir
actividad, preparar una pareja simulación/implementación correspondiente:
para este DCP, reproducir el `example_top` archivado y sus IP, buffers,
secuenciador y generador. Para tráfico AXI programable, preparar un wrapper
con ese puerto y obtener su propio DCP; no atribuirle el del ejemplo.
Una anotación parcial del MIG requeriría revisión explícita de equivalencia
y dejaría el resto del diseño con actividad sin medir.

## Ventanas para la estimación posterior

Estas ventanas requieren un banco compatible; todavía no están implementadas.
El [paquete de actividad R1](POWER_ACTIVITY_R1.md) prepara primero una captura
acotada del PRBS del ejemplo archivado; su elaboración y ejecución están pendientes.

| Ventana propuesta | Condición y evidencia necesaria |
| --- | --- |
| Reposo calibrado | Calibración completa, resets liberados, sin transacciones pendientes ni nuevas; conservar refresh y relojes normales |
| Lectura sostenida | Datos inicializados fuera de la ventana; consumidor listo continuamente; registrar bytes aceptados, stalls y patrón de acceso |
| Tráfico mixto | Lecturas/escrituras y backpressure declarados; comprobar datos, proporción realmente transferida y actividad por rank |
| Arranque o reinicio | Archivo independiente; registrar aceleraciones del modelo. El fallo de warm reset continúa abierto |

Registrar inicio, fin, unidad temporal, ciclos UI, transacciones pendientes
en ambos bordes y cobertura de refresh. Medir duración desde los eventos,
sin reutilizar un tiempo fijo de otra ejecución. Comparar subventanas y
ampliar si la actividad no es estable. Mantener ECC activo; separar la
inyección deliberada de errores del tráfico normal.

XSim permite abrir SAIF, seleccionar objetos, ejecutar la ventana y cerrar
el archivo; requiere elaboración con depuración adecuada:
[AMD UG900 2026.1](https://docs.amd.com/r/en-US/ug900-vivado-logic-simulation/Generating-SAIF-Dumping).
Seleccionar solo el diseño objetivo, excluyendo el banco y los modelos
externos del DIMM de la estimación FPGA.

Al importar, comprobar la jerarquía real y guardar las redes sin
correspondencia mediante `read_saif -out_file`; `-strip_path` elimina un
prefijo, no demuestra equivalencia ni reconstruye lógica faltante:
[AMD UG835 2026.1](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/read_saif).
No existe todavía SAIF, cobertura medida ni estimación de operación validada.
La exportación diagnóstica posterior no cambia esa condición. Conservar
actividad anotada frente a inferida/default y condiciones térmicas antes
de valorar el informe. Repetir luego con la integración real.

En paralelo faltan SPD, revisión, RCD y consumo completo del DIMM elegido.
Esas evidencias alimentarán la
[base eléctrica](ELECTRICAL_BASELINE_R1.md) antes de seleccionar reguladores
y cerrar el [diseño mecánico](MECHANICAL_LAYOUT_R1.md).
