# PCIe simbólico C/E/D: primera implementación

Implementación experimental del 2026-10-09 de la
[partición propuesta](PCIE_SYMBOLIC_FRONTEND_R2_PLAN.md). No está calificada
por simulación, síntesis ni implementación física al escribir este contrato.

## Fronteras y propiedad

`ntrpu_pcie_symbolic_control_r2.sv` deriva del S1 R1 revisado mediante
sustituciones registradas. Conserva control, stalls, contadores, framing,
propiedad y selección R3. Captura operaciones; no contiene tabla ni
composición de descripciones. Sus 256 decisiones de commit viajan completas.

`ntrpu_pcie_symbolic_effects_r2.sv` prepara consumos ordenados sin consultar
descripciones. Incluye el consumo retenido cuando corresponde y el consumo
actual para commit. El registro conserva la decisión de omitir composición
cuando el contexto solo debe copiarse.

`ntrpu_pcie_symbolic_descriptions_r2.sv` posee la tabla y el contexto de
payload por valor. Procesa un origen por ciclo, consulta su propio estado
anterior, compone cada resultado una vez y aplica RC antes de las dos
admisiones RQ. TC y atributos se consultan en D antes del commit RQ del origen.

## Latencia y cancelación

La salida de `ntrpu_pcie_symbolic_frontend_r2` corresponde a S1 R1 con dos
ciclos adicionales: C captura N, E procesa N en el siguiente flanco y D lo
retira en el tercero. El registro válido incluye orígenes inactivos para
diagnóstico. Reset frío cancela todas las etapas en el mismo flanco; reset
de usuario y pérdida de link conservan el procesamiento posterior al fallo.

`control_pending` acompaña al origen retirado; durante reset frío refleja
actividad cruda actual, como S1 R1. `pipeline_pending` agrega actividad
actual y efectos en C/E/D. Ninguna señal constituye un certificado completo
de vacío: aplicación numérica, bytes y validación posteriores siguen pendientes.

## Puertas de validación

El primer banco conserva el scoreboard S1 contra R7/modelo simbólico y añade
comparación exacta de 2.498 bits, 256 filas y descripción retenida por origen.
Contabiliza captura, retiro y cancelación, sin filtrar filas posteriores a
fallos; dos flancos fijos drenan los orígenes finales. Ejecuta los seis bancos
existentes, incluidos X/Z. La reasociación de efectos debe superar estos casos;
la prueba algebraica sobre efectos canónicos no basta por sí sola.

Después se requieren casos dirigidos de etapas ocupadas, mutantes reales,
medidas C/E/D/conjunto a 4 ns sin excepciones y replay de su propia netlist.
Pasar el banco inicial no autoriza declarar cerrado el monitor ni la placa.
