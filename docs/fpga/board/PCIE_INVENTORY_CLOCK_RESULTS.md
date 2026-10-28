# Buffer interno de reloj: resultados

Revisado el 2026-10-08. El BUFGCE interno pasó la auditoría estructural
y el replay funcional de su netlist exacta en los cinco bancos R12.
**Todavía no hay colocación ni ruteo de esta candidata.**

## Preflight y lectura independiente

El preflight insertó un BUFGCE y un VCC mediante `opt_design` sobre
el DCP de síntesis R12. Terminó con error al comparar listas Tcl de
registros. Los artefactos se conservaron y respaldaron; ese fallo sigue
siendo un fallo del preflight, sin reemplazar su resultado original.

La auditoría posterior reabrió los dos DCP sin optimizarlos. Coinciden
los 10.723 nombres de FDRE y los 1.830 puertos. Tanto las listas Tcl
crudas como los nombres materializados comparan iguales al recargar:
no se reprodujo la desigualdad original y su causa sigue indeterminada.

DCP candidato: `cbb55388614920f547bf3b4e9cfaac3943e40272c69d0e66eca81fbe8d71d440`.
Sólo se añadieron `clk_BUFGCE_inst` y `VCC_1` respecto de R1 optimizado.
Entrada desde `clk`, CE desde VCC y salida exclusivamente a los
10.723 pines C. Sin inversiones ni sincronización de arranque añadida.

La auditoría EDIF contiene 73.353 primitivas y 62.621 celdas
combinacionales, incluyendo el buffer. No hay ciclos combinacionales.
Después de contraer exclusivamente el buffer verificado, coinciden
los nombres, tipos, parámetros y conexiones de toda la lógica restante.
Pasaron ocho controles heredados y trece controles específicos de reloj;
mutaciones de parámetros y conexiones cambian la huella lógica.

## Restricciones, biblioteca y límites

Período 4 ns, E/S cero, sin excepciones ni `HD.CLK_SRC` externo supuesto.
Las restricciones permanecieron iguales durante la auditoría. DRC informa
cero violaciones, con advertencia de comprobaciones incompletas en OOC.
Se conservan dos avisos Netlist 29-101 y uno DRC 23-814.

La fuente instalada BUFGCE declara `SIM_DEVICE="ULTRASCALE"`;
[UG974 2026.1](https://docs.amd.com/r/en-US/ug974-vivado-ultrascale-libraries/BUFGCE)
documenta `"ULTRASCALE_PLUS"`. La netlist no declara parámetros.
Se preservó la biblioteca, con hash `2e0b6ec0cda231b17ec3b7335fc09711447fdd37c6ed1f516d785f96314f9431`,
y se comprobó su fuente antes y después del replay. XSIM usó
`unisims_ver`; no se demostró identidad entre esa fuente y su biblioteca
precompilada ni se modificó código de AMD.

Timing **sin colocar**: WNS −7,160 ns, TNS −70.329,727 ns;
WHS −2,702 ns, THS −68.385,430 ns. Fallan 15.852/29.676 endpoints
de setup y 29.080/29.676 de hold. No comparar estas cifras como si fueran
un nuevo ruteo o una frecuencia de placa.

## Replay y respaldo

Netlist SHA-256 `cc4d24440ee4aadd565273d17ac503df2167f9be4b7f66d5e56a989f3797dc61`.
Los cinco bancos contienen 78.712 filas de entrada: principal 22.330,
ownership 46.356, RQ 8.850, prefix 544 y lanes 632. Todas las filas
públicas, cancelaciones y estados posteriores a fallos coinciden con
R12 RTL. Pasaron los 30 controles de evidencia. Sin RTL del DUT ni SDF;
arranque 200 ns, muestreo 1 ns, cinco arranques XSIM sin ondas automáticas.
Duración real: **603.879 s**;
WDB conservados: **51125481 bytes**. No es throughput
de hardware ni una comparación causal de rendimiento.

| Tanda | ID | SHA-256 del archivo |
| --- | --- | --- |
| preflight_stop | `20261008T205100Z-V4-ffn-supervised-pcie-inventory-clock-preflight` | `c65c3ec2d699de3597e1ec4dfefde592936602c38b010fe9301c5fa1df9512e0` |
| read_only_audit | `20261008T205958Z-V4-ffn-supervised-pcie-inventory-clock-audit` | `7f372edcf1cae58ddfc840d159722e02b5f41c378b9823bb63d885f08d3408ad` |
| binary_netlist | `20261008T211339Z-V4-ffn-supervised-pcie-inventory-clock-binary-netlist` | `1fb5c8e42637144fbb1f3267441fec45c7956666e2a07625044da729819e0e0e` |

Respaldos locales bajo `D:/AI/ntransformer/artifacts/ntrpu-board-vivado/backups/`
y en NFS, certificados por miembros y SHA-256. Los intentos detenidos
y las revisiones corregidas permanecen conservados.

## Siguiente paso

Preparar una implementación aislada desde el DCP optimizado con BUFGCE.
No repetir síntesis u optimización; conservar 4 ns y E/S cero. Registrar
ubicación del buffer, distribución del reloj, rutas incompletas,
setup/hold global y registro→registro, DRC, metodología y fanout.
El DCP implementado exigirá su propio readback y replay. Permanecen
abiertos frontera nativa, reset/CDC, DDR/512 GiB, inferencia y placa.
