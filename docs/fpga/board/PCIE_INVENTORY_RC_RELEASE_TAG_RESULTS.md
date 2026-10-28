# R13: tag RC directo, resultado y decisión

Revisado el 2026-10-09 UTC. R13 pasó el banco RTL completo y el replay de
su propia netlist. **Empeora el timing OOC y no se promueve:** WNS −7,268 ns,
0,200 ns peor que R12, con 74 LUT adicionales. R12 sigue como base preferida;
ambas incumplen el período de 4 ns.

## Cambio y comprobación funcional

R13 sustituye exclusivamente el tag RC enmascarado por los bits directos
de cada descriptor, conservando las dos guardas de liberación. Selector R3,
reducción por carril, capacidad, latencia y orden RC→RQ permanecen intactos.

El banco completo conserva **22.339 filas/68 escenarios** y
**5.718.784 comparaciones de palabras** principales, incluidos estados
posteriores a fallo. Pasan los bancos heredados de selección, ownership,
ancestry, admisión RQ, aritmética/máscaras, suma RC e integraciones. Los siete
mutantes de integración y sus 17 controles de evidencia siguen rechazados.

La nueva prueba de guardas contiene **12.832 casos, 7.680 X/Z**. Observa
17.919 liberaciones válidas y 33.409 diferencias de bits enmascarados;
son ocurrencias, no tags ni paquetes únicos. Usa estado inyectado únicamente
en la unidad combinacional. La pertenencia se compara con una recurrencia
secuencial independiente; admisión usa la función antigua intacta con tags
enmascarados y el bitmap usa reconstrucción explícita. No sustituye tráfico
alcanzable ni la integración por entradas públicas. Un productor mutante
que invierte el bit 0 falla en el primer caso; pasan seis controles adicionales.

Los cinco bancos binarios comparan **78.712 filas y 30 controles**:
principal 22.330, ownership 46.356, RQ 8.850, prefijos 544 y carriles 632.
Todas las filas públicas, cancelaciones y estados posteriores a fallos
coinciden entre referencia, RTL R13 y su netlist. No se compila productor,
wrapper ni validador RTL como DUT del replay. Arranque 200 ns, muestreo
1 ns, sin SDF: comprobación funcional, no medida física de frecuencia.

## Medidas comparables de síntesis

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, OOC sin colocar.
La comparación verifica Tcl idéntico salvo rutas de candidata, reloj 4 ns,
E/S cero y ausencia de excepciones válidas o ignoradas.

| Métrica | R12 | R13 |
| --- | ---: | ---: |
| LUT | 52.234 | 52.308 |
| FF | 10.729 | 10.729 |
| CARRY8 | 1.902 | 1.902 |
| WNS ns | −7,068 | −7,268 |
| TNS ns | −70.828,984 | −72.940,734 |
| WHS ns | −0,075 | −0,075 |

R13 falla setup en 15.863/29.806 endpoints y hold en 1.931/29.806;
THS −139,703 ns. Cero latches, BRAM, URAM y DSP. El DAG del DCP tiene
75.559 primitivas y 94.029 nets, sin ciclos, con ocho controles de auditoría.
Se conservan advertencias de constantes, pines no conectados, tres registros
de captura eliminados, metodología y cobertura OOC incompleta; no hay
errores ni advertencias críticas de síntesis. `check_timing` da cero en sus
doce categorías; esto no elimina las violaciones setup/hold.

En ambas muestras de veinte caminos, todas las fuentes son `rc_user[76]`:
18 destinos de bytes y dos de rango. El peor termina en
`producer/remaining_bytes_reg[20]/D`. R12: 11,079 ns y 43 niveles;
R13: 11,279 ns y 45 niveles, con 2,801 ns lógicos y 8,478 ns de ruta
estimada. Es una muestra correlacionada, no un censo de todos los cuellos
de botella. El ruteo Quick anterior pertenece a otro experimento.

## Reproducción y respaldo

Netlist reproducida: `2aec10fd20d19f8cda924deb1b7324754ed631324ce721cd77d01f7bb9eaeed3`.
DCP: `f9856b093f08965607f65f96ef450ccee04abac561d776aee4ec4a41d6265891`.

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T233431Z-V4-ffn-supervised-pcie-inventory-join-rc-release-tag-rtl` | 372.585 | `d1b91fc10e43614a8072c2700c170b542ab92b1e59a7c9a68e43f55b7786cfd1` |
| `20261008T234355Z-V4-ffn-supervised-pcie-inventory-join-rc-release-tag-synth` | 305.445 | `069eaa8c5a95b0d15906ace86e1d52ad211345815b7ac211f79f0a67cf51a6ed` |
| `20261008T235103Z-V4-ffn-supervised-pcie-inventory-join-rc-release-tag-binary-rtl` | 139.749 | `635eaba6a7245aa76c49457c11ca4ae5b4607ace1d8e390bca4e7a96b8e399de` |
| `20261008T235622Z-V4-ffn-supervised-pcie-inventory-join-rc-release-tag-binary-netlist` | 689.286 | `2732f6d03685114c4022a05398e0b8221f3df340f927b00cb42bfa884190d7f2` |

Fuentes, miembros y archivos verificados en D:/NFS, mediante
`artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`. Revisiones en
`build/cloud-setup/20261002-board/`; la cola R102 enlaza sus hashes.
El RTL completo usa 19 arranques XSIM y los bancos binarios cinco cada uno.
Replay netlist: 689.286 s y
59535945 bytes WDB. Los tiempos observados no
demuestran aceleración causal ni rendimiento de hardware.

Siguiente trabajo: [censo de destinos críticos R12](PCIE_INVENTORY_RC_RELEASE_TAG_FOLLOWUP.md).
No se aprueban segmentación completa, integración nativa, reset/CDC,
DIMM exacto, DDR/512 GiB, inferencia integrada ni diseño físico.
