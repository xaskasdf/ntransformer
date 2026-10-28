# R14: sumas registradas, resultados

Revisado el 2026-10-09 UTC. R14 pasa RTL completo y replay de su propia
netlist. El resultado de timing es mixto: **WNS −6,442 ns**, mejor en
0,626 ns que R12, pero **TNS −71.772,922 ns**, peor en 943,938 ns.
No cierra 4 ns. Se conserva R12 como referencia preferida y R14 como
experimento con menor violación máxima; no hay promoción física o nativa.

## Cambio y validación

Registra las sumas finales RC de 15 bits y RQ con signo de 16 bits, junto
con la base truncada de 21 bits. Conserva la reducción por carriles de R12,
selector R3, capacidad, orden, estado después de fallo y latencia pública.
La referencia sigue siendo exclusiva del banco, sin alimentar el DUT.

Pasan 22.339 filas/68 escenarios principales, 5.718.784 comparaciones de
palabras y todos los bancos auxiliares R12. Los siete mutantes de integración
y 17 controles de evidencia son rechazados. El acumulador secuencial pasa
20.265 muestras y 62 inyecciones X/Z; cubre pausa, reset, estabilidad antes
del flanco, 218 resultados negativos, 10.512 sobre el límite de protocolo
y 28 sobre el almacenamiento bajo. El mutante de signo falla en la primera
resta dirigida; seis controles verifican el lector de evidencia.

Los cinco bancos binarios RTL/netlist comparan **78.712 filas y 30 controles**:
principal 22.330, ownership 46.356, RQ 8.850, prefijos 544 y carriles 632.
Coinciden estados públicos y cancelaciones, incluidos los estados posteriores
a fallo. El DUT de replay es la netlist propia, sin compilar productor,
wrapper ni validador RTL. Arranque 200 ns, muestreo 1 ns y sin SDF;
equivalencia finita de banco, no frecuencia física ni prueba universal.

## Síntesis comparable y destinos

Vivado 2026.1, `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, OOC sin colocar:
4 ns, retardos I/O cero y ninguna excepción. Tcl idéntico salvo las rutas
de candidata y la incorporación del acumulador. R14 usa **52.371 LUT,
10.758 FF y 1.902 CARRY8**: +137 LUT/+29 FF frente a R12; cero latches,
BRAM, URAM y DSP. Los 52 FF del acumulador están identificados.
Setup falla en 15.870/29.942 endpoints; hold en 1.892/29.942, WHS −0,075 ns.

La lectura del DCP reproduce sus recursos, timing global y restricciones
antes/después. Clasifica los 10.758 registros una vez y verifica 56 consultas,
con hasta veinte caminos por consulta. Los grupos son filtros de reporte.

| Destinos | FF R14 | Setup R12 ns | Setup R14 ns | Registro→registro R14 ns |
| --- | ---: | ---: | ---: | ---: |
| Bytes y sumas | 52 | -7.068 | -6.442 | -6.402 |
| Tags/lecturas | 278 | -4.407 | -4.491 | -4.491 |
| Tabla | 7680 | -5.795 | -5.583 | -5.543 |
| Contexto RC/RQ | 1421 | -4.180 | -3.896 | -3.856 |
| Metadatos canónicos | 176 | -3.708 | -3.488 | -3.438 |
| Control | 126 | -1.884 | -2.109 | -2.109 |
| Captura | 559 | 3.488 | 3.488 | 3.527 |
| Retirada/validación | 466 | -3.126 | -3.126 | -3.126 |

La categoría de bytes cambia de almacenamiento al introducir sumas y base;
no representa los mismos endpoints individuales. La peor ruta R14 va de
`rc_user[76]` a `producer/byte_counter/saved_removed_reg[12]/D`: 10,453 ns,
40 niveles, 2,575 ns de lógica y 7,878 ns de ruta estimada. La cadena incluye
selección y metadatos antes de reducir RC. Ni las muestras enumeran todos
los caminos ni el mapeo sin colocar demuestra una mejora física.

El DCP pasa auditoría combinacional con ocho controles. Se conservan los
avisos de puertos sin uso, registros de captura compartidos, reloj externo
sin contexto OOC y DRC incompleto. No hay errores ni avisos críticos;
doce categorías `check_timing` a cero no eliminan setup/hold fallidos.

## Evidencia y siguiente trabajo

DCP `d07c5012b4483ce3311f3ab8a95a8d3541255ec0d171f2c54d749afff3ad3f24`.
Netlist `6b13805dbf05e1a5f21df6d3e1f09af51a2733f18f034b7aa3847b43ebd38bc7`. Comparación `b9b89116c80d11c148a024f5f67ae0fd72ff5391675d7e82bd7be93eea397613`.

| Etapa | Tanda | Segundos | SHA-256 del archivo |
| --- | --- | ---: | --- |
| full_rtl | `20261009T005620Z-V4-ffn-supervised-pcie-inventory-join-registered-sum-rtl` | 368.858 | `4db69fe0c6529096792c805ca5553c194b649f77d61f1e262999ccc3dc158521` |
| synthesis | `20261009T063336Z-V4-ffn-supervised-pcie-inventory-join-registered-sum-synth` | 310.212 | `fdcb98958ce3377b2156dd64b8e72eed32ddffa4602106ae0e237da9124979fb` |
| binary_rtl | `20261009T064102Z-V4-ffn-supervised-pcie-inventory-join-registered-sum-binary-rtl` | 139.015 | `af326516a08e15bd641ec38d25b37dc1572bcdbd3f7e2f6cc08a57c3ec2cae38` |
| binary_netlist | `20261009T091210Z-V4-ffn-supervised-pcie-inventory-join-registered-sum-binary-netlist` | 760.156 | `ad85f5ea9fabe36ebca6db7aa1cbd25558a2d0075da55df2a5fa1dd1a6ec6eb8` |
| path_census | `20261009T093135Z-V4-ffn-supervised-pcie-inventory-registered-sum-path-census` | 193.305 | `c499c829d1129564b0a5b67efa22fd8769b90ab5a5e0e2033d3fb17b066681e5` |

Fuentes, archivos y miembros verificados en D:/NFS mediante
`artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`. Revisiones y
configuraciones en `build/cloud-setup/20261002-board/`; R104 enlaza hashes.
Los tiempos XSIM (368.858 s RTL,
139.015 s binario RTL y
760.156 s netlist) no son throughput de hardware.

Continuar con [consulta y transición segmentadas](PCIE_INVENTORY_REGISTERED_SUM_FOLLOWUP.md).
El [estudio binario inicial](PCIE_METADATA_COMPOSITION_MODEL.md) pasa 131.600
casos/518.208 agrupaciones y cinco controles; todavía no cubre selección,
generaciones, contexto retenido ni X/Z. La siguiente arquitectura aún no
está implementada. Permanecen abiertos
segmentación completa, endpoint nativo, reset/CDC, DIMM exacto, cuatro DDR/
512 GiB, cómputo y diseño físico según [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
