# Alternativa separada: un RDIMM 3DS de 128 GiB

Estado 2026-10-02: **geometría generada y revisada; simulación y DIMM sin
calificar**. Conserva el objetivo de cuatro canales de 128 GiB, pero no reemplaza
el perfil de dos módulos Micron de 64 GiB de [V2](MEMORY_QUALIFICATION.md).

## Configuración del catálogo instalado

Vivado 2026.1, DDR4 2.2 rev30, parte provisional `xcvu5p-flva2104-2-i`.
`memparts.csv` contiene 81 entradas, 16 RDIMM. No incluye un RDIMM no-3DS
de 64 GiB 2Rx4; su entrada de 64 GiB usa stack height 2. La alternativa
`M393AAK40B41-CTC` de 128 GiB usa stack height 4.

| Parámetro | Valor comprobado |
| --- | --- |
| Población | Un RDIMM; dos ranks de encapsulados, cuatro dies por stack |
| Componentes / geometría | 8 Gb x4; fila 17, columna 10, BA 2, BG 2 |
| Bus físico / ECC | 72 bits / activado |
| AXI | 37 bits de dirección, 512 bits de datos útiles |
| Selección | Dos CS y dos bits CID; segundo slot vacío |
| Periodo / referencia | 1250 ps (1600 MT/s) / 4000 ps |
| CL / CWL generados | 13 / 11 |

Capacidad útil derivada: `2^(17+10+2+2) × 2 × 4 × 8 = 137438953472 bytes`.
El verificador `tools/check_ntrpu_ddr_geometry.py --profile catalog-3ds128`
comprueba los parámetros del wrapper. Rechaza siete corrupciones y dos
intercambios de perfil; la geometría genérica anterior sigue pasando su perfil.
Esto no prueba calibración, ausencia de alias ni integridad física.

## Ejecuciones y bloqueo de simulación

- `20261002T173116Z-V2-catalog-3ds128`: salida 1, 39,475 s. El script intentaba
  escribir DataMask, selector deshabilitado que el catálogo fija a `NONE`.
  No fue un rechazo del módulo; diagnóstico conservado.
- `20261002T173548Z-V2-catalog-3ds128-r2`: salida 0, 45,483 s; targets y geometría
  revisados, sin advertencias ejecutadas. Fuentes y parámetros en el respaldo.
- `20261002T175303Z-V2-catalog-3ds-example`: salida 0, 77,457 s; ejemplo generado
  en copia nueva. Conserva el aviso Project 1-5713 de BoardPart vacío.

El banco generado define `SAMSUNG`, `DDR4_STACK_X4_4H`, `_8G`, dos ranks físicos
y `LR_WIDTH=2`. El modelo `h4_b_model` recibe CID desde las salidas A/B del RCD.
El wrapper selecciona `S2400P`, `M8192`, `X4` y modelos protegidos para
ModelSim, VCS o NCSim; no se modifican ni descifran.

**El propio banco declara que XSIM no soporta esta simulación Samsung.**
En `sim_tb_top.sv:408`, bajo `XILINX_SIMULATOR`, imprime ese aviso, después
`Test Completed Successfully` y llama a `$finish`. Esa frase no es un PASS
de memoria. No se lanza un banco largo con esa terminación conocida; falta
una combinación autorizada de modelo y simulador compatible.

## Calificación pendiente y evidencia

Faltan SPD exacto verificable, revisión RCD, contrato del modelo, timings
completos, calibración, capacidad dispersa con CID/ranks, ECC y reinicio.
Las descargas de PDFs oficiales Samsung devolvieron HTTP404; sus registros
están archivados. El texto SPD cacheado presenta contradicciones entre
cabecera y tabla; no se trata como dump SPD ni como PDF revisado visualmente.

El catálogo tiene SHA-256
`8af713d50b78aefcaccb509459cdeb0efc4532d93d4579a793c15f520991bd51`.
La tanda de geometría corregida tiene respaldo de 226 archivos, 32.448.012
bytes, SHA-256 `c9395415cb181a23af854c8adbe7df6704cac2e6a54148370314309a29fd92f0`,
verificado localmente a las 17:56 UTC.
El ejemplo tiene 524 archivos, 74.658.973 bytes, SHA-256
`58422ccf3af7b77885056e288de060851de51e5b1cea8074f02a16d37ba784a7`,
verificado a las 18:04 UTC. Las carpetas por tanda están en
`artifacts/ntrpu-board-vivado/`; consultar [estado](VIVADO_STATUS.md).
V2/V3 continúan bloqueados para aprobación de placa; las exploraciones
genéricas de capacidad/ECC y su fallo de reset conservan su alcance previo.
