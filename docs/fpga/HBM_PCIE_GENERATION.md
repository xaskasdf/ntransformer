# PCIe del nodo HBM: generación revisada

Tanda `20261009T000516217615Z/probe1` (8 de octubre en Chile). Continúa la
[recalificación HBM del lector](TRACKER_HEAD_NATIVE.md) y prepara el
[contrato del puente](HBM_PCIE_BRIDGE_CONTRACT.md). Estado:
**`VU47P_XDMA_GENERATION_REVIEWED_NOT_INTEGRATED`**.

## Alcance de la evidencia

Vivado 2026.1, build 6511674, generó XDMA 4.2 rev4 para
`xcvu47p-fsvh2892-2-e` en 170,920 segundos. Salida 0, sin timeout ni
advertencias/errores en consola. Se revisaron los perfiles `serial` y `pipe`,
35 parámetros solicitados por perfil, ocho XCI y 16 wrappers sim/synth.
Los parámetros de configuración de ambos perfiles solo difieren en `pipe_sim`.

Esta tanda exportó IP y proyectos de ejemplo. No ejecutó tráfico PCIe,
síntesis ni implementación, y no contiene el controlador HBM integrado.
El perfil PIPE es para simulación. La generación serial prepara la futura
implementación, pero no demuestra enumeración, integridad de señal o P2P físico.

## Interfaces y mapa generado

| Elemento | Configuración comprobada |
| --- | --- |
| Enlace | Gen3 x16, referencia de 100 MHz |
| DMA | AXI-Stream de 512 bits; H2C0 y C2H0/1 |
| Descriptor bypass | Lectura `0001`, escritura `0011`; direcciones de 64 bits |
| Estado | `c2h_sts_0/1[7:0]` exportados |
| BAR0 | AXI-Lite, 64 KiB, 32 bits |
| BAR1 | DMA/MSI-X, 64 KiB, 32 bits; tabla `0x8000`, PBA `0x8fe0` |
| BAR2/3 | Bypass AXI-MM, 1 MiB, 64 bits, no prefetchable |
| Traducción bypass | Base AXI `0x0000000100000000`; datos 512, dirección 64, ID 4 bits |

El reparto BAR procede del bloque **PCIE4C interno** y sus wrappers. Los
parámetros genéricos BAR del nivel XDMA superior contienen otros valores y
no bastan para deducir el mapa efectivo. La base PCIe asignada por el host
durante enumeración sigue pendiente de prueba; no equivale a la base AXI.

## Hallazgos para la integración de placa

- `axisten_freq=250` y el XDC OOC fija `axi_aclk` a 4 ns. Sin embargo, la
  etiqueta `X_INTERFACE_PARAMETER` del wrapper declara 125 MHz. Conservar
  esta discrepancia visible y contrastarla con el reloj efectivo en simulación
  y `report_clocks` antes de cerrar constraints o conectar IP Integrator.
- El lector usa dominios de 200/450 MHz. La unión debe incorporar CDC de
  payload, descriptores y eventos; su timing aislado no cubre el nuevo puente.
- El ejemplo AMD deja `c2h_dsc_byp_load_1=0` y `c2h_sts_1` sin conectar.
  Exportar dos canales no prueba tráfico ni finalización del segundo canal.
- La selección generada es `PCIE4C_X0Y0`, `select_quad=GTY_Quad_127`.
  El XDC de ejemplo busca PERST en el banco 65, usa LVCMOS18 y obtiene
  referencias mediante consultas a sitios GT. Son asignaciones del ejemplo,
  pendientes de resolver a bolas y revisar junto con memoria y auxiliares.
- Las excepciones de timing del ejemplo requieren revisión en su jerarquía
  integrada; no extenderlas a todos los cruces del diseño.

La elección entre nodo DDR de 512 GiB y nodo HBM separado, o placa combinada,
sigue abierta en [HBM_BOARD_FEASIBILITY.md](HBM_BOARD_FEASIBILITY.md).
Esta generación no cierra DIMM/MIG, pinout conjunto, potencia, refrigeración,
BOM, esquemático ni PCB. Tampoco transforma el BAR en memoria coherente.

## Respaldo y reproducción local

Directorio: `D:/AI/artifacts/ntrpu-hbm-pcie/20261009T000516217615Z`.
Se copiaron y verificaron 785 archivos exportados, 49.204.044 bytes. El
manifiesto incremental `hbm-pcie-20261009T000516217615Z-probe1` contiene
796 archivos, incluidos fuentes, revisión, procedencia HBM y verificador.
Se restauró completo en `restore-probe1` y pasó nuevamente la revisión y
ocho controles de evidencia alterada. Recibo: `BACKUP_probe1.json`.

```powershell
python -B tools/check_ntrpu_hbm_pcie_probe.py D:/AI/artifacts/ntrpu-hbm-pcie/20261009T000516217615Z/probe1
```

SHA-256 del manifiesto del almacén:
`82d8dc883474ddc100be35e1b66017a1515247e453a574563cb2ae8a42f2e62c`.
Almacén: `D:/AI/artifacts/ntrpu-content-store`. La restauración valida los
archivos y su revisión; no afirma haber repetido Vivado en otra instalación.

La revisión conserva las 103 referencias protegidas, incluida la cola
principal. No se lanzaron nuevas simulaciones ni se modificaron sus fuentes.
