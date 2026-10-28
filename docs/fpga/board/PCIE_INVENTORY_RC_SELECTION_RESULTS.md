# Monitor PCIe: selección RC separada de la aritmética

El 2026-10-08 R4 pasó el banco integrado RTL, el selector aislado y el replay
de su propia netlist. WNS OOC: **-10.047 ns**, frente
a **-9.877 ns** de la referencia; diferencia
de slack **-0.170 ns**. Se conserva el productor original como preferido entre las medidas OOC. Setup y hold siguen abiertos.

## Cambio probado

`ntrpu_pcie_inventory_scheduler_r4.sv` parte de la referencia original.
`ntrpu_pcie_rc_update_select.sv` obtiene desde entradas crudas y contexto
propio qué escritura RC precede a cada encabezado y cuál queda para cada tag.
Conserva prioridad por lane, punteros repetidos, liberación terminal en SOP,
contexto retenido y RQ posterior a RC. La aritmética, el contador global y
la latencia pública permanecen como en la referencia. No incorpora R3 ni
la suma por prefijos. [Contrato previo](PCIE_INVENTORY_RC_SELECTION_CONTRACT.md).

## Cobertura funcional

El módulo selector real pasó **89.632 casos binarios** contra un modelo
independiente de lista de eventos: 65.536 combinaciones EOP, 4.096 combinaciones
SOP/contexto y 20.000 muestras. Los vectores se regeneraron y sus hashes
coinciden. Dos controles adicionales rechazaron evidencia incompleta.

El monitor completo pasó **22.339 filas/68 escenarios de cuatro estados**,
con **5.718.784 comparaciones de palabras de tabla**, incluidas filas
posteriores a fallos. Retiró 22.136 registros, canceló 67 por cold reset y
comprobó 71 revocaciones inmediatas de vacío, 3.609 filas vacías y una vuelta
de secuencia. Detectó siete mutantes reales y 17 corrupciones de evidencia.

RTL y netlist aprobaron **22.330 filas/67 escenarios binarios**, con 12
controles cada uno. Todas las filas públicas de estado, fallos, secuencia,
cancelación y vacío coinciden. El RTL comparó 5.716.480 palabras internas;
la netlist no usa esa jerarquía ni compila el productor, wrapper, selector o
validador RTL como DUT. Sus definiciones `glbl` coinciden. Startup 200 ns,
muestreo 1 ns, sin SDF. No se afirma equivalencia universal de cuatro estados.

## Medición OOC

Parte `xcvu5p-flva2104-2-i`, Vivado 2026.1, `RuntimeOptimized`, reloj de
4 ns, I/O cero y ninguna excepción temporal; mismo monitor completo:

| Métrica | Referencia | Selector R4 |
| --- | ---: | ---: |
| LUT | 48189 | 42176 |
| FF | 10729 | 10729 |
| CARRY8 | 260 | 260 |
| WNS, ns | -9.877 | -10.047 |
| TNS, ns | -109536.984 | -127754.758 |
| Endpoints setup incumplidos | 15851 | 15865 |
| WHS, ns | -0.075 | -0.075 |
| THS, ns | -139.02 | -139.748 |
| Endpoints hold incumplidos | 1936 | 1952 |

Peor ruta: `producer/address_next_reg[51][0]/C` → `producer/remaining_bytes_reg[20]/D`, **14.030 ns y 58 niveles**.
El DCP contiene 63290 primitivas y 70270 nets auditadas
sin ciclos, con ocho controles. Las 12 categorías de `check_timing` quedaron
en cero; el DRC enumeró cero comprobaciones encontradas. Persisten las
advertencias metodológicas registradas en la revisión de síntesis. No hay
colocación/ruteo ni frecuencia física medida. La ausencia de errores de
exportación no cierra timing, CDC o integración nativa.

R4 reduce **6013 LUT**
con el mismo número de registros. El aviso `Netlist 29-101` señala una
jerarquía con demasiadas primitivas para facilitar floorplanning; también
persisten `TIMING-16` y `XDCH-2`. La reducción de LUT no prueba mejora física.

## Evidencia y respaldo

| Tanda | Segundos de herramienta | SHA-256 del archivo |
| --- | ---: | --- |
| `20261008T122158Z-V4-ffn-supervised-pcie-inventory-join-rc-select-rtl` | 158.256 | `0464493e8a3ff97c68f23553efe4b8fb133fd78a2c244abc50affc46f8dd4d7c` |
| `20261008T122832Z-V4-ffn-supervised-pcie-inventory-join-rc-select-synth` | 300.900 | `4353cfbc18f5f2bb56474ec37c729fc6800fe660b87e72c25ea13f951a101d65` |
| `20261008T123439Z-V4-ffn-supervised-pcie-inventory-join-rc-select-binary-rtl` | 44.905 | `d3ec23d581faabe40f64b443540f17818eada234b44fd3f9fd3a0dad099c5e42` |
| `20261008T123756Z-V4-ffn-supervised-pcie-inventory-join-rc-select-binary-netlist` | 103.505 | `a442f6afdfa4f4e887590d53cdeae838c39baed2db145d24a99d87cbfb1ffcd2` |

Las cuatro tandas comparten `source-20261008T122111Z.zip`,
2152 archivos, SHA-256 `7b86c8cc11250688d5b8d01db74783f56edb99b6ffc026c396a1c95a1675034f`.
Los archivos y sus miembros están verificados en
`artifacts/ntrpu-board-vivado/<tanda>/`, con respaldo físico en D: y copia
en NFS. Netlist reproducida:
`028970ac9ca2ceec5113eb423a86a11afe7057a419ca115ecc7805124b6ff026`.

Revisiones independientes en `build/cloud-setup/20261002-board/`:

- `20261008T122158Z-V4-ffn-supervised-pcie-inventory-join-rc-select-rtl-independent-inventory-join-review.json`: `f452338ad15dc6da93a9cca415136a6f5b64a80f69706428706f42ffda6f1f4a`.
- `20261008T122832Z-V4-ffn-supervised-pcie-inventory-join-rc-select-synth-independent-synthesis-review.json`: `5fa286fbac4684e107613a34263229aa6061c65842d62c02ca09d5a1acda539d`.
- `20261008T123439Z-V4-ffn-supervised-pcie-inventory-join-rc-select-binary-rtl-independent-inventory-join-binary-rtl-review.json`: `0a722f5bdf6b7a87dd57b62f0c52001d88fd4937a85c52d4d15b238bd928f938`.
- `20261008T123756Z-V4-ffn-supervised-pcie-inventory-join-rc-select-binary-netlist-independent-inventory-join-binary-netlist-review.json`: `702b536827cc479a1b8037eca1e71c0b93d2536ea3c9880a299f6c9ee3048776`.

## Siguiente experimento y límites

El [plan de prefijos RC](PCIE_INVENTORY_RC_PREFIX_PLAN.md) usa solamente
escrituras efectivas. Su estudio de modelo pasó 67.108.864 casos de un
fragmento, 2.097.152 pares límite y 1.000.000 de prefijos muestreados; no
califica aún una implementación RTL. El informe del modelo tiene SHA-256
`69ae064e4b903713d0575528f8baf9ed54090fbbb599690e4cc7ef2a7308ba48`.

Mantener el perfil completo y todos los estados posteriores a fallos.
Siguen pendientes identidad/BDF, CDC/reset nativo de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, pesos reales, ECAD e implementación física. La captura
nativa reutilizada sólo cubre tres lecturas de 32/64/32 bytes; el banco amplio
es sintético. Se conservan las obligaciones de [la cola](VIVADO_QUEUE.md).
