# Cierre pendiente del diseño NTRPU

Revisión local del 2026-10-07. Complementa la [arquitectura de placa](board/ARCHITECTURE.md), el [transporte entre nodos](NTRPU_TRANSPORT.md) y el [estado del hilo principal](board/VIVADO_STATUS.md). No cambia su cola ni considera aprobada una etapa por el resultado de otra.

## Arquitecturas que deben distinguirse

| Línea | Base actual | Qué sigue abierto |
| --- | --- | --- |
| Capacidad DDR4 | VU5P FLVA2104; VU9P como alternativa; objetivo de cuatro canales de 128 GiB | DIMM exacto, modelos, pinout conjunto y calificación física |
| Memoria HBM | Ensayos sobre VU47P FSVH2892, perfil de 16 GB, puertos AXI 00 y 16 | Integración con el transporte completo y cierre de placa |
| HBM + 512 GiB DDR4 en una placa | [Inventario VU47P y viabilidad](HBM_BOARD_FEASIBILITY.md) | El perfil evaluado admite tres grupos DDR como máximo; cuatro canales de 128 GiB requieren otra solución. Coexistencia completa no demostrada |

La consulta al usuario sobre nodos separados o una placa combinada permanece pendiente. Las pruebas del transporte continúan porque sirven a ambas alternativas; no se congelan componentes ni un PCB a partir de esa incertidumbre. El prototipo de banco sigue siendo la referencia mecánica documentada.

HBM está integrada en el encapsulado y es local al FPGA. AMD especifica 16 GB y 32 interfaces HBM para VU47P; nuestros dos puertos no acreditan el rendimiento del conjunto. [Tabla AMD](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/virtex-ultrascale-plus-hbm.html).

## Trabajo digital y evidencias necesarias

| Etapa | Evidencia disponible | Criterio que falta |
| --- | --- | --- |
| Buffers y reservas | [Localidad y receptor doble](PEER_RX_RESERVE_LOCALITY.md), compuestos con CDC y adaptadores HBM | Adaptador del protocolo externo y evaluación de concurrencia del retorno |
| CDC/HBM de escritura | [Once escenarios por política, 58 comandos con modelo HBM nativo, comparación RTL/netlist y timing interno](PEER_RX_DUAL_HBM.md), a 200/450 MHz | Integración de interfaces externas y calificación térmica; margen de memoria mínimo de 14 ps |
| Retorno HBM de lectura | [Dos canales sintetizables, reserva completa antes de DATA, errores y contrapresión](HBM_READ_RETURN.md) | Revisión 3 calificada en simulación y timing interno (+4 ps en memoria); integración peer, E/S, térmica y escalado de concurrencia pendientes |
| PCIe | El principal ya tiene DMA, trazas nativas y [monitor de inventario](board/PCIE_REQUEST_INVENTORY_RTL.md) | Unirlo al receptor nuevo, con productor cerrado, regiones, épocas y entrega final comprobada |
| DDR4 | Simulación genérica de capacidad y ECC | [Calificación del DIMM exacto](board/MEMORY_QUALIFICATION.md); SPD parcial B1/B2 no sustituye F1 |
| Cómputo | Bancos numéricos y FFN documentados en el proyecto | Inferencia integrada con memoria/transporte y pesos reales, comparada con referencia |
| Sistema | Contrato lógico de varias NTRPU | Dos endpoints reales o modelo conjunto aprobado: carga, visibilidad, pérdida de enlace y recuperación |

El retorno nuevo es RTL independiente de las lecturas de comprobación del banco de escritura. Ambos bloques experimentales reciben comandos ya autorizados; todavía necesitan el adaptador del protocolo y las regiones registradas. El error retenido y el vaciado se comprueban por separado: estar vacío no equivale a éxito.

El [plan de integración digital](DUAL_HBM_INTEGRATION_PLAN.md) distingue retorno R0–R2, rendimiento R3, peer/PCIe P0–P1 y cómputo C0. El lector conserva evidencias por revisión; P0/P1 y C0 siguen pendientes como composición del sistema. El bloque de escritura usa 11.968 LUT, 16.613 registros y 15 BRAM36; sus 13 advertencias DRC están clasificadas, sin waivers. Estos recursos y su timing no se atribuyen al lector ni se suman como si ambos ya estuvieran integrados. Las E/S y el timing del sistema completo siguen sin calificar.

## Presupuesto de comunicación

Dos entradas de 512 bits a 200 MHz tienen un techo de **25,6 GB/s**; dos puertos HBM de 256 bits a 450 MHz, **28,8 GB/s**, antes de pausas. Son cálculos de estas interfaces, no medidas de placa. Un enlace PCIe Gen3 x16 o Gen4 x8 tiene aproximadamente **15,754 GB/s por dirección después de codificación y antes de TLP/control**, según el [presupuesto del transporte](NTRPU_TRANSPORT.md).

Por eso la ampliación debe medirse por rutas completas: banco local → cómputo, enlace → banco remoto y varios enlaces simultáneos. No sumar el ancho HBM a PCIe ni contar dos veces un uplink compartido. Gen4 x8 requiere el bloque y grado adecuados; el nombre PCIE4 por sí solo no garantiza Gen4. [AMD PG213](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Minimum-Device-Requirements).

## Entradas necesarias para el diseño eléctrico

La búsqueda del repositorio no encontró proyectos KiCad/Altium ni una BOM el 2026-10-07. Los siguientes entregables siguen pendientes:

1. Fijar FPGA, encapsulado y grado, módulos de memoria y revisiones de DRAM/RCD. Obtener modelos y documentación aplicables.
2. Exportar un pinout conjunto de memoria, PCIe, relojes, arranque y depuración, sin conflictos de banco, voltaje o transceptor.
3. Definir entrada de alimentación, presupuesto por rail, secuencia, supervisión, refrigeración y dimensiones del prototipo.
4. Crear esquemático y BOM editables; verificar símbolos y huellas contra las partes seleccionadas y cerrar ERC.
5. Acordar stackup y capacidades del fabricante; resolver escape BGA, impedancias, retardos, retornos de corriente e integridad de alimentación/señal.
6. Cerrar PCB/DRC y revisión de fabricación; finalmente medir alimentación, calibración de memoria, PCIe, tráfico e inferencia en hardware.

Se puede preparar la jerarquía de hojas y los contratos de señales antes de seleccionar las piezas. Los valores de reguladores, desacoplos, terminaciones y geometrías de ruteo necesitan esas entradas: no se inventan para producir un esquema aparentemente terminado. El [roadmap de placa](BOARD_DESIGN_ROADMAP.md) conserva los entregables de fabricación y encendido.

La [jerarquía eléctrica propuesta](HBM_BOARD_FEASIBILITY.md#jerarquía-eléctrica-preparada-para-ambas-opciones) organiza ocho hojas y define los campos del intercambio Vivado/ECAD. Es preparación de diseño; no representa un esquemático, BOM o PCB terminado.
