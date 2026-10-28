# Viabilidad de placa HBM, DDR4 y PCIe

Estudio del 2026-10-07 para la variante HBM, complementario al [cierre de placa](BOARD_CLOSURE_REVIEW.md). No cambia la selección DDR del principal ni aprueba una BOM. Sigue abierta la decisión entre nodos DDR/HBM separados y una tarjeta combinada.

## Inventario comprobado en Vivado

Un proyecto independiente de planificación de E/S consultó `xcvu47p-fsvh2892-2-e` en Vivado 2026.1. Se exportaron las 2.892 bolas del encapsulado, bancos, sitios y regiones. No se asignaron puertos ni se generó una IP DDR.

| Recurso | Inventario |
| --- | --- |
| E/S de propósito general HP | 624: doce bancos de 52 |
| SLR0, columna de E/S X4 | Bancos 64–67, regiones X4Y0–X4Y3 |
| SLR1, misma columna | Bancos 68–71, regiones X4Y4–X4Y7 |
| SLR2, misma columna | Bancos 72–75, regiones X4Y8–X4Y11 |
| Transceptores | 96 sitios GTYE4_CHANNEL, 24 GTYE4_COMMON |
| PCIe | Dos sitios PCIE40E4 y cuatro PCIE4CE4 |
| Referencia HBM | Dos sitios HBM_REF_CLK |

Son recursos del dispositivo, no enlaces simultáneos ya construidos. La asignación de quads, referencias de reloj, resets, IP y rutas debe validarse conjuntamente.

## El plan DDR de 512 GiB no se traslada directamente

El [pinout DDR genérico existente](board/PIN_PLANNING.md) usa cuatro bancos para 154 puertos por canal, incluidos 72 DQ y 36 DQS. La propuesta anterior de tres bancos no está calificada. Bajo nuestro criterio de **bancos exclusivos, contiguos, en una columna y un SLR**, el VU47P admite como máximo tres grupos tanto de tres como de cuatro bancos: uno por SLR. La herramienta enumera todas las combinaciones, no solo un reparto elegido.

Por tanto, cuatro canales de 128 GiB no caben **bajo ese criterio de planificación**. Tres canales sumarían 384 GiB, pero esa cifra es aritmética: tampoco califica tres MIG, los DIMM exactos ni su coexistencia con HBM. No constituye una imposibilidad universal de alcanzar 512 GiB mediante otra topología.

AMD exige una columna común para la interfaz DDR4, concentra dirección/control en un banco y restringe el reparto de nibbles; el reloj GCIO debe corresponder a la columna y SLR de la interfaz. La regla de no cruzar SLR se mantiene como criterio conservador del proyecto y debe confirmarse en la colocación de la IP exacta. [PG150, reglas DDR4](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-Pin-Rules), [PG150, GCIO](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/GCIO).

## Opciones que permanecen abiertas

| Opción | Ventaja | Condición pendiente |
| --- | --- | --- |
| Nodo DDR de 512 GiB + nodo HBM | Conserva el trabajo de ambos candidatos; responsabilidades separadas | Probar tráfico entre nodos y presupuesto real del enlace |
| VU47P con HBM y hasta tres canales del perfil actual | Pesos y trabajo local en un dispositivo | Selección de DIMM, MIG conjunto, capacidad útil y E/S auxiliares |
| HBM + 512 GiB en la misma placa | Conserva la capacidad objetivo por tarjeta | Otra densidad/topología o FPGA; nueva calificación de memoria y pines |

Usar DIMM más densos no es una sustitución automática: cambian ranks, DRAM, RCD, geometría y modelos. La presencia de un nombre de módulo en el catálogo de Vivado tampoco valida el conjunto de canales. No se han seleccionado ni comprado componentes nuevos.

Además, PG150 limita `C_S_AXI_ADDR_WIDTH` a 37 bits: 128 GiB direccionables por instancia AXI. Dos canales de 256 GiB no son una solución directa con esa interfaz; exigirían otro controlador/interfaz y validación nueva. [PG150, parámetros AXI](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/AXI4-Slave-Interface-Parameters).

## PCIe y comunicación entre nodos

Mantener direcciones lógicas de 64 bits con nodo, región, generación y offset; traducir a la memoria local solo después de autorizar el acceso. Un BAR expone una ventana y DMA mueve datos: no crea coherencia ni une físicamente HBM de dos FPGAs. La tarjeta de apoyo puede gestionar enumeración, colas y rutas, pero no elimina límites del switch o sus uplinks.

Gen3 x16 y Gen4 x8 ofrecen el mismo techo tras codificación, aproximadamente 15,754 GB/s por dirección antes de TLP/control. Gen4 requiere el bloque adecuado: PG213 lo admite en PCIE4C x8 para grados -2/-2L/-3; no debe inferirse de cualquier sitio cuyo nombre contenga PCIe. Aún falta generar y colocar la variante concreta de endpoint/bridge sobre este encapsulado. [PG213, requisitos](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Minimum-Device-Requirements).

Para varios enlaces, medir concurrencia extremo a extremo y comprobar quads independientes, arbitraje HBM, orden y contrapresión. Un único uplink compartido sigue limitando el total. El [retorno actual](HBM_READ_RETURN.md) tiene además una limitación interna por alternar recepción/entrega; llenar el FPGA de buffers sin cambiar esa organización no la resuelve.

## Próximas entradas del diseño eléctrico

1. Resolver la topología DDR/HBM y fijar capacidad útil por nodo.
2. Seleccionar FPGA/encapsulado/grado y DIMM exactos con modelos aplicables.
3. Generar memoria, HBM y PCIe conjuntamente; exportar `puerto → banco → bola → net ECAD`, incluidos relojes, PERST, configuración y supervisión.
4. Preparar potencia por rail y actividad, secuencia, refrigeración y mecánica; después seleccionar reguladores y conectores.
5. Crear esquemático/BOM verificables y acordar stackup/escape BGA con fabricación antes de cerrar PCB, SI/PI y DRC.

La simulación funcional o el timing de un bloque aislado no sustituyen estos entregables ni las mediciones de hardware.

## Jerarquía eléctrica preparada para ambas opciones

La jerarquía propuesta separa hojas por función. Es un contrato de diseño; aún no asigna bolas ni constituye un esquemático validado.

| Hoja | Contenido y condición de cierre |
| --- | --- |
| `00_system` | Instancias, variantes DDR/HBM y mapa de conectores; capacidad útil y enlaces por nodo aprobados |
| `10_power` | Rails del FPGA, GT, memoria y auxiliares; estimación con actividad, secuencia y supervisión verificadas contra las piezas exactas |
| `20_fpga_config` | Configuración, almacenamiento de arranque, JTAG y straps; tensiones y modos compatibles |
| `30_clocks_reset` | Referencias PCIe, DDR y HBM, distribución y reset por dominio; frecuencias, jitter y secuencia documentados |
| `40_pcie_links` | Pares TX/RX, referencias, PERST y señales auxiliares; quads y endpoint colocados conjuntamente |
| `50_memory` | Variante DDR con módulos/RCD o conexiones externas exigidas por el encapsulado HBM; pinout generado por las IP calificadas |
| `60_supervision` | Telemetría, temperatura, fallos y control de ventiladores; reacción segura y procedimiento de encendido |
| `70_mechanical` | Dimensiones, conectores, fijación y disipación; revisión del prototipo y fabricante |

El intercambio con ECAD debe conservar al menos `variante, puerto HDL, dirección, dominio de reloj, estándar de E/S, banco, bola, net, componente/pin y fuente de la asignación`. No rellenar campos desconocidos con pines supuestos. Las reservas de GT y bancos también forman parte del registro, aunque todavía no conduzcan señales.

La BOM inicial debe distinguir pieza elegida de candidata y registrar fabricante, referencia completa, encapsulado, revisión documental, modelo disponible y motivo pendiente. FPGA, DIMM, reguladores, generadores de reloj, conectores y refrigeración permanecen abiertos hasta resolver sus respectivos presupuestos. Exportar una BOM sin esas decisiones no hace fabricable la placa.

## Evidencia reproducible

Inventario y revisión: `D:/AI/artifacts/ntrpu-read-return/20261007T103522780001Z/20261007T110553948819Z-package/`. Respaldo incremental verificado en el almacén local por contenido.

```bash
# En directorio de salida nuevo y fuera del proyecto principal:
vivado -mode batch -source tools/ntrpu_hbm_package_inventory.tcl -tclargs SALIDA
# Revisión de la tanda conservada, incluidos hashes y presupuesto de bancos:
python3 -B tools/check_ntrpu_hbm_package_inventory.py TANDA_PACKAGE/output
```

El script Tcl requiere que `SALIDA` exista; el ejecutor de la tanda la creó y registró resultado/hashes. El checker se aplica al paquete completo conservado, no a TSV sueltos.
