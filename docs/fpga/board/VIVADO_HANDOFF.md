# Paquete de preparación para Vivado y ECAD

Actualización posterior del hilo principal: [ejecución real y evidencias](VIVADO_STATUS.md). La revisión siguiente describe la preparación local anterior a esas tandas.

Revisión del 2026-10-02. Se contrastó el [plan de bancos](IO_PLAN.md) con los archivos oficiales de encapsulado. No se encontró Vivado en PATH ni en las rutas locales AMD/Xilinx revisadas; la generación de MIG y PCIe sigue pendiente. Este paquete se preparó localmente, sin acceder a la VM de validación.

Para encolar la ejecución en el hilo principal, seguir la [cola Vivado V0–V9](VIVADO_QUEUE.md): contiene dependencias, comandos iniciales, criterios de aceptación, integración NTRPU y respaldo por tanda, además de una orden lista para trasladar.

## Evidencia del encapsulado

Fuente: [portal AMD de pinouts UltraScale/UltraScale+](https://www.amd.com/en/developer/resources/adaptive-socs-and-fpgas/package-pinout-files/ultrascale-package-device-pinout-files.html), [ZIP oficial](https://download.amd.com/adaptive-socs-and-fpgas/developer/adaptive-socs-and-fpgas/package-pinout-files/usapackages/usaall.zip). La copia consultada tiene SHA-256 `b6398572a94213a03961501ce96cbabba1266a0f5c8525c618ef6b62ba490de8` y está en `build/ntrpu-board-study-20261002/usaall.zip`.

| Comprobación | VU5P / FLVA2104 | VU9P / FLGA2104 |
| --- | --- | --- |
| Archivo dentro del ZIP | `usaall/xcvu5pflva2104pkg.txt` | `usaall/xcvu9pflga2104pkg.txt` |
| Revisión del archivo | 1.1, Production | 1.1, Production |
| Bolas distintas | 2.104 | 2.104 |
| E/S HP de usuario | 832, en 16 bancos | 832, en 16 bancos |
| SLR de DDR0/1/2/3 según archivo | 0 / 1 / 0 / 1 | 0 / 1 / 1 / 2 |
| Quads 224–227 | Cuatro pares TX y cuatro RX por quad | Cuatro pares TX y cuatro RX por quad |

Los conjuntos de bancos coinciden con el manifiesto. Cada terna DDR permanece en un SLR del archivo. La columna sigue respaldada por el mapa de UG575; los números SLR aún deben contrastarse con la base de dispositivos de Vivado.

El [inspector local](../../../tools/inspect_ntrpu_package.py) reproduce el análisis con Python 3.11+ y biblioteca estándar:

```bash
python tools/inspect_ntrpu_package.py build/ntrpu-board-study-20261002/usaall.zip vu5p_flva2104_bench
python tools/inspect_ntrpu_package.py build/ntrpu-board-study-20261002/usaall.zip vu9p_flga2104_bench
```

Agregar `--include-pins --output RUTA_NUEVA.json` genera una hoja de conexiones con todas las bolas, función, banco, byte group y SLR. `rtl_port` y `pcb_net` comienzan en `null`: no se adjudican pines a señales de diseño por inferencia. La carpeta destino debe existir; se rechaza sobrescribir archivos.

Se generaron ambas hojas en `build/ntrpu-board-study-20261002/<perfil>_package.json`. Son resultados regenerables, fuera del código versionado. Los informes incluyen hashes del ZIP y archivo interno, candidatos GC/MGTREFCLK y señales de configuración. La presencia de un pin de reloj no prueba una ruta válida hacia una IP concreta.

## Punto pendiente: PERST y DDR2

Los dos pinouts sitúan `PERSTN0` en **AM17** y `PERSTN1` en **AP17**, ambos del banco 65 reservado para DDR2. Es una colisión potencial a nivel de bola adicional a la restricción Tandem ya documentada.

En la primera sesión MIG/PCIe, resolver explícitamente si la configuración elegida requiere AM17, si MIG lo deja disponible y cómo se adapta la tensión del reset host al banco. Si se estudia otro pin, confirmar que la IP admite esa ruta y que cierra timing. No conectar el reset a una bola alternativa solo porque esté libre. La convivencia de DDR2 y PCIe continúa abierta hasta esa comprobación.

## Inventario reproducible en Vivado

El [script Tcl](../../../tools/ntrpu_board_inventory.tcl) se ejecutará en un proceso batch nuevo. Rechaza un proyecto ya abierto y una carpeta de salida existente. Primero consulta partes instaladas, sin elegir automáticamente velocidad ni temperatura:

```bash
vivado -mode batch -nojournal -nolog -source tools/ntrpu_board_inventory.tcl \
  -tclargs discover build/board-parts-01
```

Elegir una parte completa de `available_parts.txt`, luego ejecutar:

```bash
vivado -mode batch -nojournal -nolog -source tools/ntrpu_board_inventory.tcl \
  -tclargs PARTE_EXACTA build/board-inventory-01
```

El script crea un proyecto **I/O Planning** aislado y exporta propiedades de parte, bolas, bancos y SLR, además del catálogo DDR4/PCIe/XDMA disponible. `INVENTORY_COMPLETE.txt` solo acredita que terminó esa exportación. El script no genera IP, sintetiza ni implementa. Flujo basado en [create_project](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/create_project), [get_package_pins](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/get_package_pins) y [report_property](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_property) de UG835 2026.1.

## Trabajo siguiente dentro de Vivado

| Paso | Entrega concreta |
| --- | --- |
| Cerrar parte y memoria | Parte FPGA completa; ficha, SPD y modelo del DIMM exacto |
| Generar un canal | Configuración guardada para RDIMM x4, ECC, población y capacidad correctas; revisar 37 bits para 128 GiB |
| Planificar MIG | Byte lanes, dirección/control, reloj, reset, referencias y bolas; resolver AM17/AP17 si se usa banco 65 |
| Añadir cuatro canales y PCIe | Proyecto conjunto con los bloques del plan; XDC y reportes de recursos/relojes/DRC |
| Implementar | Timing y utilización de la configuración conjunta, antes de cerrar el esquema |
| Entregar a ECAD | Una fila por bola con puerto RTL cuando aplique, net del PCB, dominio eléctrico, destino y evidencia de revisión |

Las filas de alimentación, tierra, configuración, reservadas y NC también se revisarán en ECAD, aunque no tengan puerto RTL. Conservar los nombres del fabricante y comprobar símbolo/footprint contra el encapsulado; no convertir automáticamente toda bola sin puerto RTL en «sin conectar».

## Validación de las herramientas locales

Los dos archivos reales pasan el inspector. Se rechazaron seis alteraciones: dispositivo incorrecto, bola ausente, bola duplicada, metadatos incompletos, SLR inconsistente y lane GT ausente. También se verificó que un archivo de salida existente conserva su contenido.

El script batch se ejecutó con Tcl 9.0.4 y comandos Vivado simulados en siete escenarios: descubrimiento, exportación, carpeta existente, proyecto abierto, partes ausentes, parte incorrecta y fallo al abrir el diseño. Estas pruebas comprueban flujo y manejo de archivos; no sustituyen la ejecución en el Tcl de Vivado. Se comprobaron siete enlaces locales del paquete. No hay resultados de MIG, DRC o timing en esta revisión.
