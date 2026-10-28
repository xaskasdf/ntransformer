# Roadmap de la tarjeta NTRPU: del diseño electrónico a fabricación

Fecha: 2026-10-02. Estado: planificación de una tarjeta propia alrededor de un FPGA AMD, con objetivo de hasta 512 GB de DDR4. El componente final, la memoria y el PCB están pendientes de selección y validación. El `xczu7ev-ffvc1156-2-e` utilizado en las pruebas HLS es una referencia de desarrollo; no confirma compatibilidad con esa capacidad.

Este roadmap complementa los [hitos de la prueba de concepto](POC_ROADMAP.md) y la [arquitectura entre NTRPU](NTRPU_FABRIC.md).

Primer avance: [arquitectura preliminar de la tarjeta](board/ARCHITECTURE.md), con diagrama de bloques, direccionamiento, presupuestos reproducibles y preparación del esquemático; la [matriz FPGA/DDR4](board/MEMORY_FEASIBILITY.md) registra candidatos, evidencia y condiciones pendientes.

La [cola de trabajo Vivado para el agente principal](board/VIVADO_QUEUE.md) convierte la planificación en tareas ordenadas, con entregables, criterios de validación y respaldo. La ejecución comenzó el 2026-10-02; consultar el [estado y la evidencia por tanda](board/VIVADO_STATUS.md) para conocer avances y dependencias abiertas.

Actualización 2026-10-15: el [plan de cierre FPGA y diseño físico](board/PHYSICAL_DESIGN_PLAN.md) conecta la evidencia experimental HBM con las etapas pendientes de la tarjeta DDR4. Define criterios de avance, el primer paquete de componentes y presupuestos, y el desarrollo del modelo mecánico en paralelo al PCB. Conserva el formato de prototipo de banco y las dependencias de la cola Vivado.

## Qué se diseña con cada herramienta

| Trabajo | Herramientas | Resultado |
| --- | --- | --- |
| Circuito programable dentro del FPGA | Vitis HLS / Vivado | RTL, restricciones, informes y bitstream |
| Conexiones y placa electrónica | ECAD, como KiCad o Altium | Esquemático y PCB editables |
| Fabricación y montaje | Exportaciones del ECAD | Gerber, taladros, lista de materiales y posiciones |

Los Gerber describen las capas de fabricación. El proyecto editable conserva componentes, conexiones y reglas; ambos deben archivarse. Una simulación RTL correcta comprueba lógica digital, mientras que la placa requiere además validación eléctrica, térmica y física.

## 1. Arquitectura y selección de componentes

Definir FPGA y encapsulado exactos, canales de memoria, módulos DDR4 concretos, PCIe, dimensiones, alimentación y refrigeración. Estimar ancho de banda necesario para pesos, activaciones y KV cache, además de capacidad.

Comparar `4 × 128 GB` y `8 × 64 GB` únicamente como opciones pendientes de validar: cantidad de módulos y cantidad de canales son decisiones distintas. Registrar capacidad comercial, bytes direccionables y capacidad útil para el modelo.

Comprobar tipo de DIMM, ranks, densidad, ECC, configuración soportada por el controlador y recursos del FPGA. La IP DDR3/DDR4 documentada en PG150 no configura la memoria leyendo SPD; cambiar la configuración del DIMM requiere regenerar la IP. [AMD: configuraciones DIMM](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DIMM-Configurations).

**Entregable:** diagrama de bloques, componentes candidatos y presupuestos de memoria, ancho de banda, potencia y coste, con incertidumbres explícitas.

## 2. Interfaces y mapa de pines en Vivado

Configurar DDR4, PCIe, relojes y arranque para el dispositivo candidato. Comprobar que bancos de E/S, voltajes, transceptores y recursos permiten coexistir a todas las interfaces. Coordinar el pinout con la colocación física prevista y exportarlo al esquemático.

La planificación puede comenzar temprano; debe refinarse y validarse con el diseño sintetizado e implementado antes de cerrar la placa. [AMD: planificación de E/S](https://docs.amd.com/r/en-US/ug899-vivado-io-clock-planning/Netlist-I/O-Planning).

**Entregable:** configuración reproducible de IP, mapa de pines, restricciones XDC e informes de comprobación. Resolver incompatibilidades antes de fijar conexiones.

## 3. Esquemático eléctrico

Documentar FPGA, DDR4, PCIe, reguladores, secuencia de encendido, desacoplos, relojes, memoria de arranque, JTAG, sensores y puntos de prueba. Verificar símbolos y huellas contra las fichas de los componentes exactos.

**Entregable:** esquemático revisado, lista de materiales preliminar y comprobación de reglas eléctricas (ERC), con cualquier excepción justificada.

## 4. Diseño físico del PCB

Acordar con el fabricante el apilado de capas y materiales (*stackup*), impedancias, vías y límites de fabricación. Colocar componentes, resolver la salida de conexiones del encapsulado BGA y rutear alimentación y señales.

Aplicar las restricciones específicas de DDR4 y PCIe: retardos, pares diferenciales, caminos de retorno y acoplamiento. Dimensionar distribución de corriente, disipador y flujo de aire. El número de capas se decide a partir de estas necesidades. [AMD: guía de PCB UG583](https://docs.amd.com/r/en-US/ug583-ultrascale-pcb-design).

**Entregable:** PCB editable, stackup acordado, reglas de ruteo y comprobación geométrica (DRC).

## 5. Revisión antes de fabricar

Reconciliar esquemático, PCB y pinout de Vivado. Revisar integridad de señal y alimentación, tolerancias de fabricación, refrigeración y montaje. Registrar simulaciones y supuestos; un DRC limpio por sí solo no acredita el funcionamiento de DDR4 o PCIe.

**Criterio de cierre:** revisiones resueltas, interfaces compatibles y paquete de fabricación correspondiente a una misma revisión identificada.

## 6. Fabricación y montaje

Preparar Gerber, taladros, especificación del stackup e impedancias, planos, lista de materiales (BOM) y posiciones de componentes (*pick-and-place*). Revisar los archivos exportados con un visor independiente y confirmar los requisitos del fabricante y ensamblador. [KiCad: editor PCB y salidas de fabricación](https://docs.kicad.org/10.0/en/pcbnew/pcbnew.html).

**Entregable:** paquete versionado, fuentes editables y registro de componentes efectivamente montados.

## 7. Encendido y validación del prototipo

Verificar progresivamente alimentación y temperaturas, relojes y JTAG, configuración del FPGA, calibración y pruebas DDR4, enumeración PCIe y DMA. Después ejecutar cálculo con pesos reales, comparar resultados y medir ancho de banda sostenido, latencia, consumo y estabilidad.

**Entregable:** informe de placa con revisión, bitstream, condiciones de ensayo y resultados reproducibles. Probar dos NTRPU añade validación del transporte y del runtime que agrega su memoria.

## Trabajo que puede adelantarse

Mientras se valida la lógica, preparar el diagrama de bloques, la matriz FPGA/controlador/DIMM, los presupuestos y un pinout preliminar. Usar Brandon y Llama 3.2 1B Instruct como cargas de prueba para dimensionar tráfico y buffers. El primer paquete revisable será esa arquitectura y su viabilidad de interfaces; después se consolidarán esquemático y PCB.
