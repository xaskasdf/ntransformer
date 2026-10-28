# Viabilidad preliminar de FPGA y DDR4

Investigación documental del 2026-10-02. Ninguna combinación de esta matriz ha sido homologada en Vivado ni en una placa NTRPU. La selección se cerrará con número de parte completo, versión de IP, pinout, simulación y disponibilidad verificadas.

## Tres topologías diferentes para 512 GiB

Se normaliza la capacidad a GiB para los cálculos; comprobar los bytes reales de cada módulo. Un canal se presupuesta con 64 bits de datos y 8 adicionales de ECC.

| Opción | Canales | DIMM por canal | Capacidad por DIMM | Capacidad total | Evaluación inicial |
| --- | ---: | ---: | ---: | ---: | --- |
| A | 4 | 1 | 128 GiB | 512 GiB | Menos conectores; estudiar primero RDIMM 3DS y su soporte exacto |
| B | 4 | 2 | 64 GiB | 512 GiB | Base para calificar RDIMM 2Rx4; más carga eléctrica y espacio |
| C | 8 | 1 | 64 GiB | 512 GiB | Más ancho de banda potencial; excede las E/S de los VU5P/VU9P candidatos para x4 con ECC |

A y B tienen el mismo ancho de banda teórico si operan a la misma tasa. Dos DIMM por canal comparten el bus; la tasa alcanzable y la eficiencia pueden cambiar.

## Evidencia de controladores y módulos

La sección DDR3/DDR4 de PG150 enumera RDIMM hasta 128 GB, LRDIMM hasta 64 GB, soporte 3DS y densidades adicionales mediante partes personalizadas. Esto orienta la búsqueda, sin garantizar un módulo particular. Usar [la sección correcta de DDR4](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-SDRAM?contentId=2Wsnya0Z1WaS5BRBZ~M1OA): el documento también contiene otra sección con el mismo título para Spartan UltraScale+.

| Candidato de memoria | Evidencia primaria | Qué falta |
| --- | --- | --- |
| Micron `MTA36ASF8G72PZ-3G2F1` | Micron identifica 64 GB, RDIMM, 2Rx4, DDR4-3200 y chips de 16 Gb en su [lista para AIC](https://my.micron.com/partners/odm-partners/aic) | Ficha y SPD de la revisión exacta, disponibilidad, parámetros MIG y prueba con dos módulos por canal. Esa lista valida servidores AIC, no NTRPU |
| Samsung `M393AAG40M32-CYFG0` | Su [SPD publicado](https://download.semiconductor.samsung.com/resources/data-sheet/M393AAG40M32-CYFG0.pdf) declara RDIMM, chips de 16 Gb, 3DS 2H, organización 2Rx4 y bus de 64 bits más ECC | Candidato para estudiar 128 GiB; resolver la cabecera de organización inconsistente con esos campos y obtener ficha completa antes de fijar capacidad/timings |
| LRDIMM de 128 GB | Queda fuera de la densidad LRDIMM enumerada como estándar en la sección consultada | Requiere evidencia específica del módulo, configuración personalizada y modelo de simulación; no sustituirlo automáticamente por RDIMM |

AMD documenta [dos RDIMM dual-rank por canal](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-RDIMM). La opción B encaja con esa topología documental, pendiente de generación de IP para el componente concreto. Para una parte personalizada, AMD exige incorporar manualmente el modelo de memoria y posiblemente adaptar el banco de simulación. [Configuración de partes](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/Basic-Tab?contentId=MGveBVBHGBtHBQNV7JSeVA).

## Candidatos FPGA

| Dispositivo y encapsulado a estudiar | E/S HP | DSP del dispositivo | Papel |
| --- | ---: | ---: | --- |
| `XCVU5P`, `FLVA2104` | 832 | 3.474 | Primer candidato para evaluar cuatro canales y recursos de cálculo |
| `XCVU9P`, `FLGA2104` | 832 | 6.840 | Alternativa si el cálculo necesita más recursos |
| `XCVU13P`, `FHGA2104` | 832 | 12.288 | Aumenta cálculo, conservando el mismo límite de E/S en este encapsulado |

Datos de [AMD XMP103, tablas de recursos y encapsulados](https://docs.amd.com/api/khub/documents/Fz8MLna3N2KIo_xWN_Iv3w/content) y [UG575, mapas por encapsulado](https://docs.amd.com/api/khub/documents/GnuVZDIoTcrlIuu8Aok5XQ/content). Son candidatos incompletos: quedan por elegir velocidad y temperatura, consultar precios y verificar las posiciones PCIe/DDR. Los máximos de un dispositivo no corresponden a todos sus encapsulados.

### Filtro de pines para ocho canales

En una interfaz x72 construida con DRAM x4 hay 72 DQ y 18 pares DQS: **108 pines de señal por canal antes de dirección, comandos y relojes**. Ocho canales requieren al menos `8 × 108 = 864`, más que 832 E/S HP. Esta cota descarta C en esos encapsulados bajo esas condiciones. Deshabilitar ECC no es una solución validada para el resto del presupuesto.

Para cuatro canales se reservan inicialmente tres bancos HP completos por canal, sujetos al planificador. AMD permite compartir bancos bajo restricciones, exige que cada interfaz permanezca en una columna y un SLR, y restringe la colocación de DQ/DQS. Una suma suficiente de pines no prueba que el pinout cierre. [PG150: reglas DDR4](https://docs.amd.com/r/en-US/pg150-ultrascale-memory-ip/DDR4-Pin-Rules?contentId=vIJySmS_aQOK66n82tWyUg).

## Criterios para cerrar la selección

1. Obtener ficha, SPD, revisión y modelo de simulación del DIMM exacto, incluyendo ranks físicos y lógicos si es 3DS.
2. Generar un controlador con capacidad correcta, ECC y tasa inicial conservadora; guardar configuración e informes.
3. Validar cuatro controladores junto con PCIe, relojes y arranque en el encapsulado elegido.
4. Simular calibración y tráfico, incluyendo direcciones altas, cambios de rank y errores ECC; completar implementación y restricciones.
5. Revisar señal, potencia, mecánica y suministro de componentes antes del esquemático definitivo.

**Orden propuesto:** calificar primero B como referencia documental de memoria; investigar A para reducir conectores; mantener C como arquitectura que exige otro reparto de hardware. El [diseño preliminar](ARCHITECTURE.md) conserva estas alternativas.
