# Diagnóstico de recuperación del proyecto FPGA

## Resultado

Se rescató un fragmento continuo de **1.149.629 bytes** con un informe Vitis HLS y detalles internos de síntesis del núcleo `unified_core_int8`. Es evidencia concreta de un prototipo FPGA. No se recuperaron todavía sus fuentes compilables, el proyecto Vivado completo ni un diseño comprobado de 512 GB DDR4.

## Archivo examinado

- Original: `D:\AI\recovered_projects.7z`, 90.110.075 bytes.
- Extracción: 5.497 archivos, 729.573.884 bytes, sin errores reportados por 7-Zip.
- Deduplicación SHA-256: 2.992 contenidos distintos.
- SHA-256 del archivo: `70c05a41ebd76a18c7c85f9ef2a357d9027e18b2bf84991d4e303c18b3947fb0`.

Se conservaron el original y los archivos extraídos. El directorio de trabajo es `build/recovery-20260930/`, excluido de Git por la regla existente `build/`. Allí están el listado de 7-Zip, `audit-index.json`, scripts de inspección y la copia del repositorio de transporte. Este directorio es temporal y no sustituye una copia de seguridad; la evidencia pertinente está preservada en `docs/fpga/recovered/`.

La búsqueda revisó nombres y contenido, incluso binarios y texto con bytes nulos, con términos Vivado, Alveo, U50/U250, FPGA, DDR4, RDIMM, capacidades de 512 GB, firmas HDL y comandos Tcl. Los nombres originales estaban mayormente perdidos. La búsqueda por firmas no demuestra que no existan otros fragmentos ocultos, comprimidos o truncados.

## Hallazgo principal

El miembro `recovered_projects/claude-sessions/f49844040.elf` contiene datos heterogéneos. El bloque pertinente empieza en el offset **9.199.618** y termina antes del offset **10.349.247**; los offsets son de bytes y comienzan en cero.

Se extrajo desde la cabecera del informe hasta el primer byte fuera de ASCII imprimible, tabulación o salto de línea. Se preservaron los bytes exactamente, sin corregir ni reconstruir contenido. Los hashes y límites están en [provenance.json](recovered/provenance.json). El nombre de salida es descriptivo y nuevo; el nombre original del `.rpt` no se conoce.

| Dato | Evidencia del informe |
| --- | --- |
| Proyecto | `uvp_hls_project` |
| Herramienta | Vitis HLS 2025.2, build 6295257 |
| Fecha interna | 11 de marzo de 2026, 18:05:18; zona horaria no indicada |
| Solución | `solution1`, Vivado IP Flow Target |
| Dispositivo | `xczu7ev-ffvc1156-2-e`, familia `zynquplus` |
| Módulo | `unified_core_int8_Pipeline_VITIS_LOOP_72_9_VITIS_LOOP_73_10` |
| Fuente referenciada | `../compute/unified_core_int8.cpp`, entre otras referencias a líneas 72 y 73 |
| Clock objetivo | 3,33 ns, aproximadamente 300 MHz |
| Estimación de clock HLS | 2,373 ns, incertidumbre 0,90 ns |
| Latencia del submódulo | 1.034 ciclos, 3,443 microsegundos en el informe |
| Bucle | Trip count 1.024, intervalo de iniciación 1, profundidad 10 |
| Recursos de la tabla principal | 17 DSP, 13.332 FF, 4.080 LUT, 0 BRAM y 0 URAM propios del submódulo |
| Interfaces visibles | Bancos `l1_weights_*` y `l1_in_*` de 8 bits; señales/resultados `l1_out_*` de 32 bits |

Es un **informe de un submódulo/bucle**, con interfaces `ap_memory`; los buffers pueden residir en el módulo padre. Las cifras no describen toda la FPGA ni prueban funcionamiento en placa, cierre de timing post-route, inferencia completa o capacidad DDR. Las tablas internas posteriores tienen otros recuentos; no deben sumarse como si fueran módulos independientes. El dispositivo encontrado tampoco prueba que se utilizara una U50/U250: puede ser otro prototipo o una etapa anterior.

## Otras pistas y falsos positivos

`claude-sessions/f41190816.h`, duplicado en `osito-k/`, contiene registros de instalación de Vivado/Vitis 2025.2 bajo `/mnt/extra/bin/2025.2/`, con fechas de marzo de 2026. Esa pista describe el entorno, no el diseño.

No hubo coincidencias de Alveo ni referencias útiles a U250; los tres `U50` encontrados eran cadenas aisladas en un bloque binario. La única coincidencia inicial de `endmodule`, buscada sin distinguir mayúsculas, era un método C++ `endModule`. Las referencias examinadas a 512 GB correspondían al SSD WD SN740 y otros proyectos, no al banco DDR4 buscado. Un segundo `uvp` resultó ser una extensión multimedia en un catálogo MIME.

La búsqueda dirigida a `unified_core`, `uvp_hls`, encabezados HLS y comandos de síntesis solo localizó el bloque del informe. Una búsqueda adicional del nombre NTRPU y de «ntransformer processing unit» tampoco encontró material. No aparecieron archivos identificables `.xpr`, `.xci`, `.xdc`, `.v`, `.sv`, `.vhd`, `.tcl`, `.dcp` o `.bit` en el listado del archivo. Tampoco se localizó un esquema o pinout vinculable a esta tarjeta.

## Qué se puede reconstruir

Se pueden recuperar parámetros del submódulo, nombres de señales, operaciones aritméticas y referencias a líneas del programa. Eso permite orientar una reimplementación y comparar futuras síntesis. El informe no permite afirmar que se haya recuperado el código C++ original o su interfaz superior, el scheduler, los controladores DDR/PCIe, ni los tests.

El siguiente trabajo está descrito en [la propuesta de arquitectura](README.md). Cualquier código nuevo deberá marcarse como reconstrucción y verificarse de forma independiente. Esta revisión no compiló HLS, no simuló RTL y no ejecutó programas recuperados ni scripts de configuración de hardware.
