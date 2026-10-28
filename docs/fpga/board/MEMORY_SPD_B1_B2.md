# SPD público de alternativas Micron de 64 GB

Investigación del 2026-10-06. **No se ha seleccionado otro DIMM ni
modificado MIG.** El candidato F1 sigue sin SPD público recuperable.

## Datos obtenidos

La búsqueda de la familia en la herramienta pública de Micron devuelve
`MTA36ASF8G72PZ-2G9B1` y `MTA36ASF8G72PZ-2G9B2`. El catálogo identifica
64 GB, x72, ECC, 1,2 V y 2933 MT/s; ambas partes aparecen obsoletas.
[SPD B1](https://www.micron.com/products/obsolete/obsolete-rdimm/part-catalog/part-detail/spd-data/mta36asf8g72pz-2g9b1),
[catálogo B2](https://www.micron.com/products/obsolete/obsolete-rdimm/part-catalog/part-detail/mta36asf8g72pz-2g9b2).

Cada exportación contiene 102 filas para nueve códigos de proceso:
VI, UI, VL, VK, UL, TI, TL, UK y TK. Son alternativas documentales
separadas de F1; sus datos no se atribuyen a esa revisión.

## Auditoría local y límites

`audit_micron_spd_family_r1.py` reconstruye únicamente los bytes 0–127
cuando cada intervalo coincide exactamente con su longitud hexadecimal.
Las 18 tablas cubren ese prefijo sin huecos ni solapamientos. El CRC-16
CCITT, inicializado a cero sobre los bytes 0–125, coincide con los bytes
126–127 leídos en orden little endian. La ejecución del script reprodujo
íntegramente el informe inicial.

La web entrega solo 50 bytes para intervalos reservados de 115, 64 y
128 bytes. No se rellenaron los huecos ni se generó una imagen EEPROM
completa. El CRC base no valida el resto del SPD ni el modelo funcional.
El informe conserva valores crudos de geometría, registro y revisión.

## Siguiente trabajo de V2

Contrastar geometría y timings con una especificación aplicable; confirmar
código de proceso, revisión de DRAM y RCD; obtener ficha y modelo
correspondientes y estudiar disponibilidad antes de elegir un módulo.
Solo después preparar una configuración MIG separada y verificar dos
DIMM por canal, 128 GiB y ECC, conservando el objetivo de cuatro canales.
Las tablas públicas no aprueban esa topología ni la placa de 512 GiB.

Respuestas originales e informe:
`build/cloud-setup/20261002-board/micron-spd-family-audit-20261006.json`.

## Geometría y tiempos base contrastados

Los 18 procesos comparten los mismos primeros 128 bytes, SHA-256
`4464ea3bff7c9f665b34c92537b8d43acb40fb819cea6fe50356cc5210e44dec`.
`tools/check_ntrpu_micron_spd_base.py` exige cobertura exacta, ausencia de
solapamientos y CRC antes de decodificar ese bloque. Cuatro controles
rechazan un tiempo corrompido, un byte ausente, un solapamiento y un rango
truncado.

| Campo decodificado | Valor B1/B2 |
| --- | --- |
| Capacidad útil por módulo | 68.719.476.736 bytes, 64 GiB |
| Organización | Monolítico, dos ranks x4, bus útil 64 + extensión 8 bits |
| Direccionamiento | 18 bits de fila, 10 de columna, 2 BA y 2 BG |
| tCK mínimo / máximo del SPD, sin ajuste de tasa | 682 / 1.600 ps |
| tAA / tRCD / tRP | 13,750 / 13,750 / 13,750 ns |
| tRAS / tRC | 32,000 / 45,750 ns |
| tRFC1 / tRFC2 / tRFC4 | 350 / 260 / 160 ns |

Geometría, CAS y 17 tiempos coinciden con el decodificador sin modificar
de [i2c-tools 4.4](https://mirrors.edge.kernel.org/pub/software/utils/i2c-tools/).
El paquete coincide con el SHA-256 publicado; no se verificó su firma PGP.
A 1.250 ps, el cálculo base da CL11 y RCD/RP/RAS de 11/11/26 ciclos;
ese periodo está dentro del intervalo SPD. Es análisis de los campos
presentes, sin completar CWL, registros del RCD ni restricciones eléctricas de dos slots.
Dos módulos sumarían 128 GiB y cuatro canales 512 GiB; no prueba esa placa.

El programa externo avisa del archivo truncado, pero muestra
`OK (0x0000)` para el CRC de un bloque ausente y características físicas
derivadas de bytes no suministrados. Esas salidas se excluyen explícitamente
de la comparación. No se rellenó el SPD ni se aceptó la salida global como
válida. El informe reproducible y sus logs son
`micron-spd-base-crosscheck-r1-20261006.json` y archivos asociados en la
orquestación local; se conserva también la licencia del programa original.
