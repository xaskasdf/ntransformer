# FFN completa con propiedad: resultado RTL R4

La tanda `20261015T164923Z-V7-llama32-owned-ffn-vivado-r4` pasó XSim,
revisión independiente y respaldo local/NFS. El coordinador liberó la
cola a las 20:35 UTC del 2026-10-15; el cargador residual R2 arrancó después.

## Alcance y exactitud

La prueba ejecuta gate, up, SwiGLU/Q8 y down completos de capa 0, con
dimensión 2.048 y FFN 8.192. Consume directamente el ABI1 de referencia
post-O del token 0, muestra 1, sin recuantizarlo. Usa 53.477.376 bytes de
pesos Q8 originales y HLS aritmético real.

La revisión comprueba los 56.754.176 bytes leídos y los 155.648 bytes
escritos, incluidos intermedios y los 8.192 bytes finales. El consumidor
solo lee intermedios confirmados por B: 1.179.648 bytes de lecturas dependen
de escrituras anteriores. Se conserva la identidad pública de 64 bits
y el resultado hasta el ACK, seis ciclos después de completar.

## Ciclos medidos en simulación

| Fase | Ciclos | Bytes leídos | Bytes escritos |
| --- | ---: | ---: | ---: |
| Gate | 6.454.374 | 18.874.368 | 65.536 |
| Up | 6.455.250 | 18.874.368 | 65.536 |
| SwiGLU/Q8 | 980.311 | 131.072 | 16.384 |
| Down | 6.442.730 | 18.874.368 | 8.192 |
| Total | 20.332.665 | 56.754.176 | 155.648 |

Hay 16.480 ráfagas AR, 886.784 beats R, 866 transacciones AW/B y 2.432
beats W. El banco introduce backpressure y retiene la finalización.
Los contadores corresponden a 101,663325 ms con el reloj simulado de 5 ns;
no son latencia de una placa, ancho de banda DDR ni tokens por segundo.

## Síntesis y límites

Vivado 2026.1, VU5P `xcvu5p-flva2104-2-i`, síntesis OOC a 5 ns:
41.040 LUT, 36.160 registros, 40,5 BRAM y 93 DSP. Setup estimado +0,755 ns,
hold +0,039 ns; sin blackboxes, latches ni bucles combinacionales.
Este resultado no sustituye colocación/ruteo ni presupuestos externos.

Se rechazaron 34 corrupciones de aritmética/propiedad, 14 de métricas y
siete de identidad del diseño. La prueba aritmética contiene un trabajo
completo; los fallos de protocolo pertenecen a los bancos de control previos.
Productor post-O vivo, residual final y atención/FFN completa siguen en
sus tandas dependientes. Todas las capas e inferencia completa están pendientes.

## Evidencia y respaldo

Revisión SHA-256:
`1f28cb40213920a357b7fd8f03af551697b4f747f90a81306ec31ea8a5ac76e7`.
Archivo terminal, 156.415.102 bytes:
`4a862eb606ca0527c9ced1351e987fd0aea03848629f50e2cc2af9de1f023403`.
Fuente sellada:
`cb1a49ed317ba1871b59d645f91c8fb201023172693958d2a16ec7109bdc7fbd`.
El archivo está verificado bajo `artifacts/ntrpu-board-vivado/`, con
almacenamiento físico en D: y copia NFS. No requiere una extracción local
completa para recuperar sus miembros.
