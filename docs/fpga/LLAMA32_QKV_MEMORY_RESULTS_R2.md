# Resultado aritmético Q/K/V y revisión complementaria R2

Fecha: 2026-10-16. Tanda `20261016T124000Z-V7-qkv-memory-vivado-r3`.
**Aritmética y bytes revisados; contrapresión RVALID pendiente.**

Vivado/XSim terminó con código cero en 7.003,76 segundos. Los nueve
trabajos ejecutaron 21 matrices y 627 grupos: 19.968 resultados FP64 y
79.872 bytes FP32 coincidieron exactamente con la referencia. Se comprobaron
orden AXI, cancelaciones, retención y publicación. El banco acumuló
15.677.938 ciclos; esto no es latencia de inferencia del modelo completo.

## Diagnóstico conservado

El revisor R1 exigía al menos una espera en cada canal. Su última
comprobación falló porque hubo cero ciclos con `RVALID && !RREADY`.
La revisión R2 reproduce ese fallo y conserva toda la comprobación
aritmética y sus 53 controles negativos. Registra por separado:

- 3.121 esperas AR, 14 AW y 331 W.
- Cero esperas RVALID: esa condición aún no quedó ejercitada.
- 177.776 ciclos de huecos dentro de ráfagas de lectura, máximo tres
  ciclos por hueco. La disponibilidad intermitente de datos no sustituye
  la contrapresión del receptor.

No se repitió Vivado ni se cambiaron fuentes, trazas o diagnóstico R1.
La integración aritmética puede usar esta evidencia sólo si conserva
explícitamente la cobertura pendiente. La calificación completa del
canal R requiere otra prueba que provoque y compruebe esa condición.

## Síntesis y respaldo

La síntesis OOC para VU5P, reloj de 5 ns, informa 22.365 LUT, 18.900 FF,
38 BRAM y 54 DSP; WNS +1,28 ns. No acredita frecuencia tras ruteo ni
temporización de interfaces externas.

Captura original: 159 archivos restaurados, archivo de 411.039.805 bytes,
SHA-256 `ad0c5c343567eabe90fb670f627650f928031f2da1a6f3a374b8a96bdb6395ed`.
Fuente intacta: `02acc56f2629968a8c5c02460a6cd092a2526d6f1ad76b6dac9c155ffb7febda`.

Informe: `build/cloud-setup/20261002-board/qkv-memory-terminal-r2-review.json`.
El suplemento se guarda en
`artifacts/ntrpu-board-vivado/qkv-memory-review-r2-handoff-20261016-r1/`;
su `VERIFIED.json` registra hashes, dependencias y restauración D:/NFS.
