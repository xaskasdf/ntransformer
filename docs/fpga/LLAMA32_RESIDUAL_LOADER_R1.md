# Cargador AXI de los dos operandos de la suma residual

Estado 2026-10-15 16:46 UTC: **nativo aprobado y respaldado; Vivado R2 en cola**.
Este bloque prepara el transporte de la segunda suma: residual retenido
de post-O y salida FP32 de FFN. Todavía no incluye la aritmética ni el escritor.

## Contrato y diseño

`fpga/rtl/ntrpu_residual_loader_r1.sv` carga dos regiones de 8.192 bytes,
cada una con 2.048 valores FP32, mediante el lector AXI existente. Captura
descriptores de 64 bits y comprueba identidad, alineación, límites con
aritmética de 65 bits, tamaño, datos finitos y respuestas AXI.

Divide las lecturas en páginas de 4 KiB. Usa dos RAM privadas, con salida
síncrona; sólo una respuesta correcta y no revocada permite leerlas. El
productor debe mantener ambos operandos inmutables hasta el ACK. Una
cancelación o cambio de época revoca la salida y drena el burst emitido.
Un peer sin respuesta exige reset coordinado: el watchdog no autoriza
reutilizar un puerto con tráfico pendiente.

Se deriva del cargador post-O calificado, retirando exclusivamente el
tercer operando y sus casos específicos. El caso de región vacía se aplica
ahora a la rama FFN. Los módulos y bancos originales permanecen intactos.

## Pruebas y evidencia

Tres semillas verifican 38 trabajos cada una: 114 casos, 86.016 palabras
RAM comparadas causalmente con datos aceptados por AXI y 78 controles de
revisión rechazados. Incluyen errores RRESP/RID/RLAST, NaN, desbordamiento
de direcciones, cancelación, cambios de época, retención y reset/recuperación.
Cada semilla ejecutó 81 AR y 3.275 R; entre 69.127 y 69.152 ciclos del banco.
Estos datos son sintéticos y no miden latencia de una inferencia.

Once mutantes RTL compilaron y fallaron con aserciones explícitas, sin
contar errores de compilación o timeout como detección. El revisor R2
corrige una sustitución de texto que había cambiado su permutación de
direcciones 31 por 21; RTL y banco no cambiaron ni se repitió la simulación.

| Evidencia D:/NFS | SHA-256 |
| --- | --- |
| Nativo `20261015T162007Z` | `c14308a73e75e224c73200148d3606bb2c42ba0fcbea42281ca91da8f522ae79` |
| Mutantes `20261015T162046Z` | `4858b5e4e0ce867c90cf67c35f7309ff310a03a5772fd78dfe8c19c29889bdb2` |
| Revisión independiente | `46fe9662407c7857491cd86455fd907d845d7e705e0d4fdf6908031a77721036` |
| Fuente sellada Vivado R2 | `79701d3e07a78ec1313b4f6e4294f18193fc9dad1d2f0c6e5db81e96accf2c5f` |

## Cola y continuación

`queue_llama32_residual_loader_vivado_r2.py` espera revisión y respaldo de
[FFN completa R4](LLAMA32_OWNED_FFN_R1.md). El sucesor conserva RTL, banco y
revisor; actualiza la dependencia tras recuperar la revisión física.
Ejecutará síntesis OOC a 5 ns y
tres semillas XSim, con CPU0/1, dos hilos y límite de 1.800 segundos.
Después: [propietario de RAM ya calificado en nativo](LLAMA32_RESIDUAL_OWNER_R1.md), unión al [núcleo residual](LLAMA32_RESIDUAL_R1.md)
y escritor que publique sólo tras B exitoso; finalmente integración post-O/FFN.
