# Unión de atención, proyección O y post-O

Estado 2026-10-15 15:58 UTC: **composición aritmética real aprobada y
respaldada**. Tres trabajos, dos pares exactos y una cancelación con
recuperación; 24.576 bytes publicados en 4.194.333 ciclos del banco.
La revisión rechazó 96 controles: 69 de raíz y 27 de post-O.

Síntesis revisada: 63.581 LUT, 57.943 FF, 111 BRAM y 145 DSP; WNS
−1,052 ns y WHS +0,030 ns a 5 ns. Sin cajas negras, latches, lazos o
extremos internos sin restricciones. No cierra timing de síntesis;
las E/S externas y el timing ruteado siguen pendientes. No es inferencia completa.

## Contrato del controlador

`fpga/rtl/ntrpu_attention_post_path_r2.sv` une la raíz proyectada R1 y
post-O real R2, conservando ambos bloques. Captura las direcciones al
aceptar el comando, valida límites de 65 bits y alineación de 64 bytes,
y rechaza solapamientos que destruirían los operandos de post-O.

La raíz conserva su respuesta mientras post-O consume O. Ambos ACK se
retienen hasta el ACK público final; mantenimiento y reutilización de
caché permanecen bloqueados. Se comparan secuencia y época completas de
64 bits. Cancelación, cambio de época o respuestas ajenas impiden publicar
descriptores válidos. El bus AXI pertenece a un solo hijo por etapa.

## Evidencia nativa

Tres semillas verificaron 165 trabajos: errores, cancelaciones por etapa,
revocación tardía, límites, captura de direcciones, respuestas retenidas y
reset/recuperación. La revisión independiente rechazó 36 controles.
Además, dieciséis copias RTL con defectos deliberados compilaron y fueron
detectadas por aserciones explícitas del banco. Son pruebas de protocolo
con hijos controlados, sin aritmética HLS real.

| Evidencia D:/NFS | SHA-256 |
| --- | --- |
| Banco nativo `20261015T091612Z` | `bd9c2cba695756c549cf03d5796bd54930c123041da1f107e7b52b1564a3a66a` |
| Dieciséis mutantes `20261015T092144Z` | `a70d44a6a29324f202c506b842e25db47c8b5f7827c3911d36801883e351dab7` |
| Compilación completa, firmas HLS vacías | `12cd91ef05fc8b3a63a60708b06f638968064ce944436d7232b127743f5e5294` |

## Banco con aritmética real

El paquete compila 131 fuentes: 41 RTL propios, 89 Verilog generados
por HLS y un banco. No incluye firmas vacías ni reproductores de salidas.
Los padres son la [raíz validada](LLAMA32_PROJECTED_ROOT_REVIEW_R4.md)
y [post-O validado](LLAMA32_POST_ATTENTION_REAL_R1.md).

Se ejecutan dos cadenas completas, separadas por cancelación de entrada Q.
Post-O recibe O exclusivamente desde AW/W confirmados por B exitoso.
Se conserva el verificador de raíz y se agregan trazas `APREAL_*` para
reconstruir causalidad, dirección, contenido y publicación del par residual
FP32/entrada FFN ABI1. Las referencias provienen del GGUF Q8_0 verificado
de Llama 3.2 1B Instruct; se comprueba su coincidencia entre ambos padres.
Objetivo del banco: dos pares, 24.576 bytes exactos, recuperación y síntesis
sin cajas negras ni lazos. El WNS de síntesis se informa sin confundirlo con
cierre temporal ruteado.

R1 (`20261015T142214Z`) falló antes de simular: una sustitución del nombre
del top duplicó su archivo y omitió el módulo raíz en Tcl. Diagnóstico
respaldado: `79f4a44dfc2c66570ffe94e24f11619b1a39f3b713bbd10de470785c0a191266`.
R2 restaura únicamente esa entrada y comprueba la lista completa de 41
fuentes; conserva RTL, banco y verificadores. Paquete R2 D:/NFS:
`fd86884a080ea7847dee489908a336507f5be71530ec2017636a13dbcc1fe215`.
Coordinador: `queue_llama32_attention_post_real_vivado_r2.py`.

La inspección estática detectó una suposición errónea del verificador R1:
`APREAL_COMPLETE` aparece antes de siete ciclos de respuesta retenida;
`PRREAL_COMPLETE` aparece después. El verificador R2 exige exactamente
esa separación y agrega un control negativo de ciclo. El observador
`queue_llama32_attention_post_real_review_r2.py` espera el resultado y su
respaldo para revisar el mismo archivo, sin relanzar Vivado/XSim. Su
aprobación terminal es `PASS_ATTENTION_POST_REAL_R2_XSIM_SYNTH`.

El verificador del suplemento R2 falló posteriormente por sangría en
Python remoto. La recuperación R3 verificó los mismos bytes en D:/NFS,
sin modificar la revisión ni repetir herramientas. Los tres sucesores
que se habían detenido antes de lanzar herramientas se reencolaron con
paquetes nuevos; las fuentes originales permanecen conservadas.

| Evidencia | SHA-256 |
| --- | --- |
| Resultado real `20261015T142808Z` | `41da622b888afb344a996e78317b898db71324bf638ce1300e31e7426c8b81f8` |
| Revisión independiente R2 | `28d739531bbcd194702bbd86a7c24545187dddc805022d379bbea19f074db7e3` |
| Suplemento verificado | `b33fb87d7051605aae15824a52f488b7735274f2bce8b557d301f781dfaeda3e` |

Los dos casos consumieron 16.384 bytes O previamente confirmados por B.
Post-O leyó 49.152 bytes y escribió 24.576. Cada invocación HLS tardó
173.353 ciclos; máximo intervalo sin progreso: 165.137. Son medidas
del banco RTL, no throughput físico ni tokens/s del modelo.

## Trabajo restante

Integrar FFN y la segunda suma residual. Permanecen pendientes timing,
DDR4/PCIe físico, inferencia completa y métricas en placa. El estudio
[ruteado de la raíz](LLAMA32_PROJECTED_ROOT_ROUTE_R1.md) mantiene WNS
−0,079 ns a 5 ns; la [optimización física R5](LLAMA32_PROJECTED_ROOT_PHYSOPT_R1.md)
ya terminó, fue revisada y respaldada: WNS −0,034 ns, hold +0,003 ns,
86.884 redes ruteables conectadas. No cierra setup a 200 MHz y mantiene
restricciones provisionales. La [FFN con propietario real](LLAMA32_OWNED_FFN_R1.md)
está en XSim. Se encolaron [FFN/residual real](LLAMA32_FFN_RESIDUAL_REAL_R1.md)
y el [control de enlace con post-O](LLAMA32_POST_FFN_HANDOFF_R1.md); la
conexión al productor vivo sigue pendiente.
La [continuación numérica post-O → FFN → residual](LLAMA32_POST_FFN_R1.md)
ya pasó cuatro casos nativos con ABI1 exacto, sin recuantización de entrada.
