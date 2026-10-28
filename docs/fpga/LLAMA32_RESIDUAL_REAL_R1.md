# Residual real: operandos AXI → HLS → salida AXI

Estado 2026-10-15: **validación nativa aprobada; Vivado encolado**.
La composición une el [cargador](LLAMA32_RESIDUAL_LOADER_R1.md),
[propietario RAM](LLAMA32_RESIDUAL_OWNER_R1.md),
[HLS residual original](LLAMA32_RESIDUAL_R1.md) y
[escritor AXI](LLAMA32_RESIDUAL_STORE_R1.md).

## Implementación y contrato

`ntrpu_residual_real_r1.sv` contiene `ntrpu_residual_path_r1.sv` y el
adaptador de 18 puertos `ntrpu_residual_hls_ports_r1.sv`. Se compilan 17
fuentes: siete RTL propios, nueve Verilog originales de Vitis y el banco.
Los hashes y el comando de compilación verifican esa lista efectiva.

Cada solicitud captura secuencia y época de 64 bits, dos regiones de
8.192 bytes y el destino. El llamador conserva ambas fuentes inmutables
hasta el ACK final y reserva un destino privado separado; la composición
no incorpora un asignador que compruebe alias entre todas las regiones.
El resultado contiene 2.048 FP32 y se ofrece únicamente tras los 128 B
correctos y la liberación del propietario. Escrituras parciales por error
o cancelación permanecen privadas; no se revierten.

La cancelación y los cambios de época revocan la publicación. Los canales
ya emitidos drenan; un peer que no responde requiere reset coordinado de
todos los participantes. Una respuesta retenida mantiene su diagnóstico,
pero su descriptor puede revocarse antes del ACK.

## Datos y validación

Cuatro pares de operandos de la referencia post-FFN, capa 0, tokens
`0`, `42`, `128000` y `128255`. El primer operando es el residual retenido;
el segundo, la proyección FFN `down` convertida a FP32. Los resultados
esperados solo sirven para comparación. La suma se recalcula con una
referencia independiente de enteros en unidades de 2⁻¹⁴⁹, con redondeo
al par y subnormales. El banco usa el HLS real para producir la salida.

El banco ejecuta 48 trabajos por semilla (`1`, `7`, `42`): identidades
incorrectas, regiones cortas, error R, overflow aritmético, último B
erróneo, cancelaciones en distintas etapas, revocación retenida, timeout
del último B, reset durante lectura/cálculo/escritura y recuperación.

- **144 trabajos**, 135 completions y 15 resets coordinados.
- **87 vectores exactos ofrecidos**, 178.176 palabras / 712.704 bytes.
- **78 vectores aceptados sin revocación/reset**, 638.976 bytes.
- **147 corrupciones de trazas** y **12 mutantes RTL** rechazados.

La revisión reconcilia R, escrituras del HLS, AW, W, B, completion y ACK.
Los mutantes alteran identidad, direcciones, cancelación, publicación,
operandos y longitud; cada uno falla con una aserción explícita, sin
contabilizar timeout ni error de compilación como detección.

## Medidas de simulación

| Intervalo | Ciclos |
| --- | ---: |
| Inicio → fin del HLS, 2.048 sumas correctas | 4.110 |
| HLS con overflow finito, sin salida | 2.062 |
| Primer AR → último B, ambos incluidos | 12.941–13.000 |
| BEGIN → completion | 12.946–13.005 |

Las tres semillas consumieron 471.714, 471.619 y 471.510 ciclos. Incluyen
las pausas del banco; no representan latencia DDR física, frecuencia
alcanzada ni rendimiento de inferencia.

## Evidencia y siguiente etapa

Carpeta local: `D:/AI/ntransformer/artifacts/ntrpu-residual-real-native/`.
Archivos y miembros verificados también en NFS:

| Evidencia | SHA-256 |
| --- | --- |
| Fixture de operandos/salidas | `c432ffad4e67a791dfc1302e8f6f95b21403c1c9b198f5165d14c42b355e5ce5` |
| Nativo R3, `20261015T171953Z` | `639e337e7b893297b2261f1ee8459c595f2d581ec4fb4d0ef030e7b2d5b229de` |
| Mutantes R1, `20261015T172413Z` | `fc9ea746c900e23ef4cbc76506d95b0f6a037f109f3aedc3978e1ce5539402c2` |
| Revisión independiente | `8ea1ba1db666eb62dc7b2471ef98a21950571cb4075d44fa16f7b36d6a938e42` |
| Fuente Vivado sellada, 58 miembros | `eb796426a8421e6738574df87deda653f7df5140d33d74c992a9f0dbc48d1e4b` |

Los intentos nativos R1/R2 fallaron por listas incompletas de compilación;
se conservaron sus diagnósticos. R3 corrigió el runner y el empaquetado,
manteniendo intactos el RTL y el banco. Los fallos previos del generador
se conservan como revisiones independientes. El paquete Vivado corrige,
antes del sellado, una sustitución de nombres que duplicaba el escritor
en la lista; exige 17 nombres distintos y correspondencia con el Tcl.

`queue_llama32_residual_real_vivado_r1.py` espera la revisión y respaldo
del escritor R1. Después ejecutará síntesis OOC a 5 ns y tres semillas
XSim, con CPU0/1, dos hilos y límite de 1.800 segundos. Conserva las
dependencias de [la cola](board/VIVADO_QUEUE.md).

Sigue conectar la salida de la FFN viva a esta composición y conservar
su propiedad hasta el ACK final. El [controlador de traspaso](LLAMA32_FFN_RESIDUAL_HANDOFF_R1.md)
ya aprobó 342 casos nativos y 24 mutantes con peers controlados; su síntesis
y XSim esperan a esta composición. Faltan timing integrado, DDR4/PCIe de
placa, inferencia completa y pruebas físicas; este banco usa operandos
post-FFN retenidos, no ejecuta toda la capa.
