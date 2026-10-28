# Propiedad entre FFN y suma residual

Estado 2026-10-15 17:46 UTC: **controlador nativo aprobado; Vivado encolado**.
`ntrpu_ffn_residual_handoff_r1.sv` conserva la respuesta de la
[FFN con propietario](LLAMA32_OWNED_FFN_R1.md) mientras un consumidor
[residual real](LLAMA32_RESIDUAL_REAL_R1.md) utiliza sus dos vectores.
La unión de esos bloques y su mux AXI sigue pendiente; este banco valida
el controlador con peers de completion controlados.

## Contrato

| Productor / comando | Consumidor residual |
| --- | --- |
| Residual retenido, 8.192 bytes | Operando `input` |
| Resultado FFN `down`, 8.192 bytes | Operando `branch` |
| Destino privado del comando | Vector final de 2.048 FP32 |
| Secuencia y época públicas | Identidad de 64 bits, sin truncamiento |

El controlador captura descriptores al admitir el comando. Comprueba
alineación de 64 bytes, extensiones con aritmética de 65 bits y ausencia
de solapamiento entre ambos operandos y el destino. Las bases y tamaños
del resto del workspace FFN deben comprobarse en el wrapper que lo integra.

Un comando ofrecido al consumidor permanece estable bajo backpressure,
incluso si aparece una cancelación. Las respuestas del productor y del
consumidor permanecen retenidas hasta el ACK público final: ambos reciben
ACK en el mismo ciclo. Nunca se confirma una identidad ajena. Un peer
que pierde su respuesta o no drena exige recuperación/reset coordinado.

Errores, cancelación y cambios de época invalidan la publicación. Una
respuesta pública retenida mantiene su diagnóstico, pero su descriptor
se revoca inmediatamente ante cambios del productor/consumidor. Restaurar
los pines no reactiva el resultado. Las escrituras parciales pertenecen
al consumidor; este controlador no las revierte.

## Validación y evidencia

Banco `tb_ntrpu_ffn_residual_handoff_r1.sv`, 114 trabajos por semilla
`1`, `7`, `42`: identidades incorrectas, desbordamientos de direcciones,
regiones cortas, solapamientos, ofertas bloqueadas, respuestas malformadas,
retirada de peers, cancelaciones, revocación retenida, reset y recuperación.

- **342 trabajos**, 333 completions y 15 resets coordinados.
- **204 descriptores ofrecidos**; 180 aceptados sin revocación/reset.
- **324 ACK del productor y 261 del consumidor**, reconciliados por trabajo.
- **165 alteraciones de trazas** y **24 mutantes RTL** rechazados.

Los mutantes compilan y fallan con aserciones explícitas; no se cuentan
timeouts como detección. El revisor R2 reconoce que la completion puede
observarse tras NBA en el mismo ciclo que la respuesta del hijo. Conserva
el resto de criterios; no requirió modificar ni repetir el RTL/banco.

Respaldos locales en `D:/AI/ntransformer/artifacts/ntrpu-ffn-residual-handoff-native/`,
con hashes y todos los miembros verificados también en NFS:

| Evidencia | SHA-256 |
| --- | --- |
| Nativo `20261015T173820Z`, R1 | `6bae3ff02364b4cb24fe6046788bea9eb3e010f861117196649e19e7e9b0f70f` |
| Mutantes `20261015T174140Z`, R1 | `ace00b1edeedff895cd1fd6622b8014477cd69711e12b693b76eb07f5a73774f` |
| Revisión independiente | `592b6bf1563e566a0229d2e9d651b7f3bfb57384287aac1846a732dbafc5b22f` |
| Fuente Vivado sellada | `0be6f40e2053ad4a81da2ea6e328b917834f73808c76e38940e228342f87dd7e` |

`queue_llama32_ffn_residual_handoff_vivado_r1.py` espera aprobación y respaldo
de la composición residual real R1. Después ejecutará síntesis OOC a 5 ns
y tres semillas XSim, dos hilos CPU0/1 y límite de 1.800 segundos.
Las cuentas de este banco no son medidas de aritmética ni de inferencia.

## Integración siguiente

Avance 2026-10-15 18:17 UTC: los pasos 1–2 están implementados en el
[wrapper R3 y verificados con peers AXI de protocolo](LLAMA32_FFN_RESIDUAL_CONTROL_R1.md).
La conexión aritmética completa del paso 3 todavía necesita banco y
paquete propios; las fuentes y la etapa de este controlador permanecen
sin cambios.

1. Capturar los ocho descriptores FFN, el residual retenido y el nuevo
   destino en un wrapper. Validar el destino contra todas las regiones
   todavía reservadas, incluidas entrada, pesos y workspace, antes de
   lanzar FFN. Mantener los 64 bits de identidad y los extremos de 65 bits.
2. Conectar `source_completion_ready` al ACK del propietario FFN y las
   señales `child_*` al residual real. Conmutar los cinco canales AXI sólo
   después del drenaje FFN, usando `residual_bus_owned`; liberar ambos
   bloques tras el ACK final. Reset debe incluir los dos bloques y memoria.
3. Extender el banco FFN completo: residual retenido como región adicional
   de entrada y destino final independiente. El residual debe leer el
   resultado FFN desde los bytes W realmente confirmados por B, sin
   cargar salidas esperadas en la memoria del DUT. Comparar contra
   `job-001-layer-output-fp32.bin`, conservando los 53.477.376 bytes Q8
   originales y el ABI1 post-O exacto. El tráfico añadido esperado es
   16.384 bytes leídos y 8.192 escritos; medir sus ciclos en esa ejecución.

Siguen abiertos productor post-O vivo, inferencia completa, timing integrado,
DDR4/PCIe y validación física. Este controlador no cierra esos requisitos.
