# PCIe, DDR y FFN en un mismo banco

Actualizado el 2026-10-04, 14:43 UTC. **PASS del banco frío D32/H32 con
MIG y RDIMM genéricos: transferencias, avisos y último retorno verificados.** Dependencias: [cola V7](VIVADO_QUEUE.md),
[top con filtro](FFN_GUARDED_TOP.md) y [PCIe/FFN](../PCIE_FFN_INTEGRATION.md).

## Conexiones y alcance

`ntrpu_pcie_ddr_ffn_app.sv` conecta el root PIPE y XDMA existentes con
`ntrpu_ffn_guarded_fabric`, el MIG generado y dos modelos RDIMM genéricos.
El motor usa el reloj UI real de MIG; el host usa `user_clk` de XDMA.
El filtro recibe direcciones de 64 bits y entrega 37 bits a un canal
exploratorio de 128 GiB, DDR4-1600 x4 con ECC. No homologa un DIMM comercial.

El banco espera reset observado de XDMA, interfaces inactivas, enlace,
calibración y admisión antes del primer acceso. Es un adaptador de arranque
frío para simulación; no resuelve recuperación de DDR durante tráfico.
Los arrays locales solo verifican datos: las respuestas de lectura vienen
de MIG. Se registran AW/W/B/AR/R aceptados, incluidos W anteriores a AW.

## Criterios de aceptación

- Una FFN D32/H32: 32 FP32 finales exactos y todos los buffers intermedios.
- 3.328 bytes H2C, 704 escritos por cómputo y 128 C2H; reconstrucción
  independiente de bytes observados tanto en PCIe como en MIG.
- Dos trabajos con dimensiones inválidas, identidad ABI2, dos MSI-X,
  finalización por polling y MMIO concurrente con DMA.
- ID, LAST, strobes, respuestas y finalización posterior a todas las B.
- Procedencia de fuentes/IP/HLS/CDC, avisos revisados y respaldo local/NFS.

Los trabajos inválidos de FFN no sustituyen pruebas de direcciones XDMA
fuera de rango. Tampoco se califican aquí todos los ranks/capacidad,
reset caliente, driver, pesos reales, timing integrado ni hardware.

## Tandas y diagnóstico

| ID abreviado | Resultado |
| --- | --- |
| `20261004T121229Z` | Preparación fallida, 14,648 s: el parser confundió el módulo auxiliar `short` con la aplicación. Sin simulación. Corregido con selección explícita del módulo; parche probado sobre copia real de los seis imports AMD. |
| `20261004T121651Z` (`r2`) | Salida 1, 557,915 s. DDR calibró a 28.051.098 ps; el monitor falló a 28.055.998 ps, antes de admitir tráfico. |
| `20261004T123643Z` (`r3`) | Salida 0, 1.717,862 s. Transferencias y FFN exactas; 8.208 avisos de modelo aún sin calificar. Respaldada. |
| `20261004T130150Z` | Salida 0, 1.706,094 s. 1.339 eventos idénticos a `r3`; todos los avisos correlacionados en el tramo completado. Último retorno periódico posterior a `$finish`; revisión completa rechazada y tanda respaldada. |
| `20261004T133315Z` | Salida 0, 1.719,276 s. PASS de revisión estricta, 94 alteraciones rechazadas; drenaje completo y respaldo local/NFS verificado. |
| `20261004T140242Z` | Incompleta por comprobación incorrecta de `idle_stopped`; rechazo H2C observado, recuperación no alcanzada. Fallo respaldado. |
| `20261004T144107Z` | Salida 0, 2.074,383 s. Siete casos DMA y revisión estricta PASS, 163 alteraciones rechazadas; respaldo local/NFS verificado. |

El fallo de `r2` se reprodujo en una simulación mínima con el XSIM instalado
y `debug off`: una función `void` de comprobación observa un orden incorrecto
respecto a las asignaciones del monitor. El mismo caso pasa con `debug typical`;
una tarea procedural o comprobación inline también pasan con `debug off`.
La tarea rechaza un reloj deliberadamente cambiado de 5 a 6 ns. `r3` cambia
solo esa función por una tarea; conserva los criterios y el RTL del dispositivo.

Los fallos se conservan como fallos, con fuentes y todos los miembros verificados:

- `121229Z`: SHA-256 `fde020a7d2c6fcf466658a1b35f1695442539f8ddd5b2947f63f914955636d5e`.
- `121651Z`: SHA-256 `91da1b34f93beba32e18e87822b383bfea260660da738733cd9b038ab5438091`.

Los archivos están en `artifacts/ntrpu-board-vivado/<run_id>/` y el volumen
NFS de la sesión. `r3` tiene 1.635 archivos verificados, SHA-256
`d4526e3f57299890926f9136e4e04b0874ecdd3daa3c81f43cb43dd13979e8ad`.

La observación parcial de `r3` ya contiene muestras UI a 28.055.998 y
28.060.998 ps y avance hasta 45 µs: superó la comprobación que detuvo `r2`.
No es todavía una validación de transporte ni de resultados.

Una observación posterior encuentra 6.192 avisos `Model:WARNING: Reading
unwritten address` hasta aproximadamente 72,7 µs. El primero ocurre a
29.193.811 ps, antes de la admisión a 54.125.998 ps y sin ningún AW/AR de
la aplicación en ese intervalo. Los controles DDR anteriores no contienen
esos avisos. **Diferencia sin resolver:** conservar la tanda, contrastar
comportamiento del modelo/compilación y secuencia de reset, y no aprobarla
solo por una eventual coincidencia numérica. El revisor exige que no existan
avisos de modelo con prefijo que no se hayan explicado; no se silencian.

El revisor de procedencia se ejercitó contra los archivos reales de `r2`:
266 unidades compiladas, incluidas 120 de MIG/modelos, y nueve alteraciones
rechazadas de RTL, MIG, CDC, HLS, geometría, parches, avisos e identidad.
Esto no convierte esa simulación fallida en aprobada. El nuevo verificador
también detecta errores Micron con prefijo de jerarquía (`model:ERROR:`);
un examen suplementario de seis bancos fríos certificados no encontró tales
errores. La revisión terminal de `r3` reconstruye todas las transferencias;
su estado es `TRANSFERS_REVIEWED_MODEL_WARNINGS_UNQUALIFIED`.

## Resultado numérico y de transporte de r3

La referencia independiente confirma 32 FP32 finales y todos los buffers
intermedios. MIG acepta 23 AW, 63 W, 23 B, 15 AR y 64 R: 4.032 bytes
escritos y 4.096 leídos. El host carga 3.328 bytes, el motor escribe 704 y
C2H devuelve 128. Se verifican ID, LAST, strobes, respuestas, finalización
posterior a las B, dos trabajos inválidos, dos MSI-X y MMIO concurrente.
Hay 204 escrituras y 67 lecturas MMIO; se rechaza un servicio con identidad
incorrecta. El revisor detecta 41 alteraciones de evidencia.

JOB→DONE mide **2.429 ciclos UI de 5 ns: 12,145 µs simulados** para D32/H32.
Solo se ejercita rank 0, con 145 ACT. Esta cifra no mide hardware, capacidad
completa ni inferencia de un modelo. Persisten 8.208 avisos: 57 ráfagas de
ocho tiempos, separados 625 ps, con 18 mensajes por tiempo; primero a
29.193.811 ps y último a 86.018.186 ps. El revisor completo los rechaza
hasta contar con una correlación explícita; la revisión parcial no los elimina.

## Observación de las lecturas periódicas

El RTL generado de MIG contiene una FSM que inyecta una lectura cuando
transcurre un intervalo sin lecturas normales. Esta configuración usa
200 ciclos UI; el controlador conserva una dirección para esas operaciones.
El calibrador distingue su retorno mediante `rdInj` y lo excluye de
`rdDataEn` hacia la interfaz nativa. Esto identifica un mecanismo posible;
todavía no atribuye cada aviso de la simulación a dicho mecanismo.

La tanda `20261004T130150Z-V7-ffn-supervised-pcie-ddr-periodic` conserva
el trabajo completo y añade observadores de solicitud/aceptación/retorno
periódico, READ CAS, validez nativa y comandos en pines DDR. No cambia el
RTL de MIG, el contenido inicial de memoria ni sus avisos. Arrancó con
cuatro hilos y presupuesto de 3.600 s mientras se conservaba `r3`.
Se exige correlacionar los avisos con operaciones internas y revisar que
ningún retorno periódico se publique como datos de usuario.

La primera observación contiene solicitud a 29.100.998 ps, aceptación a
29.130.998, CAS inyectado a 29.160.998, READ en pines a 29.178.186 y retorno
a 29.225.998 con `injected=1`, `native=0`. El primer aviso queda 15,625 ns
después del READ físico. Son observaciones parciales, sin cierre de la tanda.

El RCD explica los dos grupos de nueve chips: su lado B invierte direcciones
con máscara `0x22bf8`, banco/grupo con `3` y columnas altas con `0x3f8`.
Conserva los bits de comando A16/A15/A14. El observador actual guarda `adr`
completo, pero su campo `cmd` extrae A17:A15: reconstruir RAS/CAS/WE de
`adr[16:14]`, registrar esta limitación y corregir la etiqueta en una futura
revisión. Los READ/WRITE sí quedan capturados por el filtro actual.

## Correlación completa de avisos y límite de cierre

`130150Z` conserva exactamente los 1.339 eventos previos y sus tiempos,
además de los 8.208 avisos de `r3`. Hasta 221.040.998 ps se reconstruyen
184 solicitudes/aceptaciones/CAS/retornos periódicos y 64 lecturas normales.
Cada uno de los 456 tiempos con avisos corresponde a una de 57 ráfagas
inyectadas: fila abierta por ACT, READ físico, direcciones A/B transformadas
por RCD y retorno con `native=0`. No hay avisos de lecturas normales.
La historia de WRITE físicos confirma que esas direcciones aún no estaban
escritas. El registro se conserva sin filtrar diagnósticos.

El revisor detecta 43 alteraciones del testigo parcial y 41 de transporte.
Una alteración de 1 ps en ACT expuso una comprobación ausente; se añadió
la verificación de fase DDR y se repitió la prueba. Estos controles no
convierten el prefijo en una simulación completa.

La última solicitud sí llega a READ, pero `$finish` ocurre a 222.036.000 ps,
19.998 ps antes del retorno esperado. El resumen es 185 solicitudes y
184 retornos: el revisor estricto lo rechaza. No se afirma haber observado
esa respuesta. Respaldo local/NFS: 1.637 archivos, 280.859.612 bytes,
SHA-256 `fcece097e251d57b4473767acbd3c8e8a24c3e94344e014bc35d0fe9a91a5576`.

La revisión `133315Z` espera flancos negativos de UI hasta igualar los cuatro
contadores, con límite de 199 flancos. No desactiva lecturas periódicas ni
modifica MIG, modelos o datos. Su revisión y respaldo quedan en cola después
de la terminación real. Después corresponde el [ensayo de errores XDMA](PCIE_DDR_ERROR_PLAN.md).

## Drenaje final verificado

`133315Z` terminó a 222.058.498 ps simulados. La espera adicional tomó
cinco flancos negativos, 22.498 ps, e incluyó el retorno que antes quedaba
fuera del registro. Se observan 185 solicitudes, aceptaciones, CAS y
retornos periódicos, más 64 lecturas normales: 249 READ físicos y 63 WRITE.
Todas las respuestas inyectadas quedan fuera de la validez nativa de usuario.

La revisión reconstruye los 8.208 avisos sin eliminarlos y conserva los
1.339 eventos anteriores con idénticos tiempos y valores. Rechaza 94
alteraciones de transporte, direcciones, avisos, relojes, retornos y cierre.
Las 32 salidas FP32 siguen exactas, con 2.429 ciclos UI de cómputo. Se
verifican las fuentes realmente compiladas, IP, HLS, CDC y modelos contra
las referencias certificadas. La tanda de errores `140242Z` se lanzó solo
después de este PASS. El respaldo quedó verificado a las 14:06:38 UTC:
1.641 archivos, 280.955.975 bytes, SHA-256
`b46d0073f566e0bc338d8bd8af9d8df6848f0b45e6c3d19b88e91137d5deb4b4`.

Este cierre califica el banco frío y sintético descrito aquí. Continúan
abiertos el DIMM exacto, reset caliente, capacidad/ranks completos desde
PCIe, propiedad de buffers, implementación física e inferencia completa.

La extensión de errores `140242Z` terminó incompleta al interpretar mal
`idle_stopped` en el estado XDMA `0x4040`. El rechazo H2C observado devuelve
DECERR sin llegar a MIG, pero no se alcanza la recuperación. Está revisada
como fallo y respaldada. `144107Z` pasa con la comprobación corregida:
dos DECERR sin llegar a MIG, controles sin corrupción y recuperación de
ambos motores DMA. Conserva la FFN y completa 233 retornos periódicos
y 68 de usuario. La revisión posterior de atributos PCIe usa la misma
traza y rechaza 163 alteraciones; [diagnóstico, alcance y siguiente prueba](PCIE_DDR_ERROR_PLAN.md).

El banco de transporte `152800Z` añade once descriptores en cero y al final
de 128 GiB, con 193 alteraciones rechazadas. No repite cómputo. El prefijo
válido de un cruce se transfiere antes del DECERR: 64 bytes escritos en
DDR en H2C y 64 publicados al host en C2H. No hay alias hacia cero y ambos
motores se recuperan sin reset. Este resultado exige invalidar el trabajo
completo con error; no garantiza atomicidad ni propiedad de buffers.
