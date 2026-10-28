# Implementación física del receptor con dos destinos MIG

## Alcance

Experimento lateral que implementa el receptor y el supervisor de
[la prueba con dos MIG](PEER_RX_DUAL_MIG.md), conservando su RTL validado.
Comprende el pool R4, dos convertidores AXI de reloj, los filtros de
capacidad y créditos, y el arranque coordinado de una única sesión fría.
La colocación de los controladores DDR4/PHY y su pinout de placa quedan
fuera de esta frontera de implementación.

El dispositivo exploratorio continúa siendo `xcvu5p-flva2104-2-i`.
Vivado 2026.1 trabaja sobre una copia aislada, con dos CPU, `nice=10`,
dos hilos y timeout. No se modifica la cola del hilo principal.

## Relojes y frontera de medición

| Dominio | Período | Frecuencia |
| --- | ---: | ---: |
| Gestión independiente | 10 ns | 100 MHz |
| Origen | 5 ns | 200 MHz |
| UI del destino 0 | 5 ns | 200 MHz |
| UI del destino 1 | 5 ns | 200 MHz |

Se usan cuatro BUFGCE y una incertidumbre explícita de 0,100 ns para
setup y 0,050 ns para hold. La coincidencia de frecuencias de ambos UI
no presupone una relación de fase física: sus cruces usan las IP
asíncronas y el supervisor.

El banco OOC registra 812 bits en el origen, ocho en gestión y 651
por destino. Son registros para medir caminos internos, **no un
adaptador AXI**. El reset externo llega directamente al supervisor.
Los puertos externos no tienen delays ni asignación de pines. Los
recursos del banco se separan de los del núcleo implementado.

## Reset, CDC y criterios de aceptación

El flujo exige quince cadenas propias de tres FDCE, `INIT=0` y
`ASYNC_REG=TRUE`. Sólo sus pines `CLR` reciben nuevas excepciones de
timing. Los caminos posteriores siguen temporizados. Se conservan
las restricciones de las XPM y de los dos convertidores AXI, y no
se añaden excepciones globales entre pares de relojes.

La auditoría revisa conectividad, relojes, valores iniciales y
colocación de los sincronizadores; inspecciona CDC con y sin waivers,
bus skew, excepciones ignoradas y cruces aún temporizados. Que una
cadena tenga `ASYNC_REG` no reemplaza esta revisión. Véanse
[AMD UG912: ASYNC_REG](https://docs.amd.com/r/en-US/ug912-vivado-properties/ASYNC_REG)
y [AMD UG906: reglas CDC](https://docs.amd.com/r/en-US/ug906-vivado-design-analysis/Understanding-the-Clock-Domain-Crossings-Report-Rules).

Se exige ruteo completo, márgenes no negativos de setup, hold y pulso,
ausencia de latches, black boxes y errores DRC, y ningún endpoint
interno sin restricciones. Los avisos se conservan y se revisan por
causa; no se ocultan para obtener un resultado positivo.

## Regresión del netlist

Se exporta el núcleo completo desde el checkpoint ruteado. La adaptación
del banco integrado cambia únicamente el nombre de la instancia del DUT.
El banco conserva sus 56 comandos con dos MIG, manteniendo scoreboards,
observadores y controles de corrupción de trazas. Los 13 casos unitarios
del supervisor se ejecutan como referencia **RTL**, sin modificar su banco.

El primer intento extrajo también `core/startup`, pero la reconstrucción
de jerarquía de Vivado había movido lógica a través de sus puertos:
el módulo exportado ya no conservaba la interfaz RTL. Falló la elaboración
con `VRFC 10-2996`, sin evidencia funcional. Se conserva ese intento y el
netlist original. No se repara ni sustituye el netlist ruteado; la regresión
integrada usa el núcleo completo, que sí mantiene su interfaz.

Esta simulación usa primitivas sin SDF. Comprueba comportamiento del
netlist, mientras los informes STA evalúan los tiempos internos. No
modela metastabilidad analógica, comportamiento eléctrico de DIMM,
PCIe/P2P, HBM ni rendimiento sostenido de una tarjeta.

## Estado y evidencia

La tanda `20261005T004006412225Z-dual-physical` completó síntesis,
colocación y ruteo en 650,677 s. Cerraron los tiempos internos:

| Dominio | Setup mínimo | Hold mínimo |
| --- | ---: | ---: |
| Origen | +0,479 ns | +0,007 ns |
| Gestión | +8,102 ns | +0,003 ns |
| UI 0 | +0,856 ns | +0,010 ns |
| UI 1 | +1,165 ns | +0,011 ns |

El margen mínimo de pulso fue +1,958 ns. Se rutearon las 23.225 redes
rutables, sin errores DRC, latches, black boxes ni endpoints internos
sin restricciones. Los 3 ps de hold son un margen pequeño **de este
experimento**, con la incertidumbre indicada; hay que repetir STA al
integrar fuentes de reloj, PHY y restricciones de placa.

El núcleo ocupa 7.161 LUT —752 como memoria—, 17.951 registros,
15 bloques BRAM y 238 CARRY8, sin DSP. Dentro de ese total, la jerarquía
del supervisor tiene 86 LUT y 148 registros. Las optimizaciones cruzan
jerarquías internas; esta atribución no es el coste de un supervisor
sintetizado por separado. Los ocho bancos representan 32 KiB lógicos;
el coste físico incluye también FIFOs, réplicas y control del receptor.

El peor camino de setup va de `active_generation_reg` a la habilitación
de `state_reg` en el receptor: 12 niveles lógicos y 4,116 ns de datos,
de los cuales 2,990 ns corresponden al ruteo. Esto señala el control y
su interconexión como candidatos a revisar antes de aumentar el pool;
no demuestra por sí solo el cuello de botella del caudal sostenido.
El hold mínimo está en el contador `source_probe/age_reg` de gestión.

La auditoría `20261005T005326861369Z-dual-audit` reabrió el mismo DCP
sin modificar restricciones. Validó 15 cadenas propias de reset,
cuatro XPM de reset y seis XPM de detección de reloj: 75 registros
sincronizadores, en cadenas de tres, cada cadena en un mismo slice.
El registro `opened` conserva `INIT=0`, carece de reset y su realimentación
mantiene el estado una vez abierto. Cinco corrupciones deliberadas de
los metadatos de auditoría fueron rechazadas por el verificador.

Las 20 restricciones de bus skew pasaron: skew máximo 1,200 ns y
slack mínimo +3,800 ns. Se revisaron 44 false paths, 20 max delays y
20 bus skews de las IP, diez false paths XPM y quince excepciones
propias dirigidas exclusivamente a los 45 pines CLR. No hay excepciones
ignoradas ni grupos globales de relojes asíncronos añadidos.

CDC normal y sin waivers coinciden: 26 informativos `CDC-3`, 38 `CDC-9`
y **un `CDC-26 Warning` abierto** desde el reset externo al PRE de la
primera etapa de gestión. Los tres PRE reciben directamente el puerto,
sin LUT ni RAM. Se conserva la discrepancia de clasificación, también
descrita en [el supervisor anterior](RESET_SUPERVISOR.md); no se añade
waiver ni se declara CDC completamente limpio. Las 28 definiciones de
waiver existentes pertenecen a las IP AMD.

`cross_clock_paths.tsv` cuenta objetos devueltos por `get_timing_paths`,
incluidos caminos ignorados; su encabezado no demuestra caminos sin
tratar. `cross_clock_classification.tsv` y `clock_interaction.rpt`
identifican seis pares de gestión excluidos, cuatro pares origen/UI
con max delay de 5 ns y ningún camino directo entre los UI.

Los avisos restantes incluyen puertos y fuentes de reloj sin colocación
física OOC, delays externos ausentes —680 entradas y 1.443 salidas—,
señales sin carga de las IP y 16 avisos `SYNTH-6` de BRAM sin registro
de salida integrado. Los informes completos quedan preservados.
La ausencia de errores no califica esta frontera como diseño de placa.

**El conjunto no está funcionalmente calificado después del ruteo.**
La tanda `20261005T005926824094Z-dual-netlist` pasó los 13 casos RTL,
pero no admitió ningún comando del banco MIG. Ambos controladores
estaban calibrados a 34,051829 µs; se detuvo la simulación después de
78 µs de progreso registrado, a los 656,545 s reales, para investigar
sin prolongar el gasto. Se guardaron el log, el snapshot compilado y
la identidad del único grupo de procesos detenido.

Dos repeticiones cortas del mismo snapshot, sin modificar sus fuentes,
observaron `rearm_ready=X` con `admit=0`. A 1–3 µs, los relojes ya se
detectaban vivos, BVALID/RVALID eran cero y los contadores generales
habían terminado; `source_held_m=1`, pero `memory_held_m=X`. Esto localiza
la condición bloqueante, **no demuestra aún su causa raíz**. Cuatro
consultas a nombres internos con corchetes no resolvieron objetos;
se conservan esos errores de inspección y no se infieren valores para
los contadores individuales a partir de ellos.

El diagnóstico `20261005T011140320483Z-dual-probe` sí abrió la admisión
del núcleo ruteado con relojes UI estables desde el inicio, respuestas
B/R inactivas y resets de 40/400 ns. No incluye MIG ni transferencia
de payload. Se preserva como control de alcance limitado.

También se probó una copia candidata que reinicia el contador de
memoria vacía mientras `ui_rst` esté activo o la memoria no esté
calibrada. La tanda `20261005T012115272553Z-dual-candidate` sintetizó
por separado el supervisor original y la candidata. **Ambos pasaron
15 casos**: los 13 anteriores y dos inyecciones B/R desconocidas antes
de calibración. Por tanto, este banco no distingue el fallo real y
**no acredita la corrección**. La candidata permanece dentro de los
artifacts, sin adoptar cambios en el RTL del repositorio.

Evidencia local: `D:\AI\artifacts\ntrpu-dual-mig-physical`. El respaldo
`20261004-dual-mig-physical-handoff.zip` reúne ocho tandas, fuentes,
checkpoints, snapshots compilados, ejemplo MIG original y revisores.
Su archivo `.VERIFIED.json` contiene SHA-256 y verificación de CRC y
hashes por miembro. Los 46 archivos previos vigilados, incluido
`AGENTS.md`, permanecen intactos.

La puerta de salida inmediata es reproducir `memory_held_m=X` con
las transiciones iniciales reales del UI/MIG y obtener un caso que
falle en el original y pase en la candidata. Después se debe repetir
la implementación integrada, CDC y los 56 comandos con comparación
de todos los bytes. Hasta entonces, quedan bloqueadas la adopción de
la candidata y cualquier conclusión de caudal del nuevo conjunto.

## Reproducción

Desde un snapshot aislado con Vivado 2026.1 disponible:

```bash
python3 -B tools/run_ntrpu_dual_physical.py /tanda-fisica/output --timeout 1200
python3 -B tools/check_ntrpu_dual_physical.py physical /tanda-fisica
# En nuevos snapshots, copiar el DCP o los dos netlists a inputs/evidence/.
python3 -B tools/run_ntrpu_dual_physical_audit.py /tanda-auditoria/output --timeout 300
python3 -B tools/check_ntrpu_dual_cdc.py /tanda-auditoria --physical /tanda-fisica
python3 -B tools/run_ntrpu_dual_routed_mig.py /tanda-netlist/output \
  --example /ruta/ejemplo/ntrpu_ddr_geometry_ex --timeout 1800
python3 -B tools/check_ntrpu_dual_physical.py netlist /tanda-netlist \
  --physical /tanda-fisica
```

Los runners requieren directorios nuevos y producen manifiestos de
entradas y salidas. Los verificadores crean `REVIEW.json` exclusivamente.
El resultado físico inicial se complementa con la auditoría CDC y la
regresión; sus límites permanecen explícitos en cada `REVIEW.json`.

## Próxima decisión: reservas frente a ancho de banda

Mantener R4 como referencia: dos bancos dedicados por destino y cuatro
compartidos, todos de 4 KiB. La entrada única de 512 bits a 200 MHz
ofrece un techo bruto de 12,8 GB/s; dos memorias no duplican esa entrada.
Los 16 KiB de reserva absorben como máximo 1,28 µs a ese caudal si están
vacíos y pueden utilizarse completamente. Créditos, descriptores y
fragmentación pueden reducir esa tolerancia.

La siguiente tanda debe medir, después del arranque y por destino:

1. Bytes aceptados por tiempo, ocupación máxima y distribución temporal
   de bancos, créditos y descriptores; separar pausas de datos y comandos.
2. Cargas equilibradas y sesgadas, con ráfagas de 64, 512 y 4.096 bytes;
   pausas independientes de W/B y tráfico periódico del MIG activo.
3. Latencia p50/p95/p99, duración de bloqueo y progreso del destino libre
   mientras el otro se detiene; drenar y verificar todos los datos.
4. Comparar tamaños de reserva solamente después de identificar qué
   recurso limita cada escenario. Exigir ganancia medible y repetir
   implementación/STA para cualquier cambio de pool o de entrada.

Este era el plan pendiente al cerrar la tanda física; su primera medición
está en [Rendimiento dual R4](PEER_RX_DUAL_PERFORMANCE.md). PCIe/P2P y HBM
requieren sus propios modelos, topología y restricciones antes de
atribuirles ese caudal o afirmar un espacio de memoria unificado.


## Actualización: variante de arranque validada

La versión original y sus fallos descritos aquí se conservan. Una copia
aislada supera ahora la reproducción del bloqueo y el banco integrado;
ver [Arranque frío dual](PEER_RX_DUAL_STARTUP.md) para el cambio,
los nuevos resultados físicos y el respaldo. La corrección fue integrada
el 2026-10-05 y el mismo núcleo ruteado superó la tanda de
[rendimiento dual R4](PEER_RX_DUAL_PERFORMANCE.md).
