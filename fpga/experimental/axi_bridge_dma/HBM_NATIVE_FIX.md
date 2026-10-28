# Disponibilidad registrada en la integración nativa VU47P

## Alcance

Esta revisión continúa [HBM_NATIVE_JOIN.md](HBM_NATIVE_JOIN.md) en el espacio
experimental del hilo lateral. Conserva los 16 módulos del núcleo, el binding
HBM, los siete IP y la red de reset de configuración. El perfil sigue siendo
VU47P, PCIe Gen3 x16 en modo AXI Bridge y dos stacks HBM de 16 GiB en total.
El nodo DDR de 512 GiB mantiene su diseño separado.

`prepare_hbm_native_fix.py` aplica tres cambios al snapshot anterior:

- Declara las 48 redes de interfaz antes de las instancias que las usan.
- Inserta `ntrpu_hbm_native_availability` entre los estados nativos y el
  sincronizador de disponibilidad del lector.
- Conserva la restricción de referencia que genera Clocking Wizard y elimina
  su definición manual duplicada; no altera las restricciones del IP.

## Disponibilidad y conservación de obligaciones

El nuevo módulo sincroniza `LOCKED` mediante tres registros `ASYNC_REG` en
gestión a 100 MHz. Ambos APB HBM usan ese mismo reloj de gestión. Después
registra la combinación de lock y finalización APB antes de cruzarla al lector.
No combina señales asíncronas delante del sincronizador del lector.

Una pérdida muestreada después del armado queda latente hasta el reset de
configuración. Esto permite observarla incluso si el reloj del lector se
detiene y el MMCM recupera lock antes de reanudarse. La pérdida de disponibilidad
no borra colas ni activa un reset operativo de HBM. Los supervisores existentes
cierran admisión y conservan las ofertas pendientes.

La captura exige que el dominio de gestión observe la pérdida. El RTL no
garantiza capturar pulsos arbitrariamente cortos ni simula metastabilidad.
La recuperación de este banco usa un nuevo ciclo de configuración; no valida
una secuencia de recuperación de HBM en una tarjeta física.

## Los tres bits optimizados

La auditoría anterior mantiene su resultado fallido como evidencia histórica.
La nueva revisión diferencia continuidad eléctrica del netlist y optimización:

| Bits | Evidencia exigida |
|---|---|
| `AWSIZE[2:1]` | Declaración de tres bits y conexión RTL completa, escritor con `AWSIZE=6`, salida del núcleo constante uno y wrapper sintetizado sin uso de la entrada `s_axib_awsize`. |
| `BRESP[0]` | Asignación a GND en el wrapper nativo, cero en el consumidor y continuidad del bit de error `BRESP[1]`. |

`inspect_hbm_native_fix.tcl` reabre el checkpoint y exporta las observaciones y
los wrappers de simulación funcional de ese mismo netlist. Su estado de exportación
no concede aceptación: `review_hbm_native_fix.py` debe comprobar las conexiones
del RTL, las constantes y los usos efectivos del wrapper. No admite excepciones
generales para bits desconectados. La descripción de los puertos puede consultarse
en [AMD PG194](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/AXI-Slave-Interface);
la explicación de estas optimizaciones proviene del netlist generado localmente.

## Banco y comprobaciones

El banco adapta `tb_axi_bridge_hbm_clocked.sv` y utiliza las colas CDC reales,
los resets exportados y los supervisores existentes. Compara el supervisor
PCIe con su versión congelada. Los datos de memoria son fixtures explícitos;
no se presenta este banco como tráfico PCIe/HBM nativo de extremo a extremo.

Son 20 escenarios con cuatro desfases de arranque: funcionamiento continuo,
relojes ausentes o detenidos en alto/bajo, pérdidas de enlace, fallo de adaptador,
finalización APB incompleta, pérdida de lock persistente o muestreada y recuperación
del lock con el lector detenido. Se comprueban dos colas de comandos y dos de
retorno, rechazo de admisión, conservación del contenido y reinicio de época.

La primera tanda aprobó 47 escenarios y agotó 45 segundos al arrancar XSIM
para el siguiente. Además, su runner incluyó por error su log cambiante como
entrada. Se conserva completa como fallida. La revisión del runner filtra las
extensiones de fuentes y permite 120 segundos por caso; el RTL del banco no cambia.

La repetición `fix-sim-retry-20261011T215910Z` aprobó los 80 escenarios en
524,6 segundos. Las fuentes sintetizadas y simuladas coinciden. El revisor
final devuelve `NATIVE_FIX_STRUCTURAL_AND_AVAILABILITY_REVIEWED` y rechaza
15 controles negativos. La preparación local reproduce los hashes del top,
del módulo de disponibilidad y del banco ejecutado.

La revisión estructural exige 41 consumidores de reloj, 34 resets nativos,
48 conexiones, siete registros de disponibilidad y dos cruces desde registros.
Comprueba que desaparezcan los dos `CDC-10` de disponibilidad y rechaza controles
negativos de cobertura, constantes, cableado, sincronización y relojes.

## Reproducción y respaldo

```bash
python -B prepare_hbm_native_fix.py FROZEN_PREPARED NEW_PREPARED
python -B run_hbm_native_availability.py NEW_VM_OUTPUT \
  --prepared PREPARED --settings /path/to/Vivado/settings64.sh
python -B review_hbm_native_fix.py GENERATION_RESTORED SYNTH_RESTORED \
  INSPECTION_RESTORED SIMULATION_RESTORED --self-test
```

Síntesis e inspección usan `run_hbm_native_join.py`, cada una en un directorio
nuevo del namespace privado. Para inspeccionar, copiar el nuevo Tcl con el nombre
`inspect_hbm_native_join.tcl` junto con `wire_pairs.tsv`. Mantener los snapshots
de entrada, los manifiestos y los cuatro archivos de inicialización HBM `.mem`.

Los resultados, intentos fallidos y restauraciones se guardan en
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-native-fix/`. La restauración por manifiesto
reutiliza archivos locales solamente después de verificar su SHA-256 y descarga
los restantes. No afirma haber descargado el archivo remoto completo cuando usa
este procedimiento. El paquete consolidado se restaura y verifica por separado.
`FINAL_BACKUP.json` registra nombre, SHA-256 y restauración del ZIP;
`FINAL_PROCESS_AUDIT.json` confirma el cierre de 140 grupos de procesos propios,
los 236 archivos del donante sin cambios y el hash intacto del checkpoint previo.

## Resultados de la síntesis corregida

La tanda `fix-synth-20261011T214558Z` terminó en 741 segundos, sin errores,
advertencias críticas de síntesis, black boxes ni latches. Utiliza 58.407 LUT,
61.242 registros, 92 RAMB36E2 y una RAMB18E2: cuatro LUT y siete registros más
que el checkpoint anterior. Su SHA-256 es
`4613dbb14df1eff881c3e5294825d6398a9af37e3f100f37ab52cd017ab2a89e`.

La reapertura `fix-inspect-20261011T215835Z` exportó los 41 relojes, 34 resets,
1.455 bits por continuidad/constante equivalente y los tres bits optimizados.
Los siete registros nuevos sobreviven; ambos estados APB provienen de registros
del reloj de gestión, y las dos entradas del sincronizador lector reciben Q
de registros de disponibilidad. Vivado expresa los puntos de inicio de reset
como pines C; el revisor complementa esa evidencia con la inversión de reset
y sus siete consumidores CLR en el wrapper funcional.

`CDC-10` pasa de dos a cero. Permanecen 183 `CDC-3`, nueve `CDC-6`, 32 `CDC-9`
y 1.746 `CDC-15`. El nuevo `CDC-6` corresponde al vector de disponibilidad
registrada; queda visible para la revisión CDC completa, sin waiver. El lector
exige todos los bits de disponibilidad y ocho muestras consecutivas para arrancar.

Timing tras síntesis: WNS −0,194 ns, WHS −0,842 ns y margen de ancho de pulso
−0,092 ns. Estos resultados siguen sin placement/routing y no son cierre de timing.
La restauración de síntesis reutilizó 234 archivos verificados y descargó 21,
unos 25,5 MB, conservando también los cuatro `.mem` HBM.

## Límites pendientes

Esta revisión no acredita cierre de CDC global, timing integrado, placement,
routing ni pinout. Persisten presupuestos externos y restricciones por revisar,
además de la política térmica/paridad, integración del bloque de control, carga
de pesos y ACK remoto. Tampoco demuestra inferencia de Llama ni agregación de
memoria entre tarjetas. El siguiente paso debe partir de estos resultados
verificados y de los hallazgos físicos que sigan abiertos.
