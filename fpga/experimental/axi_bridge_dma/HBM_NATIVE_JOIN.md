# Unión nativa PCIe, HBM y relojes

## Alcance

Esta composición privada continúa [HBM_MEMORY_TIMING.md](HBM_MEMORY_TIMING.md)
sobre `xcvu47p-fsvh2892-2-e`. Reutiliza el RTL seleccionado, pero lo sintetiza
de nuevo junto con los IP físicos. El margen OOC anterior de +1 ps no se
transfiere a esta composición.

El perfil es **AXI Bridge**, PCIe Gen3 x16, AXI de 512 bits a 250 MHz,
HBM de 16 GiB y dos stacks. No es el perfil histórico XDMA de dos canales
C2H ni el nodo DDR de 512 GiB. El único cambio de configuración del puente
respecto del donante es `pipe_sim=true` → `false`.

## Integración

`prepare_hbm_native_join.py` crea un top a partir del ejemplo AMD conservando
su aviso de licencia. Une 35 señales de solicitud/control al núcleo privado
y 13 señales de lectura al adaptador HBM, con acceso por PC00 y PC16.
La apertura host del ejemplo termina en BRAM. Las escrituras HBM están
desactivadas: aún no existe carga de pesos en este top.

| Dominio | Fuente | Frecuencia prevista |
|---|---|---:|
| PCIe / transporte | Salida AXI del puente nativo | 250 MHz |
| Lector | Clocking Wizard desde referencia de 100 MHz | 200 MHz |
| AXI HBM | Segundo reloj del mismo Clocking Wizard | 450 MHz |
| Gestión / ambos APB | Entrada independiente con BUFGCE | 100 MHz |
| Referencias HBM | Una entrada con BUFGCE por stack | 100 MHz |

Clocking Wizard seleccionó multiplicación 13,5, división de entrada 1 y
divisores de salida 6,75 y 3. HBM recibe los relojes AXI/APB/referencia;
no proporciona el reloj AXI del lector.

`fabric_clock_reset` y `configuration_rst_n` tienen funciones distintas.
El secuenciador externo debe arrancar los relojes, esperar
`fabric_clock_locked` y después liberar el reset de configuración.
Los resets canónicos del núcleo alimentan HBM y APB. La pérdida de lock
afecta disponibilidad y genera fallo; no se usa para borrar obligaciones.
AMD restringe el uso del reset AXI HBM después de iniciar tráfico y conecta
los resets AXI internamente cuando el switch está habilitado.
Véase [PG276: puertos](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions).

## Evidencia y ejecución

`generate_hbm_native_join.tcl` genera siete IP juntos. La generación terminó
en 79 segundos. La primera síntesis falló por 34 conexiones PIPE que el
ejemplo todavía conservaba. La revisión elimina ese andamiaje de simulación
y lee una sola vez `ntrpu_hbm_read_return_cut`, cuya definición también
está incluida, idéntica, en el adaptador congelado.

`elaborate_hbm_native_join.tcl` exporta netlist, relojes, timing, CDC, DRC,
utilización y orden de compilación. Rechaza black boxes y latches.
`inspect_hbm_native_join.tcl` reabre el checkpoint para comprobar los
41 consumidores de reloj y 34 consumidores de reset nativo, incluyendo
las primitivas físicas HBM. También comprueba 48 conexiones, 1.458 bits,
admitiendo redes constantes separadas solamente si ambos valores coinciden.
`review_hbm_native_join.py` verifica manifiestos, perfiles originales,
primitivas físicas y evidencia de esa reapertura.

El runner usa dos CPU, prioridad reducida y límite de 1.200 segundos por
etapa. Trabaja únicamente en `ntrpu-side/axi-bridge-hbm-native-join/`.
La síntesis `synth-20261011T190234Z` terminó en 682 segundos, con cero
errores, black boxes o latches. Exportó 58.403 LUT, 61.235 registros,
92 RAMB36E2 y una RAMB18E2. El checkpoint tiene SHA-256
`06b029a3845840b95fc123b6203d4080891e872b2d8d3f3a69b4c08374a5e6af`.

La primera auditoría consultaba los puertos jerárquicos HBM. Vivado fusionó
las entradas equivalentes de reloj/reset y dejó varias entradas redundantes
en constante; por ello esa auditoría falló y se conserva. La revisión consulta
los pines de las 32 primitivas AXI, dos APB y dos referencias HBM.

La segunda auditoría, `inspect-20261011T191930Z`, confirmó los 41 relojes
y 34 resets físicos. Comprobó 1.455 de 1.458 bits de interfaz, incluidos
siete bits con constantes equivalentes en redes distintas. **El criterio
completo permanece FALLIDO**: `m_awsize` conserva continuidad demostrada
en un bit de tres y `m_bresp` en uno de dos. Todavía hay que distinguir una
optimización válida del IP de una conexión incorrecta; no se omiten esos
bits ni se consideran aprobados.

`REVIEW.json` declara `acceptance_passed=false` y
`native_wiring_qualified=false`. El revisor estricto rechaza esta tanda.
Su modo `--diagnostic --self-test` informa lo comprobado, conserva los
hallazgos y rechaza diez controles negativos adicionales. Este modo no
concede aprobación del diseño.

```bash
python -B review_hbm_native_join.py GENERATION_RESTORED SYNTH_RESTORED \
  INSPECT_RESTORED --diagnostic --self-test
```

Para reproducir, usar los snapshots `source/` de cada tanda: generar los IP
con `run_hbm_native_join.py --stage generate`; sintetizar en otro directorio
con `--stage synth --generation GENERATE/output`; auditar con
`--stage inspect --checkpoint SYNTH/output/synth.dcp`. Todas las etapas
requieren `--prepared`, `--settings` y un destino nuevo dentro del namespace
privado. `prepare_hbm_native_join.py` reproduce el top y `wire_pairs.tsv`.

## Hallazgos abiertos

- `check_timing`: cero `no_clock`, `multiple_clock`, `generated_clocks`,
  `loops` y `latch_loops`. Hay 1.154 endpoints sin restricción por reloj
  constante, todos en XSDB de HBM; 802 entradas y 405 salidas sin presupuesto
  externo. Esto no constituye cierre de restricciones.
- CDC: 183 `CDC-3`, ocho `CDC-6`, 32 `CDC-9`, dos **`CDC-10` críticos** y
  1.746 `CDC-15`. Los dos críticos llegan a `ready_meta_reg[0:1]` desde los
  APB nativos: la combinación de `apb_complete` y `fabric_clock_locked`
  introduce lógica antes del sincronizador. Falta sincronizar `LOCKED` y
  registrar la disponibilidad en un dominio definido antes del cruce.
- Timing estimado tras síntesis: WNS −0,194 ns, WHS −0,842 ns y margen de
  ancho de pulso −0,092 ns. No hay placement/routing ni cierre integrado.
- DRC: `AVAL-326` por LOC ausente de `refclk_ibuf`, `NSTD-1` por estándares
  I/O y `UCIO-1` por pines. No se rebaja la severidad de estas comprobaciones.
- La referencia `fabric_refclk` sustituye otra definición de 10 ns emitida
  por Clocking Wizard. Hay además 35 avisos de declaraciones de redes después
  de su primer uso. La comprobación bit a bit mantiene abiertos los tres
  bits indicados arriba.

## Condiciones para la siguiente etapa

1. Resolver la continuidad de `AWSIZE[2:1]` y `BRESP[0]` con evidencia del
   netlist/RTL del IP o una prueba dirigida; conservar el fallo hasta entonces.
2. Corregir los dos `CDC-10` de disponibilidad y validar arranque, pérdida de
   lock, reloj detenido y conservación de obligaciones. Adelantar las
   declaraciones de redes en el generador del wrapper.
3. Resolver la doble definición de la referencia del Clocking Wizard y
   adaptar las restricciones CDC al prefijo `core/` y a los relojes generados.
   Verificar cobertura; no reemplazarlas por excepciones globales sin prueba.
4. Integrar los puertos de comandos/configuración/ACK en un bloque de control
   interno. Son interfaces de integración, no pines de una tarjeta terminada.
5. Completar carga de pesos y ACK remoto antes de probar tráfico extremo a
   extremo. Temperatura, CATTRIP y paridad se exponen, pero no tienen aquí una
   política de recuperación validada.
6. Cerrar pines, osciladores, alimentación y secuenciación. AMD exige que la
   referencia HBM provenga de un GCIO del SLR correspondiente y especifica
   jitter de la fuente menor de 3 ps RMS. Véase
   [PG276: relojes](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking).
7. Hacer implementación y validar timing, CDC, DRC y tráfico sobre esta
   composición completa antes de afirmar cierre físico.

## Respaldo

Los paquetes y restauraciones locales están en
`D:/AI/artifacts/ntrpu-axi-bridge-hbm-native-join/`.
El paquete consolidado es `20261011-hbm-native-join.zip`; su recibo externo
`FINAL_BACKUP.json` registra SHA-256, tamaño y verificación de restauración.
Se conservan también el intento fallido y los cuatro archivos `.mem` de
inicialización HBM en `hbm-init-support/`, con hashes y comparación de las
copias de generación y síntesis. Los manifiestos describen exactamente lo
archivado; no se incluye una instalación completa de Vivado. Para recrear
un proyecto en otra ubicación, regenerar los IP desde las XCI y scripts.
`FINAL_PROCESS_AUDIT.json` registra el cierre de los cinco grupos de procesos
propios y confirma que los archivos del donante generado no cambiaron.
