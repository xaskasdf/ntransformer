# Validación DMA aislada

Estado vigente: [VIVADO_STATUS.md](VIVADO_STATUS.md). Estas pruebas corresponden al
ejemplo XDMA 4.2 de AMD con BRAM, `xcvu5p-flva2104-2-i`, Gen3 x16 y AXI de
512 bits a 250 MHz. No califican DDR, P2P, coherencia ni rendimiento físico.
La extensión a [direcciones altas, páginas y descriptores encadenados](PCIE_HOST64_VALIDATION.md)
se registra por separado; este documento conserva los ensayos de 768 bytes.

## Cobertura

El banco de 16 bytes H2C/C2H pasó en `20261002T101447Z-V4-dma-progress`.
La ampliación a 768 bytes **pasó en PIPE** en `20261002T135615Z-V4-dma-multi-pipe-r2`.
Un descriptor por dirección; MPS y MRRS del endpoint se programan a 256 bytes
y se verifican por lectura del registro. La primera versión falló al comprobar
MRRS; la revisión leyó `0x113f` y completó tres lecturas de payload H2C, tres
escrituras C2H y dos lecturas de descriptor. La confirmación serial del fixture
corregido **pasó** en `20261002T140854Z-V4-dma-multi-serial-r2`.

| Región del host simulado | Intervalo de bytes |
| --- | --- |
| Descriptor H2C | `0x100–0x11f` |
| Descriptor C2H | `0x300–0x31f` |
| Origen H2C | `0x400–0x6ff` |
| Destino C2H | `0x800–0xaff` |

El destino/origen AXI es cero. El patrón `byte(i ^ (i >> 8) ^ 0xa5)` distingue
bloques consecutivos de 256 bytes. No se ensayan aún direcciones DMA sobre 4 GiB,
cruces de página, descriptores encadenados, interrupciones o resets durante DMA.

## Adaptación del banco AMD

`patch_ntrpu_pcie_multi.py` solo modifica una copia nueva del ejemplo y registra
hashes de entrada/salida. Conserva los modelos e IP RTL originales.

- `DATA_STORE` tiene 4096 bytes; el completador usa una dirección interna de
  11 bits. La prueba queda dentro de esos límites y evita solapamientos.
- La primera respuesta usa `DATA_STORE[lower_addr_+...]`; las continuaciones
  originales usan otro buffer y otra base. Se reparan sus 192 referencias de
  bytes, incluidos los datos del registro de paquetes.
- Los comparadores originales muestrean sin comprobar siempre VALID y el C2H
  presupone una cabecera. Se sustituyen por esperas al scoreboard independiente.
- El scoreboard AXI compara ambos sentidos, direcciones, tamaños, respuestas y
  la correspondencia de LAST con la longitud de cada ráfaga.
- El scoreboard CQ comprueba cada handshake, cabeceras, direcciones, máscaras,
  datos, fin de paquete y estabilidad durante pausas. Exige modo DWORD de
  512 bits sin straddle; no pretende aceptar todos los formatos PCIe.

Formato contrastado con AMD: [descriptor CQ](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Completer-Request-Descriptor-Formats),
[señales CQ](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Completer-Request-Interface)
y [alineación](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Data-Alignment-Options).

## Validación del verificador

`20261002T133639Z-V4-cq-check-unit`: PASS en XSIM 2026.1, 64,069 s reales.
Un flujo válido de 768 bytes, tres paquetes C2H, seis ciclos de pausa y huecos
VALID pasó a 264 ns simulados. Siete pruebas negativas rechazaron corrupción,
dirección duplicada, LAST adelantado, máscara incorrecta, control desconocido,
cambio durante pausa y finalización prematura. Es una prueba del verificador;
no acredita funcionamiento del DMA.

Respaldo local verificado: 56 archivos, 56.249 bytes comprimidos, SHA-256
`92288744d22ba7537e1a381bcecb5b91b04a3acc0be4b31d7e144084370e8962`.

El revisor de logs también pasó siete pruebas Python, incluidas truncación,
duplicación, conteos/direcciones alterados y errores posteriores a una marca de
éxito. Está registrado como `ntrpu_pcie_multi_review` en CTest:
`ctest --test-dir build/ntrpu-sim -C Debug -R '^ntrpu_pcie_multi_review$' --output-on-failure`.
En generadores de una sola configuración puede omitirse `-C Debug`.

## Resultado DMA de 768 bytes

XSIM completó a **114,3411 µs simulados**, en **565,855 s reales** incluyendo
compilación. Comparaciones exactas H2C/C2H: 768 bytes cada una. AXI: 12 beats W,
12 R, tres AW y tres B OKAY, un AR y respuestas R OKAY. El scoreboard comprobó
direcciones, máscaras y LAST por ráfaga; ambos motores terminaron con estado `6`.
CQ: 20 handshakes, diez ciclos de pausa, tres paquetes C2H y tres lecturas H2C
de 256 bytes, más los dos descriptores. No hubo pausas W/R en este ensayo.

| Evento | Tiempo simulado, µs |
| --- | ---: |
| Enlace EP activo | 54,0481 |
| Primer / último beat W | 100,524 / 100,592 |
| Primer / último beat R | 107,892 / 107,936 |
| Primer / último payload C2H en CQ | 108,164 / 108,220 |

Son ventanas breves de actividad, no medidas de caudal sostenido. Se conserva
el respaldo de 111.084.214 bytes, 836 archivos, SHA-256
`ac648a991289d4fe9224d273fc7b1b8a754d0dc9ac2781771c17d1167d74eb18`.

La revisión automática original dio un **falso FAIL**: interpretaba ramas Tcl
impresas con `##` y una nota informativa como errores de ejecución. Se conservó
intacta. `20261002T140844Z-V4-dma-multi-review` verificó nuevamente el mismo log
SHA-256 `2e3864782f3abda2f32ed5bae0c1213ef7de6f7e239bfeb87dda0c8b29f5a8f2`;
el revisor excluye solo ecos de código y esa nota exacta, y sigue rechazando
errores ejecutados. Su documento resultó vacío por disco lleno, por lo que
`20261002T141751Z-V4-dma-review-complete` conserva la auditoría escrita completa.
Esta última tiene respaldo verificado SHA-256
`d72d7156bdada44bdbb41cc763fc449f046d63c8d4eba90a1a432669f5d4a97b`.

Avisos retenidos: índices de canales XDMA no usados, caso `unique` inicial y,
para PIPE, 16 diferencias de ancho 70/84 bits y dos puertos seriales sin conectar.
No se suprimieron diagnósticos ni se cambió RTL del núcleo AMD. El PASS queda
limitado al flujo simulado observado.

## Ejecución y aceptación

Tras cargar `settings64.sh`, siempre desde una copia identificada:

```bash
vivado -mode batch -source tools/ntrpu_board_pcie_cq_unit.tcl \
  -tclargs DIRECTORIO_NUEVO_UNIT
vivado -mode batch -source tools/ntrpu_board_pcie_sim.tcl \
  -tclargs EJEMPLO_AMD_LIMPIO DIRECTORIO_NUEVO_DMA multi
python3 tools/check_ntrpu_pcie_multi.py LOG_XSIM --json REVISION_NUEVA.json
```

Exigir las dos marcas finales CQ/AXI, conteos y secuencia exactos, enlace Gen3
x16 confirmado y ausencia de errores. Un código cero del simulador no basta.
Guardar también advertencias, fuentes, parámetros y respaldo SHA-256 por tanda.

## Aceleración de futuras tandas

Se generó una variante separada con `CONFIG.pipe_sim=true` mediante
`ntrpu_board_pcie_probe.tcl PARTE DIRECTORIO_NUEVO pipe`.
AMD documenta [PIPE como opción exclusiva de simulación](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/PIPE-Mode-Simulation).
`20261002T134243Z-V4-pcie-pipe-example` terminó en 88,205 s: se comprobaron
las 16 conexiones PIPE y el reloj EP→RP del ejemplo generado. Respaldo local:
627 archivos, 68.978.122 bytes, SHA-256
`f499fa60e83e1ade3dc23dfa7e33e34b59343255ea042d315fcce76ba149526f`.
El ensayo `20261002T134534Z-V4-dma-multi-pipe` levantó Gen3 x16 y enumeró,
pero **falló antes del DMA** a 94,416 µs: lectura de Device Control `0x013f`
frente a MPS/MRRS esperados. La tarea AMD de etiquetas extendidas escribe
`0x00000100` con byte enable `0010` en `0x078`, borrando el MRRS programado
antes. La nueva revisión hace lectura-modificación-escritura de los dos bytes
de control después de enumerar; conserva los demás bits y no escribe Device
Status. Se exige lectura de retorno antes de transferir.

Intento fallido: 512,933 s totales, CPU XSIM 238,050 s; 836 archivos y
108.886.086 bytes respaldados, SHA-256
`cc20d8c832885962ce72e1a99eab7c94aa096ee78f53a55a42f393be9be1c4c8`.
El código de proceso fue cero; el verificador lo rechazó por el fatal y por
faltar datos y marcas finales. Esto confirma la necesidad de revisar el log.
No usar PIPE para implementación.

## Comparación serial y PIPE

`20261002T140854Z-V4-dma-multi-serial-r2` terminó con salida 0 y revisión
numérica PASS. Los hashes de los dos scoreboards y del adaptador del fixture
coinciden entre ambas fuentes. La serial también verifica Device Control
`0x113f`, 768 bytes W/R, 12 beats por sentido, tres AW/B, un AR, ocho peticiones
CQ, 20 handshakes y diez pausas CQ. No hubo pausas AXI W/R.

| Medida | PIPE | Serial |
| --- | ---: | ---: |
| Duración total, segundos reales | 565,855 | 2143,482 |
| Preparación hasta tiempo 0 (`launch_simulation`), segundos reales | 237 | 461 |
| CPU del kernel XSIM, segundos | 292,390 | 1650,680 |
| Enlace EP activo, µs simulados | 54,0481 | 75,900505 |
| Primer / último W, µs simulados | 100,524 / 100,592 | 122,792405 / 122,860405 |
| Primer / último R, µs simulados | 107,892 / 107,936 | 130,464405 / 130,508405 |
| Primer / último C2H en CQ, µs simulados | 108,164 / 108,220 | 130,84048 / 130,89648 |
| Fin, µs simulados | 114,3411 | 137,01758 |

PIPE redujo el tiempo total observado por un factor de **3,79**. Son dos
ejecuciones en una VM compartida con otras tandas, sin repeticiones estadísticas.
Se reutiliza PIPE para regresiones funcionales de DMA; sus tiempos no sustituyen
la simulación serial ni las medidas físicas. La coincidencia comprobada abarca
este fixture y sus conteos, no equivalencia de todos los modos PCIe.

Log serial SHA-256 `6a9302ec352fd4569284fbde9637fd580bfdaa82db84c5a1c5c838113f422aab`.
Revisión manual: sin ERROR/Fatal ejecutados; se conservan 84 avisos VRFC 10-3705,
27 VRFC 10-9543, cuatro VRFC 10-5021 y uno de cada Project 1-5713, XSIM 43-3431,
VRFC 10-3027 y XSIM 43-4328. Son los diagnósticos del ejemplo ya examinados;
no se suprimen ni se declara resuelto el RTL interno protegido. Esta revisión
completa el campo manual pendiente del respaldo automático, sin modificarlo.
Respaldo local verificado: 949 archivos, 309.199.067 bytes, SHA-256
`b6da0b391559712d585977cf1f2acc8adb993e3bf4019ad75eb2e485c9263870`.

## Pausas AXI controladas

`ntrpu_sim_ready_gate.sv` es un elemento exclusivo de simulación, sin almacenamiento
ni CDC. Retrasa W entre XDMA y BRAM y R en sentido inverso. Una vez ofrecido un
beat, mantiene VALID hasta aceptarlo aunque cierre su ventana periódica. Comprueba
estabilidad de payload completo, igualdad de handshakes y progreso. AW/AR/B se
conectan directamente; no se ensaya reset con tráfico.

`20261002T144222Z-V4-ready-gate-unit` pasó en XSIM: 64 payloads independientes,
317 ciclos de espera del productor, 221 del consumidor, 389 ciclos de estímulo;
fin a 1580 ns. Dos negativos rechazan cambio de datos retenidos a 26 ns y una
variante defectuosa que retira VALID a 38 ns. Duración 44,118 s, solo aviso
XSIM 43-3431 en cada elaboración. Respaldo: 80 archivos, 80.397 bytes, SHA-256
`88d7faf20ff0fc9cefddc01d70eb0069833fd24aeb08fcb2fafe97f438c27f6a`.

La prueba completa `20261002T144323Z-V4-dma-stalls-pipe` **pasó**, salida 0,
532,662 s reales y fin a 114,4451 µs simulados. Conservó los 768 bytes exactos
por sentido, tres paquetes por sentido, 12 beats W/R, tres AW/B, un AR y estados
DMA `6`. Hubo **17 ciclos de espera del productor W (XDMA)** y **14 del productor
R (BRAM)**. El consumidor XDMA no retuvo RREADY: su contador `r_stalls` es cero;
recibió huecos RVALID introducidos por la compuerta. No atribuir las 14 pausas a
backpressure generado por el propio XDMA.

W primero/último: 100,544/100,644 µs; R: 107,968/108,040 µs; C2H en CQ:
108,240/108,324 µs. CQ conserva 20 handshakes y diez pausas. Preparación hasta
tiempo 0: 228 s reales; CPU del kernel XSIM: 268,080 s. No son medidas sostenidas.
El gate comprueba payload, máscaras, LAST y, en R, respuesta/ID durante esperas.

Revisión manual del log `0f52d3fa8b389a7706b1814cecba1dda4e0e0cff0a0d81497881937831218ad5`:
sin errores ejecutados. Conserva los avisos PIPE anteriores y uno nuevo
VRFC 10-2938 por declarar una señal escalar después de su uso. Se corrigió solo
el orden en `patch_ntrpu_pcie_stalls.py`; `20261002T145416Z-V4-stalls-wrapper-compile`
compiló gate y wrapper sin avisos/errores en 10,198 s. `review.json` demuestra que
el wrapper es idéntico salvo el movimiento de esa declaración. Es comprobación
de compilación, sin repetir ni atribuir una simulación funcional a la revisión.
Esta revisión manual completa el pendiente del respaldo automático sin alterarlo.

Respaldos verificados: DMA 839 archivos, 111.111.467 bytes, SHA-256
`0d615c4c6ee4ada48b38090514ced1467c92717d24d3fc5c6ac4824a532d6056`;
compilación 18 archivos, 27.362 bytes, SHA-256
`3b05ea93b2ba87cfdb835c2f9dd7ac65cae874850506dbe4a09199318c2376f4`.

Para repetir, usar modo `multi-stall` en `ntrpu_board_pcie_sim.tcl`. Exige dos
registros `NTRPU_AXI_GATE_COMPLETE`, 12 beats y pausas efectivas por productor.
El revisor admite `--require-stalls` para rechazar un log sin esa cobertura incluso
si pasa el fixture sin pausas. El watcher lo exige según los argumentos de la tanda.

## Siguientes pruebas, condicionadas al resultado

1. Ampliar pausas a AW/AR/B y combinaciones de ventanas; W/R ya pasan con el
   fixture de 768 bytes. Mantener chequeos por canal y de finalización conjunta.
2. El [fixture host64 de 1536 bytes](PCIE_HOST64_VALIDATION.md) ya pasa con
   memoria acotada nueva, cadenas y fragmentación en 4 KiB. Ampliar palabras
   altas, rangos y longitudes manteniendo sus rechazos de alias/permisos.
3. Incorporar interrupción y reinicio con tráfico, con comprobación de descriptores
   y datos pendientes. La integración sobre DDR mantiene las dependencias V2/V3.

La ampliación host64 cambia conjuntamente el receptor CQ y el completador:
el ejemplo limpio declara `lower_addr` de 11 bits en `usp_pci_exp_usrapp_rx.v`
y lo calcula con `cq_data[10:2]`; `TSK_TX_COMPLETION_DATA` recibe solo `[10:0]`.
Conservar únicamente siete bits en el campo PCIe Lower Address no autoriza
truncar la dirección usada para buscar datos. La nueva memoria del host debe
decodificar los 64 bits antes de calcular un índice local, mantener regiones
separadas para descriptores/payload y rechazar accesos fuera de ellas. Probar
regiones con iguales bits bajos y bits altos distintos, límites de página y
lecturas de bytes no inicializados. El checker CQ deberá conservar esa dirección
completa y comprobar la fragmentación; el patrón de datos distinguirá regiones.
