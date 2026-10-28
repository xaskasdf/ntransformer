# Próximo banco: errores DDR observados por XDMA

Actualizado el 2026-10-04, 15:28 UTC. **`140242Z` incompleto y respaldado;
`144107Z` PASS y respaldo verificado; cruce `152800Z` iniciado.**
Depende del cierre frío de [PCIe + DDR + FFN](PCIE_DDR_FFN.md).
Los rechazos [AXI directos](DDR_AXI_GUARD_ERRORS.md) ya están probados;
queda observarlos a través del motor DMA y sus registros reales.

## Contrato de error

Para canal 0, leer el estado no destructivo en `0x0040` (H2C) y `0x1040`
(C2H). Un DECERR de escritura debe llegar al bit 14 de H2C, máscara
`0x00004000`; un DECERR de lectura, al bit 9 de C2H, `0x00000200`.
Registrar también Busy, los demás errores y el contador de descriptores.
No confundir un error AXI con uno al obtener el descriptor PCIe.
El bit 6, `idle_stopped`, es estado de parada: conservarlo y registrarlo
por separado en los casos rechazados. No clasificarlo como otro error ni
admitirlo en una recuperación que debe terminar con estado exacto `0x6`.
Fuentes: AMD PG195 v4.2, [estado H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Status-0x40)
y [estado C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Status-0x40).

Habilitar todos los bits del campo de error correspondiente antes de Run;
el registro de control permite registrar el fallo y detener el motor.
Observar el retorno a reposo con un límite de sondeo. No reutilizar la
espera actual de éxito `status == 6` para estos casos.
Fuentes: [control H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Control-0x04)
y [control C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Control-0x04).

## Casos iniciales y evidencia

1. Conservar la FFN D32/H32, transferencias válidas, MSI-X y MMIO concurrente.
2. Emitir H2C y C2H de 64 bytes completamente fuera de los 128 GiB:
   usar una dirección válida del fixture más `1 << 37`. Conservar los
   64 bits en el descriptor, el log PCIe y los observadores AXI.
3. Reconstruir AW/W/B o AR/R antes del filtro: ID, longitud, LAST, strobes
   y DECERR. Demostrar que el intento inválido no produce AW/AR en MIG.
4. Releer el destino válido que resultaría de truncar la dirección:
   su contenido debe conservarse. Separar el buffer host usado por el
   intento C2H de los resultados válidos; registrar lo que XDMA realmente
   publique ante el fallo, sin suponer que omitirá todos los TLP de datos.
5. Detener/rearmar el canal por sus registros y efectuar una transferencia
   válida posterior. No resetear MIG para ocultar un problema de recuperación.

XDMA puede dividir un descriptor en varias transacciones. Un caso que cruza
el límite de capacidad debe revisar cada ráfaga: una porción válida previa
puede haberse escrito antes del error. No exigir atomicidad del descriptor
completo a partir del rechazo atómico de una ráfaga del filtro.
AMD describe esta división en [el canal H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel).

## Criterio de cierre

Exigir estado DMA real, error AXI correlacionado, ausencia de alias/corrupción,
recuperación posterior y drenaje periódico final. Conservar fuentes y logs
íntegros; probar que el revisor rechaza direcciones truncadas, errores
equivocados, respuestas perdidas y éxitos prematuros. Un PASS de este banco
no valida propiedad de buffers, reset caliente, cuatro canales ni placa.

## Banco preparado

`tools/ntrpu_board_pcie_ddr_errors.tcl` conserva la FFN y los dos trabajos
inválidos anteriores, y después ejecuta siete descriptores de 64 bytes:
C2H de control, H2C inválido, C2H de control, C2H inválido, C2H de
recuperación, H2C válido y C2H final. Los cuatro controles comprueban los
primeros 64 bytes del resultado FP32 conocido; el intento de escritura
inválida usa un patrón diferente. La dirección inválida suma `1 << 37`
a `0x1e0000fc0`, conservando los 64 bits del descriptor.

Los observadores de la aplicación reconstruyen tres interfaces: salida
XDMA, entrada del filtro y MIG. Las comprobaciones comparan datos, IDs,
respuestas, LAST, strobes y recuentos; exigen cero accesos MIG durante cada
intento inválido. El host usa regiones separadas para descriptor, fuente y
destino de estas pruebas. Una lectura rechazada conserva como evidencia si
XDMA publicó cero bytes o una carga de ceros, sin tratarla como resultado válido.

Se espera un descriptor completado para cada transferencia válida después
de volver a activar Run. El contador se reinicia con ese flanco; el valor
de los casos con error se registra sin atribuirle semántica de éxito.
Fuentes: [contador H2C](https://docs.amd.com/r/en-US/pg195-pcie-dma/H2C-Channel-Completed-Descriptor-Count-0x48)
y [contador C2H](https://docs.amd.com/r/en-US/pg195-pcie-dma/C2H-Channel-Completed-Descriptor-Count-0x48).

El parche fue ejercitado sobre los seis archivos originales certificados
del ejemplo AMD `193438Z`: respecto del parche anterior, solo cambian TX y
`sample_tests.vh`. Esto prueba la transformación de archivos, no la
elaboración ni el comportamiento del nuevo banco. El lanzador espera el
PASS de la revisión estricta de `133315Z`; luego crea una copia nueva de
fuentes y una tanda aislada de cuatro hilos con límite de 3.600 s. El
verificador de transferencias se ejercitará con el registro real; su estado
parcial no reemplaza la revisión periódica, la procedencia ni el respaldo.

`20261004T140242Z-V7-ffn-supervised-pcie-ddr-errors` arrancó después del
PASS estricto del banco de drenaje, cuya copia local/NFS ya está verificada.
Fuente `source-20261004T140232Z.zip`, SHA-256
`398651c73e02b034f32fb9db52061eb088ef4f6d3d031f3ae3bdb764885d0fc5`.
La tanda terminó con salida 1 en 1.825,665 s. Conserva íntegro el banco
anterior y completa la primera lectura de control. En la escritura inválida
observa AW de `0x21e0000fc0`, 64 bytes distintos, B=DECERR en XDMA y antes
del filtro, y cero handshakes MIG. El estado real `0x4040` detiene el banco:
su comprobación trataba `idle_stopped` como un error adicional. No llegó a
comprobar ausencia de corrupción, contador del caso inválido ni recuperación.
La revisión conserva además el aviso de declaración tardía de
`dma_error_mode`. Ninguno de estos resultados acredita los siete casos.

Respaldo local/NFS verificado: 1.646 archivos, SHA-256
`530aa8c5cbc58f62c65f10e9647ce2a49d9412ca71c27c082294cb3bc3771d6b`.
Las fuentes y revisores originales quedan conservados. La revisión r2
solo distingue el bit de estado documentado y adelanta la declaración;
mantiene el rechazo de los demás errores, direcciones, datos y controles.
El parche volvió a probarse sobre los seis archivos originales: respecto
de r1 únicamente cambia TX. La procedencia del banco certificado anterior
se volvió a verificar con sus 266 entradas compiladas.

`20261004T144107Z-V7-ffn-supervised-pcie-ddr-errors-r2` usa
`source-20261004T144044Z.zip`, SHA-256
`af45960f64cb36bb007beb1f884bef69234575cf7247622c1bb44999e12ecf45`.
Terminó con salida 0 en 2.074,383 s, a las 15:15:46 UTC. La revisión
independiente pasa y rechaza **163 alteraciones de evidencia**. Conserva
exactos los eventos y los avisos del banco FFN anterior: 32 FP32 y 2.429
ciclos UI. H2C inválido termina con `0x4040`, C2H inválido con `0x240`,
ambos con contador cero y sin acceso MIG. El C2H rechazado publica cero
bytes; las cuatro lecturas válidas recuperan 256 bytes exactos y la
escritura válida posterior completa 64 bytes. Las recuperaciones terminan
con estado `0x6` y contador uno, sin resetear MIG.

El drenaje final observa 233 solicitudes/aceptaciones/CAS/retornos
periódicos y 68 lecturas de usuario. Los 8.208 avisos siguen íntegros y
correlacionados. Se verifican las 266 entradas compiladas y la procedencia
de IP, HLS, CDC y modelos. El RTL del dispositivo y las IP no cambiaron.

La revisión posterior corrigió una suposición del parser: MRd usa
`attr=2` (Relaxed Ordering), tanto en el banco anterior como en los nueve
paquetes de lectura nuevos; los cuatro MWr nuevos usan `attr=0`.
Se exige concordancia exacta con las cabeceras PCIe independientes
`0x20002000 | longitud` y `0x60000000 | longitud`; no se enmascaran atributos.
La prueba negativa de resumen ausente también se corrigió para rechazar
explícitamente antes de indexar una lista vacía. Ambos fallos del revisor
quedan conservados; se volvió a revisar la misma traza, sin repetir Vivado.
Referencia: [atributos CQ en AMD PG213](https://docs.amd.com/r/en-US/pg213-pcie4-ultrascale-plus/Completer-Request-Descriptor-Formats).

Respaldo verificado a las 15:27:44 UTC: 1.652 archivos, 281.227.255 bytes,
SHA-256 `aeb1f4f0c84b9765655275c9dd197179c3b72c83a95bfd406fdfe56573d8f67a`.
Se conservan archivo, miembros y copia de fuentes comprobados en
`artifacts/ntrpu-board-vivado/20261004T144107Z-V7-ffn-supervised-pcie-ddr-errors-r2/`
y en el volumen NFS. El almacenamiento físico local está en D:.

## Siguiente dependencia: descriptor que cruza 128 GiB

Banco `20261004T152800Z-V7-ffn-supervised-pcie-ddr-boundary` iniciado en
Vivado tras satisfacer revisión y respaldo de la dependencia. **Terminó
con PASS estricto** a las 15:49:37 UTC, salida 0 y 1.293,109 s. Usa una
tanda corta de transporte tras calibración: conservar XDMA, CDC, filtro y
MIG reales, pero sin repetir la FFN. No atribuirle cobertura de cómputo.

Definir `C = 1 << 37`, `L = C - 64 = 0x1fffffffc0`. Un descriptor de
128 bytes desde `L` cruza simultáneamente una página de 4 KiB y la capacidad.
La fuente host debe quedar alineada y dentro de una página, por ejemplo
`0x600002000`; el destino C2H, en `0x700002000`, también. Mantener descriptor,
fuente y destino separados y rellenar el destino con un centinela previo.

Para cada byte `n = 0..63`, usar patrones distintos: `A = 0x3c ^ n`,
`B = 0x96 ^ n`, `C0 = 0xc3 ^ n`, `C1 = 0x69 ^ n`.

| Paso | Transferencia | Comprobación |
| --- | --- | --- |
| 1–4 | H2C de A a L y B a cero; C2H de ambos | Inicialización y lectura exactas antes del error |
| 5 | H2C de 128 B a L, carga C0 seguida de C1 | Reconstruir cada AW/W/B; DECERR de la porción fuera de capacidad |
| 6–7 | C2H de 64 B desde L y cero | Observar la escritura válida previa; cero conserva B, sin alias |
| 8 | C2H de 128 B desde L | Correlacionar AR/R válidos y rechazados, estado XDMA y bytes publicados |
| 9–11 | C2H válido desde L; H2C y C2H de B a cero | Recuperación de ambos canales sin reset MIG |

Registrar la división real de XDMA antes de imponer recuentos. AXI no
permite una ráfaga que cruce 4 KiB: verificar los segmentos individualmente,
manteniendo 64 bits antes del filtro. Toda porción aceptada por MIG debe
quedar por debajo de C; la porción inválida no puede generar ningún acceso
MIG. Las respuestas deben conservar ID, LAST, orden y número de beats.
El patrón C1 no puede aparecer en la dirección cero por truncamiento.

En C2H con error, reconstruir los TLP realmente publicados y los bytes que
permanecen intactos en el host. No exigir 128 bytes de éxito ni descartar
silenciosamente una publicación parcial. Cualquier byte publicado debe
corresponder a una respuesta AXI observada; comprobar también que no se
escribe fuera del destino. Mantener los estados DECERR, Busy y contador
correlacionados con las lecturas reales de registros, y drenar todas las
respuestas y lecturas periódicas antes del final.

El revisor deberá rechazar pérdida de la mitad inválida, dirección
truncada, escritura espuria en cero, estado DMA de éxito, respuesta o TLP
duplicado, y recuperación aparente sin un nuevo flanco Run. Esta prueba
no otorga atomicidad al descriptor: documentará qué prefijo llegó a DDR
antes del error y qué porción llegó efectivamente al host.

Las fuentes son `fpga/tb/ntrpu_pcie_ddr_boundary_{app,host}.sv`, las cabeceras
`boundary_{cases,host}.svh` y `boundary_tasks.vh`. El lanzador es
`tools/ntrpu_board_pcie_ddr_boundary.tcl`; el revisor independiente,
`tools/check_ntrpu_pcie_ddr_boundary.py`. El parche sobre seis archivos
originales AMD conserva los cambios únicamente en TX y `sample_tests.vh`.
Esa comprobación local no acredita compilación ni simulación.

El contrato inicial exige dos transacciones AXI alineadas de 64 bytes en
cada descriptor de cruce, con registros brutos anteriores a las aserciones
para diagnosticar una división distinta. Espera 5 AW/W/B y 8 AR/R antes
del filtro, frente a 4 AW/W/B y 7 AR/R en MIG. La simulación confirmó
exactamente esos recuentos. El revisor comprueba causalidad entre interfaces,
readback de Run, registro de error, cobertura exacta de cada byte C2H
y centinelas intactos; mantiene además la revisión periódica completa.

La cola local `ffn-supervised-queue-ddr-boundary-certified.py` espera el
PASS estricto y `VERIFIED.json` de `144107Z`, comprueba el hash del archivo
local, fija hashes de entradas y completa una copia
nueva de fuentes y lanza una única tanda aislada, cuatro hilos/3.600 s.
Después recopila, revisa, altera negativamente la evidencia y respalda.
Los dos observadores de dependencia anteriores se detuvieron sin lanzar
Vivado al encontrar los fallos del revisor; se conservan. Un fallo del
revisor de esta tanda requiere inspección; una pérdida del observador
nunca relanza el trabajo automáticamente.

La tanda `152800Z` utiliza `source-20261004T152750Z.zip`, SHA-256
`58d0cf80729748f891e905c06cd023e64de8d5cacc6667e9a6b34a2befad3d34`,
con 627 archivos. Conserva su propia copia de entradas y proyecto; el
observador `ffn-supervised-pcie-ddr-boundary-certified-queue` administra
recopilación, revisión y respaldo al terminar. No lanzar una segunda copia
si se pierde la conexión del observador.

### Resultado medido y consecuencia para el consumidor

Los once descriptores completan sus criterios; la revisión rechaza **193
alteraciones**. Se solicitaron 320 bytes H2C y se escribieron 256 en MIG:
el descriptor de 128 bytes que cruza el límite escribe C0 en los últimos
64 bytes válidos y rechaza C1 con DECERR. El centinela B en cero permanece
intacto. Las lecturas C2H válidas recuperan 384 bytes exactos.

El C2H de cruce entrega **64 bytes del prefijo válido al host**, conserva
los otros 64 bytes con el centinela `0xa5` y termina con `0x240`/contador
cero. El H2C de cruce termina con `0x4040`/contador cero. El trabajo con
error no es una salida válida, aunque parte del buffer haya cambiado:
publicar resultados o reutilizar una ventana exige finalización correcta,
identidad/época y drenaje. No hay atomicidad de descriptor ni rollback.

Los tres casos posteriores recuperan ambos motores sin reset. El cierre
observa 117 solicitudes/aceptaciones/CAS/retornos periódicos y siete
lecturas de usuario; termina a 155.023.498 ps, tras un flanco de drenaje.
El alcance es transporte frío con geometría genérica de 128 GiB: quedan
pendientes DIMM exacto, reset caliente, propiedad de buffers y hardware.

Respaldo verificado a las 15:55:15 UTC: 1.644 archivos, 280.862.653 bytes,
SHA-256 `9f65b38480a1d0044b6b1e82bfb8d16ee092f24436b5d65a12c4760411fbea00`.
Los 7.776 avisos del modelo se conservan y correlacionan con 54 ráfagas
periódicas no inicializadas; no se eliminan ni se interpretan como datos
válidos de usuario. Se revisan 124 READ y cuatro WRITE físicos.
