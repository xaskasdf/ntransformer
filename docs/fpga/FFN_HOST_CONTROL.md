# Control del FFN desde el host: ABI experimental 1.0

Este documento describe `EXTERNAL_RETIRE=0`, modo predeterminado que siguen
usando los wrappers FFN existentes. La [ABI 2.0 opcional](FFN_IRQ_CONTROL.md)
añade identidad de ACK y conserva la finalización hasta retiro del
transporte IRQ; tiene banco propio y todavía no sustituye esa integración.

`fpga/rtl/ntrpu_ffn_control.sv` proporciona registros AXI-Lite de 32 bits
para un motor `ntrpu_ffn_memory`, en el mismo dominio de reloj/reset.
Hay un descriptor pendiente o en ejecución y una finalización retenida.
El banco unitario usa un **par de cómputo simulado a nivel de protocolo**;
no demuestra PCIe, DMA, DDR, inferencia ni integración con HLS.
La [tanda de integración](FFN_HOST_INTEGRATION.md) conecta por separado este
control con la FFN/HLS real y conserva el alcance de estas pruebas unitarias.

## Contrato de acceso

Registros little-endian, alineados a cuatro bytes. El decodificador comprueba
los 32 bits de dirección: una dirección alta no puede convertirse en alias
del bloque bajo. Huecos, direcciones no alineadas y offsets desconocidos
responden `DECERR`. Escribir registros de solo lectura, bits reservados o un
comando incompatible con el estado responde `SLVERR` sin modificar el estado.
`WSTRB` selecciona bytes; una escritura con máscara cero no cambia datos.

AW y W pueden llegar en cualquier orden. El bloque conserva B hasta BREADY y
R hasta RREADY, sin aceptar otra transacción del mismo tipo mientras la
respuesta está retenida. Las lecturas capturan una instantánea al aceptar AR;
una escritura concurrente no cambia una respuesta R pendiente. Para observar
una escritura, el host debe completar B antes de emitir la lectura dependiente.

## Mapa de registros

| Offset | Acceso | Contenido |
| --- | --- | --- |
| `0x00` | R | Identificador `0x4652544e` (`NTRF` en memoria) |
| `0x04` | R | Versión `0x00010000` |
| `0x08` | R | Bits 0: salida FP32; 1: cancelación; 2: IRQ de nivel |
| `0x0c` | R | Bits 0: descriptor pendiente; 1: ejecutando; 2: finalización pendiente; 3: cancelación esperando aceptación; 4: cancelación solicitada |
| `0x10` | W | `0`: no-op; `1`: START; `2`: CANCEL; `4`: ACK. No combinar bits; leer devuelve cero |
| `0x14` | RW | Bit 0: habilitar IRQ |
| `0x18/1c` | R | Contadores de trabajos aceptados/finalizados, módulo 2³² |
| `0x20/24/28` | RW | Secuencia, época y época esperada, 32 bits cada una |
| `0x2c` | RW | Dimensión `[13:0]`, dimensión oculta `[29:16]` |
| `0x30` | RW | Layout `[1:0]`; el motor exige layout 2 |
| `0x34` | RW | Tiles K de gate/up/down en `[5:0]`, `[13:8]`, `[21:16]` |
| `0x40` | R | Estado `[3:0]`, estado hijo `[7:4]`, etapas `[10:8]`, fase `[12:11]` |
| `0x44/48` | R | Bytes de salida / bytes de etapa notificados por el motor |
| `0x4c/50/54` | R | Secuencia / época / estado HLS devueltos |
| `0x58/5c` | R | Ciclos desde aceptación hasta finalización, bajo/alto, módulo 2⁶⁴ |
| `0x60` | R | Diagnóstico: bit 0 secuencia incorrecta; 1 época incorrecta; 2 metadatos inválidos |

Ocho buffers ocupan `0x100 + índice * 0x20`. Sus campos son dirección
baja/alta (`+0/+4`), base baja/alta (`+8/+0xc`), tamaño bajo/alto
(`+0x10/+0x14`) y época (`+0x18`). `+0x1c` está reservado.
Orden: entrada ABI1, Wgate, Wup, Wdown, gate binary64, up binary64,
SwiGLU ABI1 y salida FP32 (binary64 si `OUTPUT_FP32=0`).

## Ciclo de un trabajo

1. Registrar y reservar buffers privados en el futuro runtime; cargar pesos y
   entrada, completar sus transferencias y escribir todos los campos sombra.
2. Emitir START. Su aceptación captura todos los campos de forma atómica;
   escribir después las sombras no altera el descriptor activo. START falla
   mientras haya descriptor, ejecución o finalización sin confirmar.
3. Consultar estado o esperar IRQ. CANCEL se admite una vez por trabajo,
   incluso antes de que el motor acepte el descriptor. No retira el descriptor:
   espera su aceptación y mantiene la cancelación hasta que el motor la acepta.
   Es una solicitud, no prueba de cancelación consumada; esperar la finalización.
4. Leer toda la finalización. Publicar la salida únicamente con estado cero y
   diagnóstico cero. Errores, cancelaciones y metadatos inválidos conservan los
   resultados privados. La IRQ permanece activa hasta ACK si está habilitada;
   leer registros no la limpia. ACK libera el espacio para el siguiente trabajo.

Los estados válidos del secuenciador son 0, 1, 2, 5, 6, 7 y 8 (`TIMEOUT`).
El diagnóstico comprueba identidad y, para éxito, cuatro etapas, fase final,
`dim*4` bytes (`dim*8` en binary64), estado hijo cero y estado HLS cero.
El motor sigue siendo responsable de geometría, rangos, épocas, solapamientos
y validación numérica. Los campos no se reinterpretan como punteros del host.
El bit «cancelación solicitada» permanece hasta el siguiente START o reset;
no indica que la solicitud haya sido aceptada por el motor.

## Evidencia funcional

Vivado/XSIM 2026.1, VU5P `xcvu5p-flva2104-2-i`:

| Tanda | Resultado |
| --- | --- |
| `20261003T120619Z-V7-ffn-csr-unit` | Rechazada antes de simular: argumento Tcl interpretado como opción. Fuentes y fallo conservados |
| `20261003T120823Z-V7-ffn-csr-unit-r2` | PASS FP32, tres semillas; 49,058 s de pared |
| `20261003T121223Z-V7-ffn-csr-fp64-unit` | PASS binary64, tres semillas; 49,112 s de pared |
| `20261003T121223Z-V7-ffn-csr-synth` | Síntesis OOC FP32, 208,932 s; 1.729 LUT, 4.372 FF, cero BRAM/DSP/latches/cajas negras |

Cada formato verifica 27 grupos de cobertura, 894 lecturas, 1.236 escrituras,
60 aceptaciones de trabajo, 57 finalizaciones, tres cancelaciones aceptadas y
45 resets. Las semillas cambian datos; las pausas y el orden AW/W recorren una
agenda determinista. Cada semilla ejercita 88 escrituras con AW primero,
141 con W primero, 183 simultáneas y 832/826 ciclos de retención B/R.
Son medidas del banco de protocolo, no caudal PCIe ni latencia FFN.

`tools/check_ntrpu_ffn_control.py` comprueba integridad del recorrido,
lecturas, cobertura, contadores y fallos de finalización. La revisión rechaza
siete alteraciones deliberadas del log. Ambos bancos tienen respaldo NFS y
local con hash de fuentes, manifiesto de archivos y `VERIFIED.json`.
Único aviso de simulación: XSIM 43-3431 sobre el entorno del compilador.

La síntesis tiene WNS +2,280 ns, WHS +0,066 ns y margen de pulso +2,225 ns
con reloj de 5 ns. El peor camino es dirección de escritura → enable de
configuración. Son estimaciones previas a colocación/ruteo: `HD.CLK_SRC` falta
y no hay retardos de 253 entradas y 1.981 salidas. No hay endpoints internos
sin restricción ni registros sin reloj. DRC reporta cero violaciones en su
alcance OOC; metodología lista 1.000 avisos TIMING-18 de fronteras.
Los seis bits AWPROT/ARPROT sin uso y el aviso de netlist plana se conservan.
Checkpoint sintetizado: SHA-256
`ffd8585542c5efb974dcbed0d9c655be7af3c892d42dad3021bdfa09c7389a94`.

## Fronteras pendientes

Esta ABI de registros **no es el formato binario de NTRPU Fabric**: Fabric usa
identidades de 64 bits y este motor usa 32. Un adaptador debe rechazar valores
no representables; nunca truncarlos silenciosamente. Direcciones, bases y
tamaños de buffer conservan los 64 bits.

El dispositivo debe exponerse mediante un mapeo privilegiado. AWPROT/ARPROT no
constituyen autorización; faltan driver, registro de buffers y traducción de
direcciones. No hay watchdog interno ni soft reset que abandone tráfico AXI.
El reset externo debe coordinar control, motor y memoria, invalidar trabajos y
renovar épocas desde el host. No conectar directamente dominios PCIe/DDR distintos:
faltan puentes CDC, arbitraje, interconexión AXI y política de recuperación.

## Reproducción

```bash
vivado -mode batch -source tools/ntrpu_board_ffn_control.tcl \
  -tclargs sim build/CSR_RUN_NUEVO
# sim_fp64 verifica el formato alternativo; synth exporta síntesis OOC a 200 MHz.
```

El banco recorre semillas 1, 7 y 42, todas las sombras, máscaras de bytes,
AW/W independientes, respuestas retenidas, descriptor inmutable, cancelación,
IRQ, errores de finalización y siete escenarios de reset. Los resultados y
respaldos se registran en [VIVADO_STATUS](board/VIVADO_STATUS.md).
