# Programador de ventanas y cliente del puente

Estado: composicion RTL local validada en 11 casos y cuatro semillas.
El cliente y la IP AMD candidata tambien pasaron la simulacion nativa privada:
enlace Gen3 x16, reloj de 250 MHz, dos traducciones y 192 bytes recibidos.
El log y el VCD coinciden en todos los handshakes revisados.
El escritor anterior permanece intacto y conserva su evidencia independiente.

## Perfil provisional de registros

Objetivo: DMA/Bridge Subsystem en modo AXI Bridge, endpoint Gen3, una apertura
AXI de 4 KiB generada en `0x10000000`, traduccion PCIe de 64 bits y tabla
de control con bit 28 a cero. Esa apertura es constante de la IP; el gestor
rechaza intentar cambiarla como si fuera un registro de traduccion.

Secuencia de seis transacciones, sin trafico DMA durante la programacion:

| Paso | Operacion | Comprobacion |
| --- | --- | --- |
| 0 | Leer `0x130` | Gen3 y endpoint: mascara `0xA`, esperado `0x8` |
| 1 | Leer `0x204` | Longitud/revision VSEC: mascara `0xFFFF0000`, esperado `0x03800000` |
| 2 | Escribir `0x208` | Bits 63:32 del destino |
| 3 | Escribir `0x20C` | Bits 31:0 del destino |
| 4 y 5 | Releer `0x208/0x20C` | Coincidencia completa de ambas palabras |

Se usan la [tabla de traduccion](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/AXI-Base-Address-Translation-Configuration-Registers-Offset-0x208-0x234),
la [seleccion de tabla](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Memory-Map-Changes)
y la [cabecera VSEC](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/VSEC-Header-Register-2-Offset-0x204).
La pagina general de [Address Translation](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/Address-Translation)
menciona `cpm4-bridge-v2-1-register.csv` y `0xEE0`; no se traslado ese mapa CPM
a este candidato. La instancia XDMA 4.2 rev. 4 de esta tanda confirma
`0x208/0x20C` en simulacion; otras variantes y el hardware requieren su propia
comprobacion. Leer dos identificadores parciales no
identifica de manera unica una version de IP ni sustituye dicha comprobacion.

## Implementacion

`ntrpu_axi_bridge_mapper.sv` reutiliza sin modificaciones el maestro local
`fpga/rtl/ntrpu_xdma_local_master.sv`. Conserva AW/W independientes, ofertas
estables, identidad de respuesta y respuesta AXI real. El timeout se retiene
y bloquea admision; nunca fabrica B/R ni termina prematuramente un mapping.
Tras una respuesta tardia devuelve fallo y conserva el bloqueo hasta un
arranque coordinado. El timeout mide ciclos de ese reloj, no ausencia de reloj.

`ntrpu_axi_bridge_endpoint_client.sv` conecta el mapper al escritor y compara
sesion, generacion y ambas bases en la respuesta. Los fallos se propagan y
una reconfiguracion espera al retiro del contexto. Las mitades del destino
se escriben por separado: su consistencia depende de la ausencia de otros
emisores y de bloquear DMA durante toda la secuencia. No hay rollback de una
programacion parcial. Una imagen parcialmente escrita queda inutilizable.

## Frontera para la instancia nativa

| Cliente | Conexion prevista al DMA/Bridge |
| --- | --- |
| `clk` | Reloj AXI de la IP |
| `link_up` | Enlace disponible y reset AXI nativo liberado |
| `m_axil_*` | `s_axil_*`, conexion privada, unico maestro |
| `m_aw*`, `m_w*`, `m_b*` | Canales de escritura `s_axib_*` |
| Lecturas `s_axib_ar*` | Inactivas en este perfil de escritura |
| `s_axib_awregion` y atributos reservados | Constantes segun el wrapper generado |

La auditoria de la instancia generada comprobo 35 conexiones por nombre,
direccion y ancho, usando `AXI_ADDR_BITS=64` en el cliente. `s_axib_awregion`
debe ser cero para esta unica ventana; el canal AR queda inactivo, con sus
campos definidos. El perfil sin paridad no expone `s_axib_wuser` ni atributos
AWPROT/AWCACHE en ese puerto. AXI-Lite si expone AWPROT/ARPROT.
`patch_native_example.py` conecta esas instancias en una copia privada del
ejemplo AMD; conserva su BRAM como destino del trafico entrante. El BAR de
aplicacion no debe alcanzar `s_axil_*`. No conectar el reset PCIe directamente
a `cold_rst_n`. Falta el supervisor independiente y el manejo coordinado de
reset. La documentacion de [puertos AXI](https://docs.amd.com/r/en-US/pg194-axi-bridge-pcie-gen3/AXI-Slave-Interface)
es una referencia; este cliente no instancia el endpoint AMD. La generacion
tambien expone `m_axib_*` para peticiones entrantes: necesitan un destino
con respuestas AXI definidas, sin acceso al control privado. Al determinar
disponibilidad considerar `user_lnk_up`, `axi_aresetn` y `axi_ctl_aresetn`.

## Prueba local y limites

```powershell
python -B fpga/experimental/axi_bridge_dma/run_mapping_local.py NUEVA_SALIDA --toolchain TOOLCHAIN --seeds 4
```

El banco integra mapper, maestro y escritor reales con un destino de registros
modelado. Prueba remap y dos transferencias sin reset, errores B/R, ambas
palabras de readback, perfil incorrecto, apertura incorrecta, timeout de
escritura/lectura con respuesta tardia y perdida de enlace. El revisor
reconstruye transacciones de control y bytes de datos desde puertos publicos.
Las direcciones PCIe del banco son traducciones calculadas por el modelo;
no son TLP recibidos por un Root Port. ACK, lector y recepcion son estimulos.
La prueba local no valida host adverso, IP nativa, DDR4/512 GiB ni timing fisico.

## Evidencia de esta tanda

En `D:/AI/artifacts/ntrpu-axi-bridge-mapping/`:

- `20261009-local-r1`: 44 ejecuciones de escenarios, 96 lecturas y 44
  escrituras de control, 768 bytes; tres mutantes RTL y cuatro corrupciones
  de evidencia rechazados. `r0` conserva el fallo previo de cobertura del
  estimulo AXI-Lite, corregido en el banco para `r1`.
- `native-products-20261009T160818Z.tar.gz`: IP XDMA 4.2 rev. 4 generada en
  Vivado 2026.1 para `xcvu47p-fsvh2892-2-e`; 166 archivos restaurados y
  verificados por SHA-256. Generar productos no ejecuta sintesis ni routing.
- `20261009-native-interface-review.json`: conexiones y constantes contrastadas
  contra el wrapper, incluida apertura y traduccion de 64 bits. No demuestra
  que los registros `0x208/0x20C` se comporten como el modelo local.

La propiedad `axisten_freq` solicitada y recuperada es 250 MHz; el wrapper
declara 125 MHz en los metadatos de `axi_aclk`. El VCD de la tanda nativa r3
muestra 4.485 periodos consecutivos de 4.000 ps, con enlace arriba y ambos
resets AXI liberados: **250 MHz en esta simulacion**. Evidencia:
`D:/AI/artifacts/ntrpu-axi-bridge-native/20261009-native-clock-review.json`.
Esto no sustituye `report_clocks` ni cierre de timing sobre la implementacion.

## Banco nativo privado

`prepare_native_example.tcl` importa la XCI en un proyecto nuevo y genera
el ejemplo AMD con Root Port. No mover una XCI y abrirla con `read_ip`:
conserva rutas relativas de generacion. La primera tentativa fue detenida
por ese motivo; la semilla permanecio intacta.

`run_native_example.tcl` copia el ejemplo, conecta las 35 senales, compila
y captura los puertos publicos. `ntrpu_axi_bridge_native.sv` prueba dos mapas
y payloads de 64/128 bytes sin reset entre ellos; despues aplica PERSTn con
un tercer comando pendiente del lector. El lector y el ACK son estimulos.
No cubre recuperacion de DMA ya emitido ni exclusion ante un host adverso.

`run_compiled_native.tcl` permite ejecutar una copia del snapshot `board_behav`
sin repetir la elaboracion; requiere `NTRPU_NATIVE_OUT` con salida nueva.
El revisor acepta solo una traza completa y reconstruye CSR, bytes CQ y
causalidad, contrastando los handshakes con el VCD:

```powershell
python -B fpga/experimental/axi_bridge_dma/review_native.py LOG NUEVO_RESULTADO.json --vcd NATIVE.vcd
```

## Resultado nativo y limites restantes

Tanda `native-r4-20261009T164601Z`, en
`D:/AI/artifacts/ntrpu-axi-bridge-native/`: **NATIVE_BRIDGE_MAPPING_AND_DATA_PASS**.
El snapshot compilado se ejecuto en 241,6 segundos, con dos nucleos y
prioridad reducida; la semilla compilada permanecio intacta. Las tandas
anteriores conservan los errores de preparacion y los limites de tiempo.

- Ocho lecturas y cuatro escrituras de control; identificadores observados
  `0x130=0x0000002C` y `0x204=0x03800002`, compatibles con las mascaras.
- Readback exacto de ambas mitades para `0x1234567800001000` y
  `0x9ABCDEF000002000`; el Root Port recibe respectivamente 64 y 128 bytes.
- Dos paquetes PCIe, cinco beats CQ, tres beats AXI de datos. Sin reset entre
  mapas; contexto retenido hasta ACK atribuido y consumo del resultado.
- PERSTn conserva el tercer contexto pendiente del lector y bloquea
  admision. No demuestra recuperacion de TLPs ya emitidos ni falta de reloj.
- Cinco corrupciones deliberadas de evidencia fueron rechazadas: destino,
  readback, periodo, respuesta B ausente y datos VCD incongruentes con el log.

Los avisos internos de AMD permanecen en los logs; este resultado no certifica
ausencia de avisos ni todas las rutas de la IP. El banco mantiene lector y
ACK modelados. Pendientes: exclusion del control frente al host, supervisor
independiente de reloj/reset, lector de memoria nativo, recuperacion durante
DMA, sintesis/timing y placa fisica. No integrar aun con el top o cola principal.
