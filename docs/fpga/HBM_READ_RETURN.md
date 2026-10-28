# Retorno sintetizable de lectura HBM

Revisión del 2026-10-07. Extiende el [receptor de escritura doble](PEER_RX_DUAL_HBM.md) dentro del [plan R0–R2](DUAL_HBM_INTEGRATION_PLAN.md). Las fuentes nuevas permanecen en `fpga/experimental/`; la cola y el RTL calificado del principal se conservan. **La revisión 3 pasó la regresión funcional, la comparación RTL/netlist, el modelo HBM nativo y el timing interno aislado a 200/450 MHz. El margen de memoria es de solo +0,004 ns; placa e interfaces externas siguen sin calificar.**

## Contrato y almacenamiento

`ntrpu_hbm_read_return.sv` contiene dos lectores independientes. Cada canal reserva un frame completo de **4 KiB** y admite un comando pendiente. `cmd_len` conserva la convención AXI de longitud menos uno: 0–63 representan 1–64 beats de 64 bytes. Dirección y tag tienen 64 bits; se comprueban alineación, fin del rango con 65 bits, límite de 16 GiB y cruce de 4 KiB antes de reducir la dirección a 34 bits.

Comandos, DATA, CPL, `stop` y `fault_in` pertenecen al dominio `clk`; el AXI nativo pertenece a `mem_clk`. El bloque de arranque sincroniza las indicaciones APB asíncronas. El integrador debe respetar estos dominios y el reset frío coordinado; no conectar un control asíncrono directamente a `stop` o `fault_in`.

La ruta es frame de 512 bits → CDC AMD de lectura → conversión 512/256 → AXI4/AXI3 → HBM. El adaptador comprueba los **seis bits de RID**, RLAST, longitud y RRESP antes de que las IP reduzcan IDs o combinen beats. Mantiene una dirección nativa ofrecida aunque aparezca un fallo. Los fragmentos nativos se serializan; esta versión no demuestra múltiples lecturas pendientes por canal.

La revisión 1 añade dos slots registrados de 265 bits por canal antes de clasificar la respuesta. `RREADY` depende de estado registrado; no de RID/RRESP/RLAST/RVALID/ARREADY entrantes. Una respuesta malformada capturada queda retenida y no avanza al conversor. También se eliminó esa dependencia en la frontera del frame. Esto responde al requisito de evitar caminos combinacionales entrada→salida en la misma interfaz AXI. [Arm IHI0022H, A3.1](https://developer.arm.com/-/media/Arm%20Developer%20Community/PDF/IHI0022H_amba_axi_protocol_spec.pdf).

La revisión 2 usa punteros independientes para que la contrapresión no controle las escrituras de payload. La revisión 3 añade otra cola de dos slots **después** de validar la respuesta; corta el camino combinacional validación→VALID→conversor→READY→contador. Cada canal tiene así cuatro slots nativos de 32 bytes, repartidos en dos etapas, además del frame de 4 KiB. Los datos ya validados conservan VALID ante un fallo posterior y la siguiente dirección espera a que ambas etapas estén vacías. `pending` incluye ambas colas.

```mermaid
flowchart LR
    H[Respuesta HBM de 256 bits] --> Q[Dos slots de captura]
    Q --> V[Comprobar RID, RLAST y protocolo]
    V --> R[Dos slots validados]
    V --> F[Cuarentena si hay fallo de protocolo]
    R --> A[Conversores AMD y CDC]
    A --> B[Frame de 4 KiB]
    B --> C[DATA y CPL del consumidor]
```

El diagrama muestra un canal y el sentido de las respuestas. Sus cuatro slots almacenan 1.060 bits de datos y metadatos; no son cuatro frames de 4 KiB ni reservas compartidas con el escritor. SLVERR/DECERR se propagan hasta el frame para completar con error y sin DATA.

El lector reúne la respuesta entera antes de publicar DATA. Así, un SLVERR/DECERR en el último fragmento no deja escapar un prefijo aparentemente válido. La finalización llega después de consumir DATA; conserva tag, estado y bytes bajo backpressure. Su retirada libera la reserva.

| Estado CPL | Significado |
| --- | --- |
| 0 | Se entregó el payload completo sin error RRESP |
| 1 | Comando inválido; sin AR ni DATA |
| 2 | Error de lectura; sin DATA, cero bytes entregados |

RID ajeno, RLAST incorrecto, EXOKAY inesperado o respuesta sin transacción provocan cuarentena. `stop` cierra admisiones y conserva el trabajo aceptado. El error queda retenido; `drained` significa ausencia de comandos retenidos, **no éxito ni autorización de rearme**. CPL=0 tampoco reemplaza la comprobación de época y salud del supervisor. Solo se admite reset frío/coordinado; no se implementa recuperación por timeout.

## Validación funcional

El banco escribe patrones por el receptor existente y los recupera por este RTL. Los **20 escenarios** pasan con reloj común y con relojes independientes, incluyendo memoria pausada, cambio de frecuencia, consumidores asimétricos, CPL retenida, stop, pérdida de inicialización, error externo, cinco comandos inválidos y cinco fallos de protocolo en cuarentena.

Cada ejecución completa de las revisiones 1, 2 y 3 contiene 46 lecturas, 41 escrituras de preparación, 145 AR nativos, 2.136 beats R capturados de 32 bytes y **936 beats entregados de 64 bytes**. Cuatro respuestas malformadas quedan en cuarentena. Un ledger independiente reconstruye memoria, fragmentos, errores y finalizaciones; rechaza 23 corrupciones de evidencia. Una perturbación entre flancos comprueba que RREADY no cambia al cambiar las entradas; el mutante que restablece la dependencia de RID se rechaza. La revisión 0 tenía 2.129 beats R y 22 controles, sin esa prueba estructural; no se considera sustituto de la nueva revisión.

Las latencias se midieron nuevamente: conservar los mismos bytes y completions no implica conservar el tiempo de servicio.

El perfil corto contiene cuatro escenarios, 14 lecturas/escrituras, 876 beats R y 438 beats entregados. `--native-profile 1` solo selecciona ese perfil: **no activa el modelo HBM nativo**.

## Implementación física aislada de la revisión 3

Parte `xcvu47p-fsvh2892-2-e`; origen 5 ns, memoria 2,222 ns; incertidumbres de setup/hold 0,100/0,050 ns. El checkpoint final conserva esos presupuestos y completa el ruteo:

| Dominio | Setup | Hold |
| --- | ---: | ---: |
| Origen | +0,919 ns | +0,014 ns |
| Memoria | +0,004 ns | +0,010 ns |

Ocupa **4.757 LUT, 8.046 registros y cero BRAM36**; 1.872 LUT son memoria distribuida, incluidas las IP. Las reservas de frame suman 8 KiB y las dos etapas nativas añaden 256 bytes de payload entre ambos canales, además de metadatos y almacenamiento interno de conversión/CDC. Cero BRAM no significa coste de memoria cero.

Hay 10.764 nets completamente ruteadas, cero errores de ruteo, cero latches/blackboxes y ningún endpoint interno sin restricción. Setup y hold no tienen endpoints incumplidos. CDC informa 11 CDC-3 y ocho CDC-9 informativos; las ocho restricciones de bus skew tienen margen mínimo +1,509 ns. Ancho de pulso mínimo +0,579 ns. La inspección estructural encuentra diez orígenes registrados para RREADY y ninguna entrada combinacional.

Las siete advertencias DRC permanecen, sin waivers: seis entradas LUT sin efecto según INIT y RTSTAT-10 con 1.343 nets. La clasificación cubre 1.241 puertos externos sin carga interna y 102 nets sin carga lógica; quedan cero objetos sin explicar. No es un informe DRC limpio.

El netlist final coincide con RTL en los 20 casos: 5.475 muestras de origen, 6.375 de memoria y doce comprobaciones de reset frío. Dos alteraciones del comparador —PAYLOAD y READY— se detectaron. Es regresión funcional acotada, sin SDF ni equivalencia formal.

La primera comparación de esta revisión detectó una carrera del banco: el contador de beats cambiaba la inyección de RID en el mismo flanco que propagaba el BUFG del netlist. El banco generado fija ahora ese cursor en el flanco descendente de memoria, para presentar metadatos estables a ambas implementaciones. Se conservaron la comparación ciclo a ciclo, todos los escenarios y el netlist original, sin modificar sus bytes. La tanda fallida también quedó archivada.

Este cierre excluye hard HBM, endpoint PCIe y placa. Quedan 815 entradas y 1.398 salidas sin retardos externos. **Los 4 ps de setup en memoria son un margen mínimo, no un presupuesto suficiente para declarar cerrado el sistema integrado.** Su colocación, relojes e interfaces deben volver a calificarse al integrar.

### Historial conservado

| Revisión | Cambio | Resultado y límite |
| --- | --- | --- |
| 0 | Primer retorno | Funcional/nativo positivo; tenía dependencia combinacional de RREADY y no es la implementación vigente |
| 1 | Dos slots antes de validar | Funcional/nativo positivo; setup inicial −0,173 ns. Explore llegó a −0,141 ns; AggressiveExplore dejó ruteo parcial y fue rechazado |
| 2 | Punteros independientes | Funcional y elaboración nativa positivos; setup −0,161 ns. No se ejecutó su tráfico nativo porque fue reemplazada |
| 3 | Segunda etapa tras validar | Setup inicial −0,054 ns; dos pasadas físicas conservaron el RTL y cerraron en +0,004 ns |

Las tandas fallidas o superadas siguen archivadas con sus fuentes, hashes y motivo. Ningún resultado de una revisión anterior se atribuye al RTL cambiado.

## Modelo HBM nativo

La revisión 3 completó elaboración en 488,024 s y tráfico en 949,387 s. Se usaron ocho CPUs permitidas y prioridad reducida, con copia privada del cache de IP. Las cifras excluyen la preparación/copia del cache.

Pasaron cuatro escenarios: 14 escrituras de preparación y 14 lecturas, 60 AR nativos, 876 beats nativos y 438 beats entregados, equivalentes a **28.032 bytes**. Ambos stacks anunciaron inicialización antes del arranque observado a 48,3902 μs. El ledger reconstruyó el tráfico y rechazó 21 corrupciones de evidencia. No se forzó la inicialización.

CATTRIP permanece X/X: no se califica estado térmico. La evidencia del modelo no incluye PCIe ni una placa física. El snapshot, sus fuentes y configuración tienen hashes propios y respaldo local verificado.

## Rendimiento y siguiente cambio

Esta referencia alterna recepción y publicación: necesita al menos N ciclos para recibir y N para entregar N beats en el dominio de 200 MHz. Su techo analítico es **menor de 6,4 GB/s por canal**, menos de 12,8 GB/s entre ambos, antes de latencias, fragmentación y pausas. Los 25,6 GB/s brutos de dos interfaces no son el caudal sostenido de este diseño.

El siguiente candidato de rendimiento necesita dos reservas por canal, cola de comandos y contabilidad por frame: llenar una mientras se entrega otra sin publicar frames incompletos. Debe probar errores tardíos, completions retenidas, agotamiento, orden y stop con ambas reservas ocupadas. El pool de escritura no resuelve automáticamente esta limitación del retorno. No escalar buffers antes de medir la ruta completa.

La revisión 3 nativa proporciona una referencia de servicio: dos lecturas simultáneas de 4 KiB tardaron 1.740 ns desde aceptación hasta CPL, equivalentes a 4,708 GB/s agregados **en esa ventana de simulación**. Cada lectura tuvo ocho fragmentos y separaciones entre beats de hasta 144,430 ns; la publicación de sus 64 beats ocupó 320 ns. No es una medida sostenida ni de hardware. El [analizador](../../tools/analyze_ntrpu_read_return.py) conserva esos límites y excluye la escritura de preparación.

Por ello hay dos experimentos distintos: aumentar fragmentos nativos pendientes (1/2/4/8) y después solapar dos frames por canal. Cada fragmento debe reservar su entrada de seguimiento antes de AR y retirarla solo con su respuesta completa. Usar inicialmente un ID común conserva orden; varios IDs requieren además reconstrucción del orden. La IP permite comportamientos distintos según IDs y configuración, que deben verificarse con el modelo nativo. [PG276, descripción del núcleo](https://docs.amd.com/r/en-US/pg276-axi-hbm/Core-Overview).

### Experimento R3 preparado

**Avance posterior del 2026-10-07:** la [prueba aislada de los conversores](HBM_READ_CONCURRENCY.md) pasó 28 escenarios y demostró hasta ocho fragmentos pendientes con las IP vigentes. El barrido de latencia muestra cuándo cuatro créditos bastan y cuándo ocho ayudan. Es una preprueba del barrido A; todavía no modifica ni califica la concurrencia del adaptador, las reservas de frame o el sistema HBM completo.

**Barrido A completado después, el mismo día:** el [candidato W1](HBM_READ_WINDOW.md) conserva ocho créditos tras comparar cuatro y ocho con HBM nativa. La ventana de dos lecturas de 4 KiB bajó de 1.740 a 920 ns; su timing interno aislado es +5 ps y el netlist pasó 20 casos. Sigue usando un frame por canal. Estas evidencias pertenecen a W1, compuesto en copias privadas, y no sustituyen los resultados ni fuentes de esta revisión 3.

**Barrido B completado después, el mismo día:** el [candidato F1](HBM_READ_FRAMES.md) conserva dos frames por canal, 16 KiB totales y ocho créditos. Una ventana de 64 KiB después de calentamiento mejoró de 8,445 a 14,711 GB/s con HBM nativa. Pasó la regresión funcional y de transacciones del netlist; el timing interno quedó en +1 ps y la comparación exacta por ciclo sigue abierta. Tampoco sustituye las fuentes calificadas de esta revisión 3 ni acredita PCIe o placa.

**Optimización física posterior:** el [estudio C1](HBM_READ_CUT.md) conserva el mismo RTL F1 con un nuevo checkpoint de +87 ps de setup y +7 ps de hold en memoria, frente a los +1/+2 ps iniciales. Mantiene recursos y ventana de rendimiento; el nuevo netlist y DRC tienen evidencia propia. Las colas adicionales probadas no fueron seleccionadas. El cierre sigue siendo interno, previo a la composición con PCIe y placa.

| Barrido | Capacidad de payload | Variable que se mide |
| --- | --- | --- |
| A: 1/2/4/8 fragmentos nativos pendientes | Un frame de 4 KiB por canal | Separación entre AR/R, latencia hasta primer dato, tiempo hasta CPL y ocupación del seguimiento |
| B: uno/dos frames por canal | 8/16 KiB totales | Solapamiento de llenado/publicación, caudal sostenido y espera por reserva |
| C: consumidores y enlaces asimétricos | La configuración ganadora de A/B | Aislamiento entre canales, saturación, equidad y retención ante fallos |

En A, todos los fragmentos pertenecen al único frame ya reservado: aumentar su concurrencia requiere seguimiento adicional, pero no otro frame de payload. El seguimiento debe distinguir dirección ofrecida, AR aceptada, respuesta capturada y respuesta retirada del buffer registrado. El crédito del fragmento vuelve solamente tras validar su último beat. Mantener el ID común y comprobar que las IP intermedias admiten la profundidad solicitada; un parámetro propio no cambia por sí solo sus límites.

Antes de comparar rendimiento, cada configuración debe pasar errores de RID/RLAST/RRESP con varios fragmentos pendientes, stop antes/después de AR, llenado de las dos entradas de respuesta, pausa de memoria, CPL retenida y agotamiento de reservas. Registrar máximo y trayectoria de ocupación, no solo bytes finales. Repetir timing, recursos y simulación nativa para cada candidato que se conserve.

La medición sostenida necesita una secuencia de comandos después de calentamiento y una ventana explícita, incluyendo huecos de admisión. Publicar por separado latencia de servicio, bytes útiles/tiempo, eficiencia sobre el límite de cada interfaz y recursos. Con dos frames, PCIe, el reparto HBM y el consumidor pueden convertirse en el límite siguiente; el pool de reserva se dimensionará con las pausas medidas.

Después siguen permisos/épocas P0, composición PCIe P1 y cómputo C0. El [contrato de lectura peer](PEER_READ_INTEGRATION.md) define la tabla de contexto, rechazos y retención hasta confirmación remota; no se acredita mediante esta prueba aislada.

## Evidencia y reproducción

Raíz local: `D:/AI/artifacts/ntrpu-read-return/20261007T103522780001Z/`. Las referencias `COMMON.json`, `ASYNC.json`, `SHORT.json`, `PHYSICAL.json`, `NETLIST.json`, `DRC.json`, `NATIVE_COMPILE.json` y `NATIVE_TRAFFIC.json` identifican las tandas. Cada una conserva fuentes, logs, hashes, revisión y respaldo incremental en `D:/AI/artifacts/ntrpu-content-store`; los ZIP de transferencia no se retienen.

```bash
python3 -B tools/check_ntrpu_read_return.py TANDA_ASYNC/output
python3 -B tools/check_ntrpu_read_return_physical.py TANDA_PHYSICAL/output
python3 -B tools/check_ntrpu_read_return_netlist.py TANDA_NETLIST/output
python3 -B tools/check_ntrpu_read_return_drc.py TANDA_DRC/output
python3 -B tools/check_ntrpu_read_return_native.py TANDA_NATIVE/output
```

La primera prueba se rechazó por una longitud incorrecta en el **banco de escritura**; un diagnóstico de 1 μs permitió corregirla. Otra tanda exploratoria antecede a la convención definitiva de `cmd_len`. El primer auditor DRC no arrancó por una dependencia ausente en su copia privada. Esas evidencias se conservan como diagnósticos, sin atribuirles calificación.
