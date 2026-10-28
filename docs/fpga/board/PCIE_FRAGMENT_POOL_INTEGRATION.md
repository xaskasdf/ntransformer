# Integración entre fragmentos PCIe y multibuffer

Estado del 2026-10-05: **adaptador, pool/CDC y ambos MIG validados en simulación fría;
recolector con propiedad conjunta validado por separado en RTL/netlist**.
La [tanda nativa](PCIE_FRAGMENT_NATIVE.md) está calificada y respaldada. El experimento
[dual MIG](../PEER_RX_DUAL_MIG.md) conserva su evidencia independiente y su
[regresión física pendiente](../PEER_RX_DUAL_PHYSICAL.md). No modificar ni
relanzar esas tandas para preparar esta conexión.

## Adaptador aislado: evidencia disponible

`ntrpu_fragment_pool_bridge.sv` conserva una reserva y asigna un cookie
interno de 64 bits a cada línea aceptada. Compara canal y cookie de retorno,
retiene época/tag originales y exige drenaje antes de publicar DONE. STOP
impide nuevas reservas y deja avanzar transferencias ya expuestas. La pérdida
de disponibilidad queda enclavada: recuperar `admit` no rehabilita la sesión.

Tanda `20261005T022818Z-V7-ffn-supervised-fragment-pool-bridge`:

- 30 escenarios y otro de agotamiento de cookies, con semillas 1/7/42 en
  RTL y netlist sintetizada: 12 simulaciones positivas, siete mutantes RTL
  detectados y 100 alteraciones de evidencia rechazadas por un verificador
  independiente de los estados internos del DUT.
- En cada ensayo completo: 37 reservas, 35 líneas, 32 comandos/datos,
  38 retornos —incluidos los ajenos—, 30 DONE y 30 retiros. Los casos de
  pérdida de sesión conservan propiedad hasta el reset frío coordinado.
- 324 LUT, 881 registros, sin BRAM/URAM/DSP. Reloj OOC de 5 ns;
  WNS interno sin colocar de 2,507 ns. No acredita timing de interfaces:
  faltan retardos de 1.171 entradas y 951 salidas y estimación de skew.
- Trazas idénticas desde el ciclo 2; el primer muestreo precede al reset
  síncrono y difiere entre registros RTL desconocidos e INIT de primitivas.
  Todas las señales de transferencia están inactivas en ese muestreo.
- Respaldo local/NFS verificado: 408 archivos, 6.872.468 bytes, SHA-256
  `4d5bc7a8f76269597ef5461195d3406708eee73951278efa8ca00ced84d6bcd4`.

Los pares son drivers de protocolo: esta tanda **no contiene pool RTL,
CDC, MIG ni PCIe**. La prueba de agotamiento usa los dos últimos cookies;
el cruce de la palabra baja de 32 bits quedó calificado en la tanda
`20261005T024535Z-V7-ffn-supervised-fragment-pool-carry`: cookies
`ffffffff → 100000000 → 100000001`, seis simulaciones positivas,
un mutante de incremento truncado detectado y 100 alteraciones de evidencia
rechazadas. Respaldo verificado de 133 archivos / 2.296.027 bytes, SHA-256
`4ed96aa20df598d9b199b4f314a137ec1ab490c440470b02cb200ffd1dcc6cbd`.
La saturación del contador de retornos rechazados sigue sin probarse.

Al conectar el pool, mantener su STOP inactivo mientras exista una reserva:
el STOP del pool bloquea nuevos comandos y podría impedir aceptar un comando
que el puente ya tiene expuesto. Aplicar STOP al pool solo entre reservas;
el puente continúa bloqueando ARM inmediatamente. Un DONE ya expuesto no
cambia de contenido si se pierde disponibilidad: el consumidor debe invalidar
la sesión mediante `session_invalid`, incluso si llega a consumir ese DONE.

## Conexión al pool y CDC validada

`ntrpu_fragment_pool_transport.sv` compone el puente con
`ntrpu_peer_multibuffer_credits`, ocho bancos y cuatro créditos por canal.
Mantiene STOP del pool entre reservas. El banco tiene dos perfiles de relojes
asíncronos, destinos por encima de 4 GiB y cerca de 128 GiB, pausas AW/W/B,
errores de respuesta y pérdida de sesión. Usa los CDC AMD reales; memoria y
supervisor son drivers de protocolo. No incorpora el recolector PCIe ni MIG.

El intento `20261005T025506Z-V7-ffn-supervised-fragment-pool-transport`
falló antes de simular: Vivado interpretó el valor `-testplusarg` como opción
de `set_property`. Se conserva respaldado, SHA-256
`314447925b3cd74e5f82f7c6e83acbd9b3944e068841de5203a52f110d4f6d45`.

R2 usa `set_property -dict`, sin cambiar RTL ni banco:
`20261005T025843Z-V7-ffn-supervised-fragment-pool-transport-r2`,
`PASS_FRAGMENT_REAL_POOL_DUAL_CDC`, ejecución de 173,788 s. Cada perfil
verifica 13 reservas, 12 líneas, 11 comandos/datos, 11 escrituras AXI de
64 bytes, diez retornos, once DONE y doce retiros. Las diferencias son
intencionales: STOP parcial no escribe; una identidad incorrecta produce
error local; la pérdida de sesión retiene propiedad hasta reset frío.
Los errores de destino 2/3 se conservan en DONE. Tres variantes RTL
defectuosas —STOP directo, canal invertido y bit de datos alterado— fallan
donde corresponde. El verificador independiente rechaza 137 corrupciones.

| Perfil | Períodos origen / destino 0 / destino 1 | Línea → B | B → retorno al origen |
| --- | --- | --- | --- |
| 0 | 5 / 4 / 7,3 ns | 84,213–183,487 ns | 53,787–62,613 ns |
| 1 | 5 / 8,5 / 3,5 ns | 71,887–201,463 ns | 53,113–66,037 ns |

Intervalos medidos sobre los diez comandos completados por perfil, con
pausas deliberadas; no representan caudal sostenido. Los modelos FIFO
advierten que no modelan los retardos de sincronización asíncrona. No usar
estos intervalos como latencia física ni como aprobación de CDC en placa.
Respaldo local/NFS: 116 archivos, 18.361.833 bytes, SHA-256
`c79f33e56870ebd0e4d24a050a04ff45727f53cb6745db852bf6c2df8f8088dd`.

La tanda dual MIG descrita abajo sustituye los destinos de protocolo por
ambos controladores y su supervisor frío, con lecturas de retorno. Conserva
la revisión RTL original del experimento dual MIG. La integración del
recolector PCIe y el selector de canal en la ABI siguen pendientes.

## Propiedad ante el reset del pool

`ntrpu_fragment_dual_mig.sv` conserva los módulos anteriores y separa la
vida del puente de los resets operativos del pool. Un bit inicializado por
configuración habilita el puente al primer arranque y permanece activo
cuando el supervisor reinicia el pool. La reserva, época y cookie quedan
retenidas e invalidadas; la recuperación requiere reconfiguración completa.
`restart_required` cruza de gestión a origen mediante XPM con registro de
entrada y tres etapas. `admit` conserva su cierre local del supervisor.

Tanda `20261005T031705Z-V7-ffn-supervised-fragment-dual-lifetime` aprobada:
seis instancias del supervisor/pool/CDC real, una transferencia sana y cinco
pérdidas de calibración, reloj o reset externo durante reserva, escritura,
DONE retenido o espera de retiro. Los cinco fallos conservan propiedad tras
restaurar señales y solicitar rearme. El mutante que conecta directamente
el reset del pool al puente falla con `LOST_LEASE`; el revisor rechaza
123 corrupciones. No prueba reset caliente DDR ni reconfiguración física.
Respaldo local/NFS verificado: 120 archivos, 20.929.678 bytes, SHA-256
`e404311431aa478053197501063f073d7dcdc73903e0b653331a9b56524e0ef5`.

La tanda `20261005T032917Z-V7-ffn-supervised-fragment-dual-mig` pasó el
cotejo independiente: ambos MIG originales, 17 reservas, 15 escrituras y
28 lecturas de retorno, con los observadores periódicos intactos. La variante
de arranque ruteada del trabajo lateral ya tiene evidencia propia en
[PEER_RX_DUAL_STARTUP](../PEER_RX_DUAL_STARTUP.md); esta tanda conserva
el supervisor RTL original y no adopta ese patch ni acredita su ruteo.

## Resultado con ambos MIG

Estado `PASS_FRAGMENT_DUAL_MIG_COLD_READBACK_PERIODIC`: 363,567 s de
Vivado y 41.819.229 ps simulados. Se comparan 960 bytes escritos y
1.792 bytes leídos. Hay 15 retornos, 16 DONE y 17 retiros: STOP parcial
no emite línea; la identidad incorrecta produce error local sin escritura.
Se comprueban los cuatro ranks por canal, direcciones por encima de 4 GiB
y cerca de 128 GiB, además de conservar el contenido anterior tras rechazos.

| Canal | Escrituras / lecturas de 64 bytes | Línea → B | AR → R, incluidas pausas |
| --- | --- | --- | --- |
| 0 | 7 / 14 | 88,498–698,498 ns | 115–740 ns |
| 1 | 8 / 14 | 89,229–149,229 ns | 115–745 ns |

Mediciones funcionales de una reserva, con pausas deliberadas. No son
caudal sostenido ni latencias físicas; los modelos FIFO omiten retardos
de sincronización. Los 13 casos originales del supervisor pasan de nuevo.
El verificador rechaza 228 alteraciones de evidencia; compara la continuidad
de la ventana de 1.535 muestras de origen. No hay un marcador separado de
liberación de reset que autentique un prefijo inactivo omitido de esa ventana.

Los 35 archivos de modelo coinciden con el ejemplo respaldado salvo los
tres macros documentados de geometría/timing. La auditoría identifica 138
fuentes compiladas. Las 25 advertencias de herramienta coinciden con el
banco dual anterior. El canal 0 produce 720 avisos de lectura sin escritura,
atribuidos exactamente a cinco ráfagas periódicas; el canal 1 produce cero.
Los contadores periódicos drenan a 10/10 y 3/3, respectivamente. No se
modifican los modelos ni se suprime tráfico o advertencias.

Respaldo: 644 archivos, 76.114.427 bytes, SHA-256
`b07ea8dabb4ede75461eb506d325401bfb39da574b4ddb7cde5349a244f8477c`.
Esto completa esta conexión funcional fría; PCIe nativo, reset caliente,
DIMM comercial exacto y ruteo del nuevo conjunto siguen pendientes.

## Recolector y retiro conjunto: prueba de composición

`ntrpu_pcie_fragment_pool.sv` conecta los módulos conservados de decodificación,
recolección y puente. Ambos aceptan ARM en el mismo flanco. La interfaz ACK
con identidad completa consume RESULT y retira la reserva del puente
simultáneamente. Recomprueba barreras externas, sesión válida, drenaje y
ausencia de AW/W nuevos. Un RESULT visible sigue disponible para diagnóstico
tras perder la sesión, pero no permite ACK ni nueva reserva.

Tanda `20261005T040503Z-V7-ffn-supervised-fragment-ownership`:

- 11 casos por ejecución; semillas 1/7/42 en RTL y circuito sintetizado.
  En cada ejecución: 23 tramas, cuatro rechazos, 11 reservas, ocho líneas y
  ocho retiros. Siete transferencias reciben completion; la octava conserva
  la reserva invalidada tras perder disponibilidad, sin publicar éxito.
- STOP parcial y conflicto antes de LINE no consumen cookie. La cancelación
  local se limita a la reserva activa: error → ACK → nueva época funciona
  aunque `failed` del recolector permanezca retenido hasta ARM.
- ACK con época/tag incorrectos, barreras retiradas, AW simultáneo, W
  pendiente o B retenido no libera propiedad. Tres pérdidas de sesión
  cubren resultado visible, línea parcial y comando expuesto.
- Ocho mutantes RTL detectados y 184 alteraciones de evidencia rechazadas.
  Trazas iguales desde el ciclo 2; el primer muestreo precede al reset
  síncrono y difiere entre valores RTL desconocidos e INIT de primitivas.
- 1.272 LUT y 2.481 registros; sin BRAM/URAM/DSP. Síntesis OOC de 5 ns:
  WNS 1,985 ns y WHS 0,066 ns, sin colocación ni presupuesto de E/S
  (1.257 entradas y 2.011 salidas). No acredita timing físico.
- Duración de Vivado: 279,892 s. Línea → ACK: 32–66 ciclos de 5 ns en
  transferencias retiradas, con pausas deliberadas; no es caudal sostenido.

El destino es un par de protocolo, no el pool RTL ni MIG. El reset frío del
banco no demuestra reconfiguración física. El canal viene de un descriptor
confiable: todavía no se asigna una ABI MMIO nueva. Se conservaron las
advertencias de ámbito OOC y las diez de entorno `LIBRARY_PATH`; no hubo
advertencias de simulación positiva ni supresiones.

Respaldo local/NFS verificado: 381 archivos, 6.828.339 bytes, SHA-256
`e87d50c8f11f0ed6f574aab2eaf913bbd60e953c3436dc8f719c7ba9df9a97fd`.

## Control del host y próxima conexión al destino dual

La composición separada ya satisface ARM/ACK conjuntos ante pares de
protocolo. El [perfil NTRD](PCIE_DUAL_CONTROL.md) añade control MMIO y canal
explícito, validado en `043738Z` con seis simulaciones RTL/netlist.
Integrar ahora sus puertos
CMD/DATA/CPL con `ntrpu_dual_multibuffer_mig`, conservando una sola instancia
del puente. No encadenar esta composición con `ntrpu_fragment_dual_mig`
sin adaptar el top: ambos ya contienen un puente. Reutilizar el contrato
de vida fría probado en `031705Z` para recolector/control/puente y mantener
separados los resets operativos del pool. No fabricar `source_stopped`,
`ingress_isolated` ni `quiesce_epoch` a partir de contadores locales.

Conservar la regresión error local → ACK → nueva época al añadir control,
CDC y MIG. El ACK MMIO debe responder error si el retiro conjunto no está
listo o la sesión está invalidada; no ocultar el resultado diagnóstico.

El selector NTRD usa `0xb0`; `0xb4` expone el canal capturado y `0xb8`
el estado de sesión/reserva. `0xac` conserva `snapshot_valid`; no
reinterpretar el bit 37 de destino. El nuevo perfil identifica ID
`0x4452544e`, versión `0x10000` y capacidades `0xff`; el host debe
comprobar la tupla completa. NTRF y FFN conservan sus módulos anteriores.

## Fronteras existentes

El host de fragmentos admite una reserva de 64 bytes. Comprueba época y
tag completos antes de ensamblar dos mitades; su respuesta AXI B confirma
recepción, mientras que RESULT exige finalización del destino y barrera
externa. Actualmente usa `ntrpu_peer_rx_guard` hacia un MIG.

`ntrpu_dual_multibuffer_mig` admite comandos `{channel,addr,len,tag}`,
datos de 512 bits y completions por canal `{tag,status}`. Su interfaz no
transporta la época de la reserva PCIe. La admisión depende del supervisor
de arranque; las direcciones son locales a cada MIG, con comprobación de
rango antes de reducir a 37 bits.

## Contrato de la primera conexión

Conservar ambos diseños y usar la variante separada de una reserva y un
comando de un beat. Esta prueba del adaptador no demuestra aprovechamiento
concurrente del pool.

| Origen / destino | Obligación del adaptador |
| --- | --- |
| `line_valid/ready` → comando y datos | Capturar la línea una sola vez; mantener dirección, tag y datos mientras haya pausa; comprobar que el comando precede a sus datos |
| Época/tag de reserva → completion | Conservar ambos hasta el retiro; asignar un identificador interno de transporte a `cmd_tag` y cotejarlo junto al canal; una completion ajena no libera la reserva |
| `cpl_status` → DONE/RESULT | Definir una traducción explícita de estados; no convertir recepción del comando o último W en éxito de memoria |
| STOP | Bloquear nuevas reservas; drenar el comando ya aceptado y sus datos; conservar el resultado hasta ACK válido |
| `admit`, `restart_required`, resets | Mantener una sesión fría coordinada; una pérdida de disponibilidad invalida la sesión y exige recuperación explícita |

El identificador interno no se reutiliza dentro de la sesión, aunque el
host repita su tag en otra época. Su agotamiento debe impedir admisión,
sin wrap ni reinicio parcial del contador. La época no puede recuperarse
del retorno del pool: debe permanecer en el contexto del adaptador.

El canal debe proceder de un descriptor confiable registrado, no de bytes
sin validar de la trama. Su registro y descubrimiento requieren una
variante de ABI explícita. No asignar silenciosamente el bit 37 de una
dirección existente como selector de canal: cambiaría el contrato actual.

## Orden de pruebas

1. Banco pequeño del adaptador con pares de protocolo: pausas independientes
   en comando/datos/completion; STOP antes y después de aceptación; tag,
   canal o época obsoletos; error de destino y retiro retenido.
2. Conectar el pool RTL conservado; cotejar conservación de comandos,
   datos y completions, direcciones altas y ausencia de escrituras tras un
   rechazo. Mantener registros por tanda y referencia independiente.
3. Integrar ambos MIG con sus modelos originales y lecturas de retorno.
   Adoptar evidencia ruteada únicamente cuando su regresión y su arranque
   estén revisados; STA positivo por sí solo no basta.
4. Incorporar XDMA nativo y después varias reservas simultáneas. Antes de
   medir caudal, definir propiedad por ventana, agotamiento de identidades,
   orden por destino y retiro de errores. Medir bytes útiles, pausas,
   ocupación y créditos por canal bajo contención.

Cuatro canales, cómputo FFN y memoria entre tarjetas siguen siendo etapas
posteriores. Compartir aperturas BAR no establece coherencia ni propiedad.


## NTRD y pool: vida retenida — 2026-10-05

`20261005T045125Z-V7-ffn-supervised-managed-dual-lifetime`: aprobado y respaldado.
Ocho escenarios con supervisor/pool/dos CDC reales; memoria AXI simulada por
drivers. Siete pérdidas de disponibilidad conservan sesión y transacciones
MMIO/F01 pendientes; tres mutantes RTL y 259 corrupciones detectados.
Respaldo SHA-256 `d1df17a5c33a86224de5c7410d6c6a30e2a899f5ac640d07b8c113fdcc77cc9c`.
Véanse [contrato, evidencia y límites](PCIE_DUAL_CONTROL.md#vida-retenida-con-supervisor-y-pool-reales-2026-10-05).
La tanda posterior con dos MIG se registra por separado; V7 integrado y
el resto de la cola conservan sus criterios abiertos.


## NTRD con dos MIG — 2026-10-05

`20261005T050945Z-V7-ffn-supervised-managed-dual-mig`: aprobado y respaldado. 18 retiros MMIO, 35 tramas, 16 escrituras (1.024 bytes) y 28 lecturas (1.792 bytes) comparadas. Cuatro ranks por canal, direcciones altas, STOP parcial y durante W/B retenidas. 209 corrupciones detectadas; modelos/tráfico periódico auditados. SHA-256 `d555e2efcb6c27c2f7da0b87c529ca60021b306cc3a6b709fbd3ed10334c5bc5`. Véanse [evidencia y límites](PCIE_DUAL_CONTROL.md#ntrd-sobre-dos-mig-readback-completo-2026-10-05) y [siguiente integración PCIe](PCIE_DUAL_NATIVE_PLAN.md). Host AXI de prueba y memoria genérica; V7 completo y los demás criterios abiertos de la cola siguen pendientes.

## Transporte host CDC — 2026-10-05

`20261005T055601Z-V7-ffn-supervised-managed-host-cold-r2`: aprobado y
respaldado. Los dos CDC del host conservan 35 tramas, 276 escrituras y
diez lecturas de control; 18 trabajos terminan con drenaje completo.
El revisor rechaza 80 alteraciones. El primer banco queda conservado
como fallo por cierre anticipado de la última confirmación AXI-Lite.
Véanse [evidencia, latencias y contrato frío](PCIE_DUAL_HOST_CDC.md).
SHA-256 del respaldo:
`73793b715a45fdc5be7adc3163196cb0ceab967fc5369ae736be5d9b4a78cbb9`.
Sigue la conexión nativa BAR0/BAR2; no se cierra V7 completo ni la placa.
