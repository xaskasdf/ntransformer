# Transporte del host hacia NTRD y dos MIG

## Composición y alcance

`fpga/rtl/ntrpu_pcie_fragment_dual_host_cold.sv` conecta los transportes AMD
AXI4 (512 bits, dirección de 64 bits, ID de 4 bits) y AXI-Lite (32 bits)
con una sola instancia de `ntrpu_pcie_fragment_dual_mig`. Ambos convierten
del reloj del host a `source_clk`, con tres etapas de sincronización.
`ntrpu_axil_cdc_recode` conserva los códigos de respuesta del control.
Los dos cruces hacia memoria siguen dentro del pool existente.

El contrato cubre **arranque frío coordinado**. La vida retenida del núcleo,
el reset operativo del pool y los resets del transporte son señales
distintas. Reiniciar un extremo del transporte durante tráfico requiere
otro contrato de invalidación y tratamiento de respuestas perdidas. La IRQ
permanece en el dominio fuente; todavía no hay confirmación MSI/MSI-X.

## Confirmación y drenaje

La aceptación no ocurre en el mismo orden en ambos tipos de conversor.
El FIFO AXI4 acepta en el productor antes de entregar al consumidor. En el
conversor AXI-Lite observado, el consumidor puede aceptar antes de que
regrese la confirmación al productor. Por eso el verificador distingue
primera presentación (`VALID`), aceptación (`VALID && READY`) y estabilidad
mientras se espera. Compara orden, cantidad y contenido en ambos extremos;
no interpreta una diferencia negativa entre aceptaciones como latencia
física negativa.

Las barreras de ARM, publicación y ACK requieren que todas las escrituras
de datos aceptadas hayan devuelto B al host y que no haya AW/W presentadas.
`transport_idle` local no prueba por sí solo drenaje del CDC ni de PCIe.
Estas señales de aislamiento siguen siendo entradas confiables del banco.

El verificador cruza además las observaciones de reset con los flancos de
liberación que programa el banco. La revisión inicial dejó pasar una
alteración del primer ciclo de reset del host; esa omisión del verificador
se corrigió y se conserva su versión anterior para auditoría. No cambió
el RTL ni la evidencia de simulación.

## Intento conservado y revisión R2

`20261005T053839Z-V7-ffn-supervised-managed-host-cold` queda **rechazado**:
el banco terminó tras consumir la última lectura en el host, antes de que
la confirmación llegara al extremo fuente. Se observaron diez respuestas
R aceptadas por el host y nueve por el núcleo. El PASS interno y el código
de salida cero no cerraban esa obligación.

Respaldo local/NFS: 704 archivos, 114.466.655 bytes; SHA-256
`3bd3f463d926279624fe2814ee92ff246fcfefb434cfeb57fb14ae41eeae03bf`.
El RTL se conserva sin cambios. El nuevo banco
`tb_ntrpu_pcie_managed_host_cold_r2.sv` espera inventarios de control
completos en ambos extremos y ocho ciclos fuente sin actividad antes de
terminar. Su ejecución es
`20261005T055601Z-V7-ffn-supervised-managed-host-cold-r2`;
la revisión independiente aprobó `PASS_MANAGED_HOST_COLD_CDC_DUAL_MIG`.
Respaldo local/NFS verificado: 716 archivos, 114.582.301 bytes; SHA-256
`73793b715a45fdc5be7adc3163196cb0ceab967fc5369ae736be5d9b4a78cbb9`.
El snapshot ejecutado es `source-20261005T055549Z.zip`, SHA-256
`db9afecabb88b3670321c0328508c7c45156b48c555c8bc909d393da8c1160f5`.
La revisión y sus verificadores finales viajan dentro del respaldo,
incluida la corrección posterior del verificador de reset.

## Evidencia R2

18 reservas, resultados y retiros; 35 tramas F01 con un rechazo esperado;
16 escrituras DDR (1.024 bytes) y 28 lecturas comparadas (1.792 bytes).
Los extremos host/fuente coinciden en 35 AW/W/B de datos, 276 AW/W/B
de control y diez AR/R. Se conservan SLVERR y DECERR, incluso después
del cruce. Las 54 barreras observadas incluyen el retorno B al host.

La traza contiene 12.447 muestras del reloj host de 7,3 ns y 18.173
del reloj fuente de 5 ns. El drenaje de control se declara a 90,820 us;
los ocho ciclos sin actividad terminan a 90,860 us, antes del cierre
a 90,865 us. Vivado empleó 597,874 segundos.

| Intervalo medido en simulación | Mínimo–máximo |
| --- | --- |
| Primera presentación AW de datos → aceptación destino | 42,519–47,019 ns |
| Primera presentación W de datos → aceptación destino | 42,519–47,319 ns |
| Primera presentación B de datos → aceptación host | 78,181–85,381 ns |
| Línea ensamblada → respuesta B de memoria | 88,498–619,229 ns |
| Lectura DDR AR → R | 115–655 ns |
| Línea ensamblada → retiro de reserva | 820–1.345 ns |

Son medidas del banco con pausas deliberadas. Los modelos FIFO no
representan retardos físicos de sincronización; estos intervalos no
cierran timing de placa ni demuestran caudal sostenido.

Se verificaron 35 archivos del modelo y 147 entradas compiladas. Los
32 avisos de herramienta coinciden con la referencia y los dos CDC
añadidos. Ambos canales ejercitan cuatro ranks; se completan 56/51
lecturas periódicas y 14 lecturas de usuario por canal. Los 1.296/720
avisos del modelo corresponden a nueve/cinco ráfagas periódicas sin
escritura previa, con multiplicidad por chip comprobada; no se suprimen.

El revisor rechaza 80 alteraciones y la evidencia incompleta del primer
intento. Reutiliza intacto el verificador del núcleo con dos MIG; solo
normaliza cargas desconocidas de peticiones inactivas para su entrada
numérica. No normaliza respuestas, señales de control ni datos válidos.

## Siguiente conexión PCIe

La conexión se aprobó y respaldó en `063837Z`; consultar
[la evidencia nativa](PCIE_DUAL_NATIVE_PLAN.md#tanda-nativa-revisada).
Los requisitos siguientes quedan como contrato de esa integración.

Seguir [el plan nativo](PCIE_DUAL_NATIVE_PLAN.md) con el ejemplo XDMA y el
productor de tramas de 512 bits ya comprobados. Conectar `m_axil_*` del
endpoint al control y `m_axib_*` al ingreso. La ventana AXI del ejemplo
comienza en `0x100000000`; el productor PCIe usa la base `0x900000000`.
Conservar la traducción observada y el destino DDR de 37 bits por canal.

El banco nativo deberá observar los IDs que genere XDMA y las respuestas
AXI a escrituras PCIe publicadas; estas no devuelven una finalización PCIe
de escritura al productor. Correlacionar TLP, AXI de ambos dominios,
reserva, cookie, memoria, resultado y retiro. Las lecturas directas por UI
siguen siendo testigos de memoria, sin acreditar lectura DDR por BAR.

El perfil actual usa dos tramas F01 de 64 bytes para transportar una línea
útil de 64 bytes: 50 % de carga útil antes de cabeceras PCIe. Es una cuenta
del formato, no una medición de caudal. La tanda de rendimiento deberá
medir las pausas y valorar tramas mayores conservando identidad y retiro.
