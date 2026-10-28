# Inventario nativo DMA y BAR

## Contrato y montaje

El banco `fpga/tb/ntrpu_xdma_dma_bar_inventory_r1.sv` conecta dos instancias
del RTL `ntrpu_axi4_drain_monitor_r2` a los puertos públicos `m_axi_*`
y `m_axib_*` del XDMA generado por Vivado 2026.1. Observa las 48 conexiones
de cada instancia, incluidos IDs, direcciones, longitudes, strobes,
atributos y respuestas. El revisor comprueba el cableado del archivo
compilado y reconstruye las obligaciones desde handshakes reales.

Esta integración usa ID de 4 bits, datos de 512 bits, profundidad de
escritura 32 y profundidad de lectura 8 por ID. El tráfico del banco es
de un beat. Compartir parámetros no sustituye las pruebas de capacidad,
ráfagas y múltiples IDs de los bancos AXI4 anteriores.

## Secuencia exigida

1. Completar descubrimiento, configuración y una escritura/lectura BAR.
2. Emitir una segunda escritura BAR antes del cierre del host y retener B.
3. Publicar dos descriptores H2C adyacentes con cargas distintas, iniciar
   Run y transferir el control al secuenciador W1C/CSR real.
4. Liberar por separado las dos B DMA, conservando la obligación BAR.
5. Observar el snapshot local con ambos descriptores completados mientras
   el inventario BAR aún contiene una AW y una W pendientes.
6. Liberar B BAR y comprobar que solo esa respuesta retira su obligación.
7. Retirar el cierre del host y comprobar revocación retenida.

`source_fenced` permanece en cero: vaciar inventarios públicos no demuestra
que no queden solicitudes en el host, PCIe o colas protegidas. Ningún
monitor puede emitir ACK. No existe autorización de reset global en este
banco; se conserva el Completion Timeout nativo de 50 ms.

## Estado de validación

El primer intento, `20261006T215941Z`, terminó sin simulación: XSIM rechazó
dos referencias jerárquicas a nets `RLAST` implícitos dentro de conexiones
de instancia. Se verificaron las declaraciones explícitas de los mismos
puertos de entrada XDMA y su conexión con esos nets. El diagnóstico está
respaldado en D:/NFS; 345,075 s de ejecución, 77 entradas compiladas y
129 archivos AMD conservados. R0 usaba profundidad de escritura 8.

R1 conecta los puertos explícitos y usa profundidad 32, sin alterar RTL de
producción ni código AMD. Tanda revisada:
`20261006T220923Z-V4-ffn-supervised-xdma-dma-bar-inventory-r1-frozen`.
Fuentes congeladas: `source-20261006T220841Z.zip`. Pasó la revisión
independiente y está respaldada en D:/NFS.

| Medida simulada | Resultado |
| --- | --- |
| Tráfico H2C | Dos descriptores, 128 bytes habilitados |
| Tráfico BAR | Dos escrituras de 4 bytes y una lectura de 4 bytes |
| Gestión local | 35 lecturas, 9 escrituras; 26 operaciones autónomas |
| Conteos H2C por ronda CSR | 0 → 1 → 1 → 2 |
| Inicio del secuenciador → snapshot | 1,448 µs |
| Retención controlada de la segunda B BAR | 2,732 µs |
| Snapshot local → aceptación de B BAR | **36 ns** |
| Máximo observado DMA AW/W pendientes | 2/2 |
| Máximo observado BAR AW/W/AR pendientes | 1/1/1 |
| ACK de monitores sin cierre del productor | Cero durante todo el ensayo |

El snapshot aparece a 107,040 µs; B BAR se acepta a 107,076 µs.
El inventario BAR conserva una AW y una W durante todo ese intervalo.
Ambos Done H2C siguen a sus B reales. La retirada posterior del cierre
conserva fallos `0x200c/0x100` en secuenciador y ledger.

La auditoría rechaza **65 corrupciones de evidencia**, incluidas pérdida
de inventario durante el snapshot, tag incorrecto, ACK inventado y payload
BAR alterado. Comprueba 77 entradas compiladas, 129 archivos AMD intactos
y los 139 avisos asociados a sus declaraciones originales. Ejecución:
612,763 s. No se probó C2H en esta composición ni se midió inferencia.

Revisión: `20261006T220923Z-V4-ffn-supervised-xdma-dma-bar-inventory-r1-frozen-independent-dma-bar-inventory-review.json`
en `build/cloud-setup/20261002-board/`; SHA-256
`f33aa110ed11d9e47d0ef83b31b482f0d1f130457c7f606e6a47a13c4e9e8f18`.
Archivo de resultados: SHA-256
`ad72275cc292703d8fb5e11f50387f310353a210c1071886e8efd5f0f89b1e45`.

## Composición bidireccional y recepción host

La tanda `20261006T224150Z-V4-ffn-supervised-xdma-dma-bar-c2h-r1-frozen`
pasó la revisión independiente con el mismo perfil de 32 escrituras y
ocho lecturas por ID. Integra un descriptor H2C y otro C2H, el maestro
local, secuenciador, ledger y los dos monitores AXI4 de producción.

| Medida simulada | Resultado |
| --- | --- |
| H2C / C2H / recepción host verificada | 64 / 64 / 64 bytes |
| BAR | Dos escrituras de 4 bytes y una lectura de 4 bytes |
| Gestión local | 29 lecturas, 13 escrituras; 20 operaciones autónomas |
| Rondas CSR H2C/C2H | 0/0 → 1/0 → 1/1 |
| Inicio del secuenciador → snapshot | 1,108 µs |
| Retención H2C B / C2H R / segunda BAR B | 1,196 / 0,852 / 3,144 µs |
| Ledger completo → recepción host completa | **103,999 ns** |
| Snapshot local → aceptación de B BAR | **36 ns** |
| Pico DMA y BAR AW/W/AR | 1/1/1 en cada bus |
| ACK sin cierre del productor | Cero |

C2H R se acepta a 107,188 µs; Done aparece a 107,336 µs y el ledger
registra completado a 107,336001 µs. Los últimos bytes llegan al host a
107,440 µs. El snapshot CSR ocurre a 107,452 µs, 12 ns después de esa
recepción; BAR sigue pendiente hasta 107,488 µs. Cada obligación se
reconstruye desde su propio handshake. Retirar el cierre conserva fallos
`0x200c/0x100`; no se suministra `source_fenced` ni se autoriza reset.

La revisión rechaza **65 corrupciones**, incluyendo pérdida del inventario
C2H, payload de lectura incorrecto, recepción host anticipada y pérdida
de BAR durante el snapshot. Verifica 77 entradas compiladas, 129 archivos
AMD intactos y 139 avisos mapeados. Ejecución: 586,838 s.

Revisión: `20261006T224150Z-V4-ffn-supervised-xdma-dma-bar-c2h-r1-frozen-independent-dma-bar-c2h-review.json`,
SHA-256 `2c2a31df3c03fca4ce779346130d8560ec3286faffb19a630e93eae8e8ced922`.
Resultados: `1d636e9bcbbb9ba8e3c7d03a8fbdd8505b9136699235e2d02a0dfcabdaac3915`.
Fuentes: `source-20261006T224118Z.zip`, SHA-256
`8781225330d60a1474b1023d486e84133ac880d42f39d254bb71ad21c5e832e9`.

El intento previo `20261006T223045Z` compiló, pero se detuvo a tiempo cero
porque la captura pedía `ledger_first_image/ledger_second_image` de la
cadena anterior. R1 cambia solo esos nombres a
`ledger_h2c_image/ledger_c2h_image`; los 247 objetos se comprobaron antes
de relanzar. Diagnóstico, fuentes y ambas tandas están respaldados D:/NFS.

## Integración restante

Implementar los [inventarios PCIe del perfil generado](XDMA_PCIE_INVENTORY_PLAN.md),
resolver su conexión de producción y un contrato real de cierre del
productor. Solo entonces transferir identidad y revocación al
join coherente de siete contratos. Permanecen vigentes DIMM exacto,
cuatro DDR/512 GiB, inferencia con pesos reales y cierre físico.

Referencias del proyecto: [secuenciador](XDMA_STOP_REGISTERS.md),
[proveedor completo](XDMA_INVENTORY_PROVIDER_PLAN.md) y
[cola de requisitos](VIVADO_QUEUE.md).
