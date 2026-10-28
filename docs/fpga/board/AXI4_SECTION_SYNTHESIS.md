# Síntesis de la sección AXI4 retenida

La [medición posterior del candidato RTL R2](AXI4_SECTION_SYNTHESIS_R2.md)
ya está revisada; este documento conserva la síntesis de los módulos originales.

## Revisión posterior del índice inicial

La ampliación con fallos antes/durante el certificado encontró un índice
indeterminado al arrancar en la simulación RTL original. Se conserva la
evidencia previa con su alcance; no prueba todas las secuencias iniciales.
La revisión candidata RTL R2 y el banco R3 corrigen el índice y una espera
prematura del banco, y pasan la matriz ampliada y la regresión anterior.
La síntesis previa pertenece a los módulos originales. Ver
[diagnósticos, validación y siguientes pasos](AXI4_SECTION_FAULT_LIFETIMES.md).

## Resultado y alcance

`20261006T000729Z-V4-ffn-supervised-axi4-section-synth` sintetiza conjuntamente
la admisión retenida, protector de respuestas, certificados y **Clock Converter
AMD real**. Vivado 2026.1, build 6511674; `xcvu5p-flva2104-2-i`; dos hilos,
212,573 s. AXI4 de 512/64 bits, ID de cuatro bits, asincrónico y tres etapas.
La adenda registra `AXI4_SECTION_SYNTH_REVIEWED_WITH_NATIVE_CDC_WARNING`.
El diseño sigue sin aprobación física ni autorización de reset del endpoint.

Las tres instancias y conexiones del top coinciden textualmente con el
[banco R2](AXI4_RETAINED_SECTION.md); los 39 puertos de S conservan el contrato
de admisión. Es correspondencia de fuentes, no equivalencia formal del netlist.
Los resets H/D del CDC y el reset frío de retención siguen separados.

## Recursos y temporización preliminar

| Composición | LUT totales | LUT de memoria | Registros |
| --- | ---: | ---: | ---: |
| Sección completa | 10.631 | 818 | 13.771 |
| Admisión y protector | 4.947 | 24 | 3.495 |
| Clock Converter AMD | 998 | 746 | 3.122 |
| Certificados y monitores | 4.686 | 48 | 7.154 |

No se infirieron BRAM, URAM, DSP, latches ni cajas negras. Las medidas aisladas
anteriores no son sumables: la optimización conjunta modifica los recursos.

Con H=7 ns, D=5 ns y gestión=10 ns: WNS **+1,523 ns**, WHS **+0,013 ns**,
WPWS **+1,968 ns**; sin violaciones reportadas en esas comprobaciones internas.
Hay **1.438 entradas y 1.587 salidas sin retardo de E/S**. Los tres relojes
carecen de `HD.CLK_SRC`; faltan colocación/ruteo y restricciones de integración.
Estos resultados no establecen frecuencia máxima ni cierre temporal físico.

Los cinco caminos reportados hacia READY tienen retardo de datos estimado
máximo de 1,410 ns; los veinte desde READY, 2,376 ns. Todos muestran margen
no acotado (`inf`) por falta de restricciones externas. Son inspección de lógica,
no aprobación del protocolo temporal entre módulos.

## CDC, fallos y excepciones

El informe conserva 38 CDC-3 informativos, diez CDC-9 informativos y **un
CDC-15 Warning** dentro del FIFO AMD, con extremos protegidos ocultos. R4
confirma que ese aviso no está eximido. No se añadieron exenciones manuales.

Los 25 cruces propios CDC-3 usan tres etapas: seis alarmas registradas de
admisión, doce del par, seis señales de los tres mailboxes y una alarma agregada
de gestión hacia H. Las alarmas de admisión cruzan sin registro fuente adicional.
El bit lógico 6, diagnóstico de estado inválido, se optimizó a cero: **no hay
un séptimo detector físico de fallo ni cobertura de SEU**.

Las excepciones de RAM nativas cubren 888/602 pines de la IP. Recrear globalmente
el selector impreso por `report_exceptions` incluye además 144 primitivas de RAM
propias porque pierde el contexto de la IP; no usar ese texto como nueva XDC.
La inspección directa encuentra 93 salidas propias con caminos internos
acotados, 51 salidas cuyo recorrido acaba en ellas mismas y nueve contenedores
sin pines de primitiva. El menor margen de los 93 caminos es +2,052 ns.

En una copia de inspección en memoria, R4 aplica una única exclusión deliberada:
un camino pasa de margen +3,819 ns a no tener camino acotado. Se usa
`get_timing_paths -no_report_unconstrained` contra el mismo destino. La consulta
normal aún devuelve un objeto con `EXCEPTION=False Path` y margen vacío.
Este control negativo **no escribe checkpoints** ni modifica las restricciones
selladas. R3 sólo contaba objetos y no distinguía ese caso; se conserva como
antecedente limitado, sin relanzar la síntesis.

Los diez límites de bus skew de los punteros FIFO reportan 5 ns requeridos,
0,149 ns estimados y +4,851 ns de margen. Son estimaciones de síntesis.
Ambos resets H/D alcanzan 20 celdas; no se eliminó la entrada de reset D.

## Advertencias y evidencias

El DRC OOC reporta cero comprobaciones fallidas y advierte que omite ciertas
comprobaciones de conectividad. Metodología llega al límite de 1.000 avisos
TIMING-18 por E/S sin retardo; no equivale a una revisión física completa.
`Synth 8-7129` alcanza 100 avisos de puertos no usados y desactiva los siguientes.
También quedan registrados un objeto de exención nativa LUTAR-1 sin coincidencia
útil y once consultas de propiedad `USED_IN_SYNTHESIS` no disponibles. Los bytes
de las fuentes, XCI, XDC y bibliotecas se verificaron por separado.

| Tanda UTC del 2026-10-06 | Resultado |
| --- | --- |
| `000729Z` | Síntesis integrada y netlist/DCP preservados |
| `001725Z` | Diagnóstico: comando Tcl `redirect` no disponible |
| `001931Z` | Inspección R2 del DCP, cobertura y bus skew |
| `002738Z` | R3: caminos propios y control negativo inicialmente incompleto |
| `003422Z` | R4: control negativo discriminante y CDC con exenciones visibles |

Las cinco tandas están selladas y verificadas localmente y en NFS. La extracción
de la primera se reanudó tras falta temporal de espacio; no se repitió ni se
reselló la ejecución. Una compresión NTFS local conservó todos los hashes.

- Síntesis: `d9fb5666a1af1771bbefe3f58d7403482e1247f80ac4d72c5a0acba539b0fd86`.
- DCP: `1b36d56215364d280047ed46e42fa0a90bce0611b93bfb9939b1e74347b90e4e`.
- Inspección R4: `db19a4722671da8ae8c03e1dc81b8fea046ff1d43ea6eeee69a530b991a0db74`.
- Adenda conjunta: `1212ef9833949cfd454369878334385158b10dc2b0c6c9e8029f8a5bc6f476ad`.

## Siguiente dependencia

La [revisión de cancelación](AXI4_SECTION_CANCELLATION.md) ya cubre cuatro
reemplazos de tag con mailboxes ocupados. Faltan reemplazos de generación en
esa situación y fallos reales antes/durante el certificado. Conservar respuestas y solicitudes retenidas,
sin forzar bits internos. Investigar CDC-15 antes de aprobar la implementación;
las colas protegidas XDMA, coordinador global, DIMM exacto y placa física
permanecen abiertos en [la cola](VIVADO_QUEUE.md).
