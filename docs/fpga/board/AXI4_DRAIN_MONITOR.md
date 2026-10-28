# Inventario retenido AXI4: primera implementación aislada

## Revisión posterior del índice inicial

La ampliación con fallos antes/durante el certificado encontró un índice
indeterminado al arrancar en la simulación RTL original. Se conserva la
evidencia previa con su alcance; no prueba todas las secuencias iniciales.
La revisión candidata RTL R2 y el banco R3 corrigen el índice y una espera
prematura del banco, y pasan la matriz ampliada y la regresión anterior.
La síntesis previa pertenece a los módulos originales. Ver
[diagnósticos, validación y siguientes pasos](AXI4_SECTION_FAULT_LIFETIMES.md).

## Contrato y capacidad

`fpga/rtl/ntrpu_axi4_drain_monitor.sv` observa el perfil de datos de 512 bits,
direcciones de 64 bits e IDs de 4 bits. Solo `cold_rst_n` borra inventarios;
no tiene entrada de reset operacional. Es pasivo: no impone el fence externo,
no controla READY ni contiene respuestas inválidas.

La primera configuración probada permite **ocho obligaciones AW y ocho W
completas hasta B**, más ocho lecturas por ID. AW y W se registran por
separado, incluyendo W anterior a AW. Su asociación comprueba LEN/LAST;
B necesita crédito previo para su ID. Una asociación interna pendiente
no retrasa artificialmente el crédito de una B legal inmediata. AR/R usan
colas por ID, con orden dentro del ID e intercalación entre IDs.

Se retienen seis fuentes de fallo: capacidad, B huérfana, R huérfana,
límite W, límite R y cambio/retirada de una oferta bloqueada. El fallo
congela el inventario e invalida permanentemente el ACK. Una petición sana
requiere identidad de 64 bits, fence continuo, inventario vacío, ningún
VALID y dos observaciones tranquilas después de capturar la identidad.

Este primer banco usa ocho escrituras y alcanza ocho lecturas de un ID.
La [continuación de capacidad y CDC](AXI4_CDC_DRAIN.md) ya prueba el perfil
de 32 escrituras y satura las 128 lecturas agregadas. La futura admisión AXI4
debe imponer esos límites; el límite AXI-Lite de 32 permanece igual.
La vigilancia del payload no valida por sí sola todos los atributos AXI,
rangos DDR o semántica de respuestas con un ID que sí tiene crédito.

## Simulación y referencia independiente

`20261005T210303Z-V4-ffn-supervised-axi4-drain` pasa **22 escenarios** en
32,965 s de ejecución Vivado; tiempo simulado total 13,33 µs. Usa XSIM real,
sin manager ni Clock Converter. La referencia Python reconstruye colas,
datos de 512 bits, WSTRB, respuestas y ACK desde cada handshake.

- Cinco casos sanos: longitudes 1/2/4/16/64/256, los 16 IDs, W antes de AW,
  B reordenadas entre IDs, R intercaladas, pausas y direcciones sobre 4 GiB.
- Saturación de ocho escrituras y de ocho lecturas de un mismo ID; 24
  retiros/altas simultáneos con la cola de lectura llena y vuelta de índices.
- Secuencia de 64 operaciones con AW/W/AR nuevas y B/R anteriores aceptadas
  simultáneamente. Mide capacidad del observador, no caudal DDR/PCIe.
- Diecisiete casos inválidos: huérfanas, respuesta prematura, LEN/LAST,
  overflow y cambio de payload/VALID bloqueado. El fallo persiste con tag nuevo.

Las ráfagas de 256 transferencias usan tamaño de 16 bytes dentro de 4 KiB;
las restantes usan 64 bytes. Se rechazan 34 corrupciones de la evidencia.
Único aviso conocido de entorno XSIM; ninguno durante ejecución del banco.

## Controles negativos del RTL

`20261005T210557Z-V4-ffn-supervised-axi4-drain-mutants` compila cuatro copias
privadas contra el banco positivo intacto. Todas activan la aserción prevista:

| Cambio deliberado | Detección |
| --- | --- |
| ACK conserva autorización al cambiar tag | Caso 0, 7.471 ns |
| B recibe crédito de AW/W aceptadas en el mismo flanco | Caso 7, 10.760 ns |
| Se elimina comprobación de límites W | Caso 9, 11.080 ns |
| Se elimina comprobación de RLAST | Caso 11, 11.410 ns |

## Respaldo y continuidad

Las tres tandas conservan fuentes, compilación, trazas, revisión y manifiestos en
`artifacts/ntrpu-board-vivado/<ID>/`, verificados localmente y en NFS.

| Tanda | SHA-256 del archivo |
| --- | --- |
| `210303Z` | `da197d05d8668495a3ff696aa2d154644dece1de710e49fa6f08f8ae4a70eed6` |
| `210557Z` | `7d9ec12d74612ec9e9e646b4d65775b541a44376f903eca895407c34247f7f26` |
| `210852Z` síntesis | `ac455de557975d08c6714e7088f51f5e0f4936c6f7be18faff402f4b13f21580` |

La síntesis `210852Z`, revisada mediante adenda, mide **2.438 LUT**
(2.414 de lógica y 24 de memoria distribuida) y **2.771 FF**. No tiene
BRAM, URAM, DSP, latches ni cajas negras. Con reloj OOC de 7 ns:
WNS +3,961 ns, WHS +0,020 ns y WPWS +2,968 ns. No hay endpoints internos
sin reloj/restricción ni excepciones temporales. Es un único dominio: el
reporte CDC vacío no comprueba el cruce futuro.

Faltan retardos de 1.357 entradas y 97 salidas. Los dos avisos revisados
son granularidad del netlist para floorplanning y falta de `HD.CLK_SRC`,
que impide estimar retardo/skew de reloj OOC. Estos márgenes no acreditan
timing de placa. Adenda SHA-256:
`64dc946e510f30aa84f8c84bf3255f17e92d31e6a268ee13ad8d569610997a6a`.

La [continuación](AXI4_CDC_DRAIN.md) cubre capacidad y ambos extremos del
CDC AMD real. Siguiente: frontera activa con cierre de grupo finito y
contención causal de respuestas antes del endpoint protegido.
El [plan AXI4](AXI4_DRAIN_PLAN.md) mantiene separadas las colas internas XDMA,
el coordinador y reset nativo completo. Este resultado no autoriza reset
global, reutilización NTRD, inferencia Llama completa ni fabricación de placa.
