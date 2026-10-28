# Fallos durante certificados AXI4 y corrección del índice inicial

## Resultado vigente

El banco **R3** `20261006T013125Z-V4-ffn-supervised-axi4-section-lifetime-r3` pasa **18 casos** con los módulos candidatos
**RTL R2**: ocho fallos B/R y un cambio sano de generación por cada perfil
H/D=7/5 y 7/37 ns; gestión a 10 ns. La revisión reconstruye las transacciones
y los tres mailboxes desde las trazas completas y rechaza **224
alteraciones de evidencia**. No se han sustituido las versiones selladas.

| Momento del fallo externo | Respuestas que deben conservarse |
| --- | --- |
| Antes de la primera solicitud de certificado | B y último R normales |
| Durante la prueba de admisión | B y último R normales |
| Durante la prueba del par host | B/R locales después de cerrar admisión |
| Durante la prueba del par dispositivo | B/R locales ya retenidas |

Cada caso inyecta por el bus del dispositivo una B o R con ID 9 sin
solicitud causal. El Clock Converter AMD la transporta; los monitores la
detectan y el guard la contiene en el retorno host. No se fuerzan bits
internos. Detener el reloj inmediatamente después del registro del fallo
no impide que gestión lo observe. La latencia simulada observada es
20.9–26.9 ns desde D y
21–27.5 ns desde admisión.

Tras el fallo, una nueva generación y tag no autorizan el certificado ni
reabren admisión. Las respuestas sanas terminan y las operaciones locales
no alcanzan el CDC. Se verifica una escritura y lectura normal de 16 beats
por caso; los casos cerrados añaden una operación local de 16 beats por
dirección. El control sano cambia la generación con D detenido y su
mailbox ocupado: conserva el mensaje antiguo, registra una finalización
obsoleta y exige un certificado de la generación nueva.

## Dos diagnósticos conservados

1. `20261006T011354Z-V4-ffn-supervised-axi4-section-lifetime`: nueve casos rápidos pasan, pero el primer caso lento agota
   el tiempo antes de inyectar el fallo. La sonda de solo lectura
   `20261006T012312Z-V4-ffn-supervised-axi4-section-lifetime-probe` muestra `partial_index=x` con cabeza y cola a cero.
   Al llegar AW con W parcial de 14 beats, a 3,5945 µs, el detector lee
   ese índice; al siguiente flanco se registra un estado indeterminado.
2. `20261006T012510Z-V4-ffn-supervised-axi4-section-lifetime-r2`: el índice continuo elimina esa indeterminación. El banco
   entonces falla una precondición porque su contador ya anuncia 15 R
   consumidas mientras la respuesta final aún atraviesa el CDC.

La corrección RTL R2 usa asignaciones continuas para el índice en el guard
y los monitores: la expresión se evalúa desde tiempo cero aunque sus
entradas inicializadas no cambien. Otros tres módulos R2 solo actualizan
esas dependencias. Es una corrección reproducida de simulación RTL; el
contraejemplo no demuestra un fallo físico del silicio.

El banco R3 espera explícitamente RVALID/RLAST y los contadores de ambos
extremos vacíos antes de comprobar la precondición original. Conserva las
aserciones, transacciones e inyecciones. Ningún intento fallido se reetiqueta
como aprobado ni se sobrescribe.

## Regresión y respaldo

`20261006T014212Z-V4-ffn-supervised-axi4-index-r2-regression` pasa los bancos anteriores sobre RTL R2: 22 casos del
monitor, 11 de capacidad, 19 del guard y 13 de admisión; además, ambos
perfiles completos de cancelación/reset. Estos conservan 24 escrituras y
lecturas, 1.045 beats por dirección, cuatro cancelaciones, tres certificados,
reset solapado del CDC y contención posterior de R huérfana. Se reutilizan
los comprobadores independientes sellados y estímulos sin cambios.
La adenda corrige la colección: el proyecto final solo enumera su top
activo; los compiladores guardados por banco identifican los otros tops,
contrastados con hashes de fuentes y snapshot sellado. No se repitió Vivado
para resolver ese diagnóstico de colección.

Las cinco tandas conservan fuentes, logs, diagnósticos y archivos verificados
en NFS y `artifacts/ntrpu-board-vivado/`. SHA-256 principales:

- Matriz aprobada: `79073e05bd70212c85460dbcb730623a5c86c5a975d5477d2d02c9ced1c95749`.
- Adenda conjunta: `4371260c11e5e833aff03d3851d6080c02de38ef0a5c1501ecb12b0df2807636`.
- Regresión: `1a3fdb45b334ecb8e6b004c8c38d8bf9f39d2c47e3b75c5f8fa0e42ac8056393`.

## Dependencias abiertas

La [síntesis candidata R2](AXI4_SECTION_SYNTHESIS_R2.md) ya está revisada:
10.603 LUT y 13.771 FF, con CDC-15 nativo y restricciones físicas pendientes.
La [matriz de solicitudes](AXI4_SECTION_REQUEST_FAULTS.md) añade cinco causas
externas en dos perfiles y dos controles RTL negativos. Siguen pendientes
otras causas durante reenvío, inventario protegido XDMA, reset global,
DIMM exacto y diseño físico.
El modelo FIFO AMD omite retardos de sincronización de datos: estas medidas
no son caudal físico ni autorización para resetear el endpoint o reutilizar
propiedad. Ver [cola principal](VIVADO_QUEUE.md).
