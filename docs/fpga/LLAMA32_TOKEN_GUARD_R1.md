# Control de compromiso por token para dieciséis capas

Fecha: 2026-10-16. **Protocolo RTL probado con pares sintéticos;
síntesis/protocolo Vivado encolados; integración aritmética pendiente.**

Registro 15:01 UTC: coordinador `token-guard-vivado-r1`, PID 24452, al
final de la cola y después de `final-vocab-arithmetic-vivado-r1` revisado
y respaldado. Conserva las 85 entradas previas y el lugar de DDR R2.
Dos CPU, 1.800 segundos máximos, tres semillas XSim y ningún reintento
automático. El paquete fuente de 27 archivos está restaurado en D:/NFS;
SHA-256 `0c579ea86c2bdf01aafa1804c39e5f9409f130d5d4c917fd616cfa4c8055cf83`.

`fpga/rtl/ntrpu_llama32_token_guard_r1.sv` ordena dieciséis ejecuciones
de capa y la normalización/cabeza final. Retiene el residual de la capa
15 hasta drenar la cabeza; sólo confirma el token al aceptar su resultado
público. Valida identidad completa, época, posición, longitudes, direcciones
y prefijos KV. Cada capa recibe un tramo distinto de identificadores de
reserva; sus buffers siguen reservados por el propietario externo.

## Recuperación y límites

Un fallo después de aceptar una capa invalida el contexto: el compromiso
parcial de KV no se revierte. Para volver a admitir trabajo exige una
asignación real de caché y consultar los dieciséis prefijos en cero.
Cancelación, cambio de época o pérdida tardía del resultado revocan la
publicación de forma persistente. Una finalización ajena no se reconoce.

El plan de memoria llega mediante un certificado externo. Este módulo
comprueba alineación, desbordamiento y alias inmediatos; **no implementa
el asignador ni acredita que todas las regiones globales sean disjuntas**.
Todavía falta conectarlo con las capas reales, arbitraje y memoria de
pesos. No constituye una inferencia RTL del modelo completo.

## Evidencia de protocolo

Icarus, semillas 1/7/42, banco R2: 1.182 trabajos, 13.680 invocaciones
de capa y 678 de cabeza. Incluye avance de posición, fallos en las
dieciséis capas, caché sucia, identidades ajenas, cancelación tardía,
timeouts y recuperación. El revisor rechazó 42 alteraciones de trazas;
veinte mutaciones RTL fueron detectadas por comprobaciones funcionales,
sin contar expiraciones como éxito. Los tiempos del banco son sintéticos.

Captura nativa R2: `20261016T143743Z-token-guard-native-r2`, SHA-256
`17a3020c55291179b30239d2da614b52b03c60876b0b7251060de3fe61747138`.
Revisión y mutaciones: `20261016T144647Z-token-guard-native-r3`, SHA-256
`1305ad46c46021e62f60b002018d93f609d98f3ad561c1222d630b43fefaaef4`.
Ambos paquetes tienen copia D:/NFS y restauración comprobada. Se conservan
el banco y revisor previos, incluida la corrección del borde temporal de
asignación. La ejecución Vivado esperará la cadena existente; todavía
no hay recursos ni temporización medidos de este controlador.
