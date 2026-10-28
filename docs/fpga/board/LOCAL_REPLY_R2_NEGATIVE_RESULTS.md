# Controles negativos nativos del proveedor local R2

## Resultado

`20261006T082416Z-V4-ffn-supervised-local-reply-negatives-r2-frozen` terminó
con código 0. El control positivo pasó la revisión independiente de
transacciones y VCD; las tres copias defectuosas fallaron por su aserción
prevista:

| Defecto privado | Aserción observada |
| --- | --- |
| Omitir `local_idle` y `reply_pending` | `NEGATIVE_COMPLETE_WITH_LOCAL_OBLIGATION` |
| Emitir un tag de ACK incorrecto | `NEGATIVE_ACK_IDENTITY` |
| Desconectar `section_fault_mgmt` | `shared section fault reaches local provider with host stopped` |

La revisión verifica el contenido exacto de cada proveedor mutado y del
monitor añadido al banco, preservación de todos los demás archivos RTL/TB,
y las rutas relativas resueltas de los proyectos de compilación XSIM. Así
se comprueba que cada simulación compiló su copia privada prevista. El
control positivo mantiene las comprobaciones de revocación CDC y retención.
La primera tanda con monitor demasiado estricto sigue registrada como
[diagnóstico](LOCAL_REPLY_R2_NEGATIVE_MONITOR_DIAGNOSTIC.md), sin aprobación.

## Evidencia y alcance

Respaldo local/NFS: 151.237.796 bytes y 1.986 archivos verificados.
SHA-256: `c5e131878aa2c21c760e58e3bc0a3e64b873406a7a5f37c5c6625c4078d52855`.
La revisión independiente tiene SHA-256
`ae756deb796096d6397aef5e7cf91a4f8e15e789853f2c54dcf267b25217d23c`.

Esta etapa reúne cuatro perfiles originales, cuatro con R2, dos de fallo
compartido y un control positivo adicional; detectó tres defectos RTL y
29 alteraciones de evidencia. Los 12 CDC-11 por duplicación desaparecieron
en síntesis. Véanse [composición y latencias](LOCAL_REPLY_R2_VALIDATION.md)
y [recursos OOC](DRAIN_SECTION_INDEPENDENT_REVIEW.md).

Quedan abiertos el CDC-15 nativo, timing físico e interfaces de E/S,
proveedores externos reales de ownership/host/XDMA y el reset completo del
endpoint. Los fixtures no autorizan reset nativo ni califican la placa de
512 GiB. El siguiente trabajo conserva estas dependencias: integrar contratos
externos demostrados antes de conectar el permiso global al secuenciador.
