# Frontera activa AXI4: contrato de integración

## Orden y bloques

La frontera retenida se sitúa entre el productor protegido y el Clock
Converter. Observa solicitudes aceptadas en ese límite; el
`ntrpu_axi4_response_guard` intercepta B/R de retorno antes del productor.
Los [monitores de ambos extremos](AXI4_CDC_DRAIN.md) siguen observando la
interfaz real del CDC. El protector no reemplaza esos inventarios ni demuestra
vacías las colas internas XDMA. La [implementación aislada](AXI4_RETAINED_ADMISSION.md)
ya pasa trece escenarios y tiene síntesis revisada. La
[primera composición con CDC real](AXI4_RETAINED_SECTION.md) añade certificados
frescos, reset exclusivo del CDC y cuarentena R posterior al certificado.
Este contrato sigue abierto para otras vidas de fallo y colas protegidas XDMA.

## Capacidad y grupo finito

Usar 32 obligaciones AW/W y ocho lecturas por ID, incluyendo respuestas
retenidas por el protector hasta su aceptación aguas arriba. Sus señales
de capacidad son conservadoras: una respuesta retirada libera espacio en
el ciclo siguiente. Conservar direcciones de 64 bits, datos de 512, WSTRB,
IDs, LEN/LAST y atributos; REGION/QOS deben tener un contrato explícito.

Al cerrar, calcular con el estado **anterior al flanco**:

```text
N = max(AW_pendientes + AW_ofrecida,
        W_completas_pendientes + (W_parcial != 0 o W_ofrecida))
AW_restantes = N - AW_pendientes - AW_aceptada_en_este_flanco
W_restantes  = N - W_completas_pendientes - WLAST_aceptada_en_este_flanco
AR_restante = AR_ofrecida y no aceptada_en_este_flanco
```

`W_restantes` cuenta ráfagas, no transferencias. Una ráfaga parcial cuenta
una sola vez. El grupo puede incluir la oferta número 33, pero jamás más
de 32 obligaciones simultáneas. No ampliar N después del cierre. Terminar
únicamente esas AW/W/AR; una mitad faltante o reloj detenido impide certificar.

Una oferta VALID ya presentada conserva permiso y payload hasta su handshake,
incluso si aparece un fallo y cae una señal de capacidad. No conectar esas
señales directamente a VALID sin preservar la oferta. Un fallo de inventario
impide certificar; no habilita reset, reapertura ni descarte silencioso.

## Respuestas locales después del drenaje

Solo después de vaciar el grupo y las respuestas sanas, pasar irreversiblemente
a rechazo local SLVERR. Implementar AW y W independientes, incluyendo W antes
de AW; producir B con el AWID correcto después del WLAST y LEN completos.
Para cada AR rechazada, producir exactamente ARLEN+1 respuestas R con RID y
RLAST correctos, conservando payload bajo backpressure. Empezar con una
operación local por dirección; documentar esa capacidad separada.

Mantener separados `source_fenced`, prueba del tramo CDC y `local_idle`:
el primero solo afirma que ya no se emitirán solicitudes downstream;
el último incluye solicitudes y respuestas locales todavía pendientes.
El reset del endpoint necesita además la detención y el inventario interno
XDMA. Ninguna identidad nueva borra cuarentena o reabre la frontera.

## Criterios del próximo banco

- Cierre durante AW/W/AR bloqueadas, W parcial y W completa antes de AW;
  capacidad 32/128, IDs repetidos, longitudes 1/2/4/16/64/256 y vuelta de colas.
- Conteo independiente del grupo, comparación de ambos lados y ausencia de
  nuevas solicitudes downstream después del fence; preservación de VALID.
- Rechazos locales B/R completos y estables; respuesta local retenida durante
  reset exclusivo del CDC, sin borrar el estado de la frontera.
- Fallo antes/durante/después del certificado y mailbox pendiente; fuente
  detenida inmediatamente después de registrar el fallo.
- Composición con CDC AMD real a 5/37 ns, controles RTL negativos, síntesis,
  revisión de cruces y respaldo antes de integrar colas protegidas XDMA.
