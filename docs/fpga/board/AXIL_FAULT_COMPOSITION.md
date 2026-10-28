# Fallos AXI-Lite: composición real y respuesta huérfana

## Resultado R2 y continuación

La contención posterior se documenta en [AXIL_RESPONSE_GUARD](AXIL_RESPONSE_GUARD.md).
Ese trabajo conserva el fallo R2 descrito aquí y añade una revisión de hardware;
los resultados siguientes corresponden a la composición anterior al protector.

La composición `ntrpu_axil_fence_section_r2` une el productor real
`ntrpu_axil_retained_admission_r2` con `ntrpu_axil_fence_certificate_r4`.
Pasa la regresión de doce casos posteriores al certificado, con Shutdown
Manager y Clock Converter AMD reales en simulación. La síntesis de la
composición reconoce once CDC-3 de tres etapas, sin CDC Critical/Warning.

El banco de fallos anteriores al certificado **no pasa**: una respuesta B
huérfana inyectada alcanza al Shutdown Manager abierto y activa su aserción
de contador vacío. Esa ruta necesita contención de respuestas antes de la
IP. No se aprueban todavía todos los estados de fallo, reset global, drenaje
interno de XDMA ni reutilización de sesiones NTRD.

## Corrección del origen CDC

La síntesis `20261005T193408Z-V4-ffn-supervised-shutdown-section-synth-r1`
conserva los bloques anteriores y conecta el origen real. El antiguo aviso
sobre un puerto externo se convierte en un CDC-10 concreto:
`admission/late_response_reg` y el fallo del inventario se combinan antes
del sincronizador de gestión.

La revisión de admisión solo expone sus dos fuentes registradas como
`fault_sources={late_response, ledger_fault}`. Mantiene intactos el límite
de 32, los contadores, el prefijo finito, el rechazo local y la lógica de
estado. El controlador R4 cruza ambas fuentes de manera independiente y
las combina en gestión. No añade un flanco necesario en origen: el host
puede detenerse inmediatamente después de registrar el fallo.

El bit 0 de `fault_mgmt` sigue representando fallo de admisión; los bits
1/2 siguen correspondiendo a H/D. El retorno agregado a host conserva el
registro de gestión introducido en R3. No se modifican los archivos sellados.

## Recursos y restricciones

| Composición | LUT del informe | FF | CDC |
| --- | ---: | ---: | --- |
| R1: admisión original + certificado R3 | 756 | 1.722 | 9 CDC-3 y 1 CDC-10 crítico |
| R2: admisión R2 + certificado R4 | 754 | 1.725 | 11 CDC-3, sin avisos |

La tanda corregida es `20261005T193725Z-V4-ffn-supervised-shutdown-section-synth-r2`.
No contiene RAM, DSP, latches ni cajas negras. Estos recursos cubren
admisión y certificación; las IP AMD de manager/CDC y el endpoint quedan
fuera de este top OOC. La reducción de LUT es el resultado de síntesis de
la composición, no una medida física de área de placa.

Se revisan once false paths XPM a primera etapa y tres max-delay
datapath-only de 10/7/7 ns. Los waivers de bus XPM son los del proveedor;
no se añaden waivers propios, grupos generales de reloj ni retardos de
entrada ficticios. El informe identifica las dos fuentes registradas reales.

WNS 4,716 ns, WHS 0,077 ns y WPWS 3,225 ns son estimaciones OOC. Faltan
retardos en 161 entradas/244 salidas, colocación de reloj, ruteo y verificación
funcional del netlist. Los diez avisos de síntesis corresponden a XPM y al
modelo OOC, y se conservan. No acreditan timing ni frecuencia física.

## Regresión funcional y fallo nuevo

`20261005T194221Z-V4-ffn-supervised-shutdown-section-post` instancia la
composición nueva en la regresión completa anterior: saturación, payload,
strobes, respuestas, cierre permanente, identidad fresca y reset local.
Los tres fallos posteriores al certificado pasan con dispositivo a 5/37 ns
y origen activo/detenido. Las latencias de revocación siguen siendo
20,8–28,8 ns simulados. La prueba retira ambas fuentes reales de admisión
y confirma que gestión conserva la alarma; 36 alteraciones de evidencia
son rechazadas. La IP usa las mismas 86 bibliotecas verificadas.

`20261005T194744Z-V4-ffn-supervised-shutdown-section-lifetime` pretende
probar cuatro situaciones: fallo de admisión previo a la prueba, buzón
pendiente con host detenido, B/R locales bloqueadas y cierre provocado solo
por fallo del dispositivo. El primer perfil falla y se conserva completo.

En el caso 3 se fuerza un handshake B en D sin AW/W previos. Es una
inyección deliberada de protocolo inválido; no tráfico legal generado
espontáneamente por la IP. La traza registra:

| Hecho | Tiempo simulado |
| --- | ---: |
| B huérfana aceptada en D | 2.783,6 ns |
| B huérfana aceptada en H | 2.825,2 ns |
| Aserción del manager: contador RTXC vacío | 2.846,2 ns |

Vivado termina con código 1. El mensaje es
`Failure: AXI4MM Write Channel: Decrementing RTXC when it is empty`.
No se desactiva la aserción ni se borra el contador mediante reset. Los
otros escenarios no obtienen aprobación por haber empezado a ejecutarse.
El cierre comunicado entre relojes no sustituye a validar la causalidad
de una respuesta en la frontera que la entrega al manager.

## Siguiente cambio necesario

Introducir una revisión con contención de respuestas entre el CDC y la
entrada de respuestas del Shutdown Manager. Debe observar continuamente
AW/W/AR aceptadas y admitir B solo con una escritura asociada y R solo con
una lectura asociada. Una respuesta huérfana queda retenida del lado CDC,
activa un fallo permanente y no entra al manager. No debe cortar VALID de
tráfico legítimo ya presentado ni reiniciar o descartar obligaciones válidas.

Sus fuentes registradas deben cruzar por separado a gestión. Repetir la
regresión completa y los cuatro escenarios, incluyendo backpressure,
orígenes detenidos, retención de payload e identidad de un buzón pendiente.
Añadir controles negativos sobre pérdida de una fuente y cancelación
prematura del envío XPM. El banco fallido permanece como evidencia.

## Reproducción y respaldos

Usar Vivado 2026.1 y directorios nuevos con:

- `tools/ntrpu_board_axil_section_synth_r1.tcl` y `_r2.tcl`.
- `tools/ntrpu_board_axil_section_post.tcl`.
- `tools/ntrpu_board_axil_section_lifetime.tcl` para reproducir el fallo.

| Tanda | SHA-256 del archivo local/NFS |
| --- | --- |
| Síntesis R1 diagnóstica | `2afc381438cadbeaa8f84f2d47678bb1a87c8bf4d3509a393b33857e772c6dc3` |
| Síntesis R2 corregida | `91ee4498211749ae530c886b145e1a5208d000fb80448c459a39ff7b0a89c1cc` |
| Regresión posterior al certificado | `e197ba5e3fe5050626de50d0db04ae32bc0fd886bb1acca8d48b3faf6c9f79e3` |
| Fallo antes del certificado | `0a062a6ab4964adbdc1f27ba4b9192a2f3c549843f2b91f251f62491d153b7f8` |

Adenda comparativa de síntesis:
`68064d90df62fc243de075bfac7450644686a889d767b548255442784a57381e`.
Fuentes, scripts, diagnósticos y miembros de los archivos están respaldados
y verificados. Continúan separados los requisitos V2–V9 de DIMM, placa,
inferencia y ensayo físico.
