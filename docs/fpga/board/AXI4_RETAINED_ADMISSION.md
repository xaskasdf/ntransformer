# Admisión AXI4 retenida y respuestas locales

## Implementación y alcance

`ntrpu_axi4_retained_admission.sv` incorpora el
[protector causal](AXI4_RESPONSE_GUARD.md) sin modificarlo. Mantiene estados
RUN → FINISH → REJECT durante toda la configuración; no tiene entrada de
reset operacional. Conserva datos de 512 bits, direcciones de 64, WSTRB,
IDs de cuatro bits, LEN/LAST y atributos, incluyendo REGION/QOS.

La admisión limita a 32 escrituras y ocho lecturas por ID. Al cerrar, termina
el [grupo finito](AXI4_ACTIVE_BOUNDARY_PLAN.md) de AW/W/AR ya aceptadas u
ofrecidas. Una W parcial cuenta una sola ráfaga, incluso entre transferencias
sin WVALID. Puede incluir la oferta número 33, respetando 32 obligaciones
simultáneas. Las ofertas bloqueadas conservan payload y permiso hasta el
handshake, también cuando un fallo hace caer la capacidad disponible.

Tras vaciar las solicitudes y respuestas downstream, REJECT responde SLVERR
localmente: B con el AWID después de la W completa, y ARLEN+1 respuestas R
con RID/RLAST correctos. Admite una escritura y una lectura locales a la vez,
incluyendo W anterior a AW. No reabre al cambiar tag ni solicitar otra prueba.
Un productor que cambie una AW bloqueada no altera la oferta original que ya
ve el receptor; queda registrado el fallo y se prohíbe certificar.

`source_fenced` afirma que no habrá nuevas solicitudes downstream.
`drain_ack` añade identidad fresca, inventario downstream vacío y dos
observaciones tranquilas. **No incluye respuestas locales pendientes**:
comprobar `local_idle` por separado antes de cualquier decisión sobre XDMA.
Los fallos del protector, del productor o de una W local invalidan la prueba
sin cortar respuestas sanas ya ofrecidas.

## Validación aislada R2

`20261005T225132Z` pasa **13 escenarios** en 37,354 s de Vivado, con reloj
de banco de 10 ns. El revisor usa colas Python e historiales absolutos de
transacciones para comprobar el grupo, capacidad, campos completos,
estabilidad de ofertas, respuestas y certificados. Rechaza **35 alteraciones**
de las trazas. La tanda inicial `223804Z`, diez casos y 26 alteraciones
rechazadas, permanece respaldada con sus fuentes intactas.

Los seis casos sin fallo de inventario completan 101 escrituras y 132 lecturas downstream:
4.196 transferencias W y 5.580 R. Además terminan 13 escrituras y 13 lecturas
locales: 626 transferencias W y 631 respuestas R con error local.

- Capacidad completa 32/128, con AW/W número 33 y AR número 129 bloqueadas
  al cerrar. El grupo incorpora esas ofertas sin exceder las capacidades.
- 31 W completas y una parcial antes de AW; 32 AW antes de W; todas las
  ofertas bloqueadas; cierre dentro de una W y entre sus transferencias.
- Rechazos locales de 1/2/4/16/64/256 transferencias, W antes/después de AW,
  backpressure y nueva prueba mientras B/R locales continúan retenidas.
- Siete casos de fallo: R huérfana, LEN de W incorrecto, AW/W bloqueadas
  durante pérdida de capacidad, corrupción de AW, W local incorrecta o sin
  WLAST en la transferencia 256 y B/R huérfanas posteriores al certificado.

El receptor es una fixture de transacciones que reordena B entre IDs e
intercala R con datos deterministas. Usa las cuatro codificaciones RESP como
patrones de transporte; EXOKAY no homologa accesos exclusivos. No es un controlador DDR ni PCIe.
Aquí no se ejecuta reset de un CDC real ni se detienen relojes.

## Controles negativos del RTL

`225532Z`, 104,015 s: cuatro copias privadas del módulo contra el banco R2
intacto. Tres defectos disparan aserciones HDL; el cuarto termina el banco
HDL y es rechazado por la comparación independiente. Ese PASS del banco no
aprueba el diseño mutado: el revisor forma parte del criterio funcional.

| Defecto | Evidencia de detección |
| --- | --- |
| Ignorar W parcial cuando WVALID está bajo | Caso 10, 183.270 ns |
| Cortar AW bloqueada al caer capacidad | Caso 6, 181.936 ns |
| Cortar W bloqueada al caer capacidad | Caso 11, 183.646 ns |
| Adelantar RLAST de la respuesta local | Revisor: caso 0, ciclo 9.786 |

## Recursos revisados

`224808Z`: **4.113 LUT / 3.496 FF**, incluidas 24 LUT de memoria distribuida;
sin BRAM, URAM, DSP, latches ni cajas negras. Incluye el protector, pero no
el Clock Converter ni el endpoint. Hay 3.495 FDRE con INIT=0 y un FDSE con
INIT=1 para el estado inicial one-hot de la admisión.

Con reloj OOC de 7 ns: WNS +3,675 ns, WHS +0,020 ns y WPWS +2,968 ns.
No hay endpoints internos sin reloj/restricción ni excepciones temporales.
Faltan retardos de 1.373 entradas/1.424 salidas y colocación/ruteo. El único
aviso es ausencia de `HD.CLK_SRC`; no hay aprobación temporal de placa ni
equivalencia funcional del netlist. Revisar los caminos combinacionales de
READY al integrar los componentes.

## Evidencia y siguiente integración

Tandas con prefijo `20261005T`, conservadas localmente y en NFS:

| Tanda | SHA-256 del archivo |
| --- | --- |
| `223804Z-V4-ffn-supervised-axi4-admission` | `b53bad05238162847f0dade1d3fbb4d230707b2811389efce972382f13f4cda4` |
| `224808Z-V4-ffn-supervised-axi4-admission-synth` | `f23a32ecb28dcb6065fb81493cba2e1ddbf4c2078eecf92034d14b00f0d8efab` |
| `225132Z-V4-ffn-supervised-axi4-admission-r2` | `1dc40b5b9eb0239b30eb4fdab51b71db373457edecebd215109268f6cff707d5` |
| `225532Z-V4-ffn-supervised-axi4-admission-mutants` | `3241f49c1dee5b339fd4dd443705bc369bad510a7e0bb25f47d7f45934c6c885` |

Adendas: inicial `a347ec328a4fdd0e52d33a17637d8f9812289ad52aa098d4488648ddadac182d`,
R2 `d4927d45342c389b464b597c69ebb725c36ac6be8d960c6978b3b0b147e4d504`
y síntesis `93261cc35fdd733ee87e663157605f397911e21134e2ad69bb665bd88d7fd0ec`.

La [primera composición con CDC AMD real](AXI4_RETAINED_SECTION.md) ya
verifica certificados frescos del productor y ambos extremos, reset exclusivo
del CDC con B/R locales retenidas y cuarentena R posterior al certificado.
Faltan otras vidas/fuentes de fallo, cancelación de pruebas ocupadas y síntesis
integrada. Las colas protegidas XDMA, coordinación global, reset nativo completo
y cierre físico mantienen sus dependencias; `complete_drain` sigue abierto.
