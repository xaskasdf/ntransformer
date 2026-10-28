# Propuestas aritméticas para completions PCIe

## Alcance y resultado

El 2026-10-07 se implementó `fpga/rtl/ntrpu_pcie_rc_proposal.sv` para
separar el cálculo de bytes/dirección de la consulta del banco de tags.
Procesa cuatro descriptores RC por ciclo, en posiciones DWORD 0/4/8/12,
y registra las propuestas con un ciclo de latencia. El resultado es una
**propuesta pendiente de validación**, no una transición autorizada.

El RTL pasó 69.876 ciclos contra cálculos enteros independientes: 234.656
cabeceras normales, 12.030 no normales, 54.510 ciclos con cuatro cabeceras
y 19 ciclos de reset. Se cubrieron los 2.048 valores de `DWORD Count`,
los cuatro desplazamientos bajos y casos de desbordamiento/longitud inválida.
Ocho controles de corrupción fueron rechazados por el verificador.

Las cabeceras no normales se capturan, pero no producen aritmética válida.
La dirección propuesta conserva el wrap de 12 bits. El cálculo explicita
el comportamiento unsigned de R7 cuando count=0 y la dirección no está
alineada; esos casos siguen marcados como longitud inválida.

## Comparación con el monitor completo

El banco `tb_ntrpu_pcie_rc_proposal_shadow_r1.sv` ejecutó los **22.339
ciclos/68 escenarios** de R7. Todas sus salidas públicas, incluidas las
posteriores a fallo, coinciden con la tanda R15 revisada. El verificador
independiente reconstruye cada propuesta y los eventos candidatos del
contexto retenido. Comprueba también la pertenencia desde el bitmap previo
y las liberaciones terminales anteriores del mismo beat.

Se observaron 1.094 cabeceras normales, dos no normales y una con bits X;
134 ciclos usaron las cuatro posiciones. Los 22.155 ciclos sin fallo no
presentaron diferencias de eventos. Los 184 ciclos con fallo se conservaron
íntegros, con estas diferencias explícitas:

| Fila / escenario | Diferencia observada | Consecuencia |
| --- | --- | --- |
| 4131 / 11 | El descriptor declara 5 bytes; R7 espera y descuenta 4 | No aplicar la propuesta sin comparar el estado esperado |
| 4140 / 12 | Dirección declarada 257; esperada 256; evento final coincidente | Comparar direcciones aunque el descuento terminal coincida |
| 4291 / 29 | Respuesta no normal: propuesta sin resolver; R7 completa y descuenta 4 bytes | Conservar obligación y fallo explícitos; faltan metadatos esperados |
| 4300 / 30 | Error con campos X: mismo desacuerdo de completion/descuento | No interpretar campos inválidos como prueba de finalización |

Hay tres filas con diferencias de eventos y dos con diferencias de
metadatos. Nueve controles de corrupción del registro fueron rechazados.
Los metadatos esperados y eventos de referencia proceden del **R7 sin
modificar**; el contexto candidato sigue siendo código de simulación.
Esto no demuestra equivalencia general ni integra un parser nuevo.

La primera tanda shadow se detuvo en la fila 0: la máscara combinacional
del banco carecía de inicialización y sus primeras entradas coincidían
con los valores iniciales. R1 inicializa esa máscara e imprime los eventos
antes de comprobarlos. Ambas revisiones y el fallo están preservados.

## Síntesis real en Vivado

Vivado 2026.1, `xcvu5p-flva2104-2-i`, reloj de 4 ns, restricciones I/O
explícitas de cero y ninguna excepción temporal:

| Medida OOC | Resultado |
| --- | ---: |
| LUT / FF / CARRY8 | 449 / 552 / 20 |
| Latches / BRAM / URAM / DSP | 0 / 0 / 0 / 0 |
| WNS / TNS | +1,616 ns / 0 |
| Setup: endpoints fallidos / total | 0 / 1.596 |
| WHS / THS | −0,075 ns / −29,852 ns |
| Hold: endpoints fallidos / total | 460 / 1.596 |
| Peor camino | `header_data[37]` → `length_error_reg[0]/D` |
| Retardo / niveles | 2,394 ns / 9 |

El EDIF final contiene 1.096 primitivas y 1.710 nets, sin ciclos
combinacionales; cuatro controles verificaron el auditor. Los doce controles
de cobertura temporal dieron cero incidencias. Hold sigue fallando: no hay
cierre físico, placement/routing ni reloj del diseño padre.

Se conservan cuatro avisos `Synth 8-589` por convertir `===` en `==`;
la reproducción del netlist se limita a entradas binarias conocidas.
Los avisos `Synth 8-7129` corresponden a padding alto de los slots de
128 bits. Permanecen los avisos OOC de cobertura DRC, reloj y 1.069
ocurrencias XDCH-2 por min/max I/O iguales; no se han silenciado.

El **netlist funcional pasó los mismos 69.876 ciclos**, con cada fila
idéntica al RTL y al cálculo independiente. Se compiló el archivo sintetizado
sin el DUT RTL, con espera inicial de 200 ns y muestreo a 1 ns; no se aplicó
SDF. Ocho controles detectaron corrupción de resultados. El aviso de doble
definición de `glbl` se revisó comparando ambos cuerpos archivados: son
idénticos. Esto no extiende la prueba al dominio X/Z ni resuelve hold.

## Tandas y evidencia

Cada ID tiene fuentes, configuración, revisión y archivo verificado en
`artifacts/ntrpu-board-vivado/<ID>/`, con almacenamiento en D: y copia NFS.

| ID, prefijo `20261007T` | Trabajo |
| --- | --- |
| `180516Z-V4-ffn-supervised-pcie-rc-proposal-rtl` | RTL y cálculo independiente |
| `182557Z-V4-ffn-supervised-pcie-rc-proposal-shadow` | Diagnóstico de inicio del banco |
| `183007Z-V4-ffn-supervised-pcie-rc-proposal-shadow-r1` | Comparación completa con R7 |
| `183220Z-V4-ffn-supervised-pcie-rc-proposal-synth` | Síntesis, timing y auditoría EDIF |
| `183640Z-V4-ffn-supervised-pcie-rc-proposal-netlist` | 69.876 ciclos de netlist revisados, idénticos al RTL |

## Siguiente integración

Combinar estas propuestas con el [banco de metadatos](PCIE_TAG_METADATA.md)
sin volver a encadenar su multiplexor de lectura con la aritmética.
El parser debe producir eventos ordenados y conservar contexto RC hasta
EOP, independiente de la reutilización del tag. La validación debe comparar
bytes, dirección, identidad y atributos contra fuentes esperadas independientes.
Ningún dato recibido puede definir su propia identidad esperada.

Queda por implementar la programación de escrituras/consultas, el forwarding
entre eventos y el retiro correlacionado. Una propuesta especulativa o una
respuesta sin resolver debe mantener una obligación explícita y revocar vacío.
Definir reset, épocas y wrap antes de integrar, manteniendo 256 tags,
RQ2/RC4, 257 lecturas y 1.052.672 bytes sin burbujas del observador.
Siguen vigentes el [contrato de pipeline](PCIE_INVENTORY_PIPELINE_CONTRACT.md)
y todos los criterios de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
