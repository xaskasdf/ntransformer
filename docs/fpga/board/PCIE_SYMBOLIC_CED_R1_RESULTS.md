# Frontend simbólico C/E/D: validación RTL

Revisión del 2026-10-09. La partición C→E→D está implementada y superó
sus bancos RTL. **Todavía no tiene medidas propias de síntesis, replay
de netlist ni cierre físico.** La aplicación numérica y el retiro de bytes
del monitor completo siguen pendientes.

## Implementación y comparación

C conserva selección R3, propiedad y control; E prepara efectos sin leer
descripciones; D mantiene tabla y contexto por valor, con commit RC antes
de RQ. El resultado corresponde al origen S1 R1 con dos ciclos adicionales.
El [contrato](PCIE_SYMBOLIC_CED_R1_CONTRACT.md) delimita las señales pendientes.

Los seis bancos originales suman **80.120 filas**. Se comparan los 2.498
bits externos, las 256 filas simbólicas y el contexto retenido por origen,
además del scoreboard heredado S1/R7/modelo, incluidas filas posteriores a
fallos y X/Z. El reset frío cancela correctamente orígenes en vuelo.

Para observar también el resultado de tráfico que esos resets cancelaban,
se repitieron los seis bancos con dos filas inactivas antes de cada reset:
**88.808 filas**, conservando intactas todas las originales. Se retiraron
**71,420 orígenes** y compararon **18,283,520 palabras de tabla**;
ninguna fila de tráfico quedó cancelada en esta repetición. Las filas
añadidas mantienen visibles los errores de handshake que puedan provocar.

Otros **217 ciclos en 16 escenarios** cubren reset a distancias 0/1/2,
actividad cruda durante cold, contexto separado del tag reutilizado,
reset de usuario/pérdida de link, tráfico adyacente durante wrap y tag X/Z.
El banco ampliado comprueba también control y captura de C en cada fila.

## Controles negativos y diagnóstico

Cinco fuentes físicamente distintas fueron rechazadas por el scoreboard:

| Mutante | Fila detectada | Fallo |
| --- | ---: | --- |
| `omit_eop` | 9 | `CED_PORT_MISMATCH` |
| `repeat_e` | 5 | `CED_PORT_MISMATCH` |
| `wrong_ownership` | 176 | `CED_C_CONTROL_MISMATCH` |
| `reload` | 8 | `CED_RETAINED_CONTEXT_MISMATCH` |
| `cold` | 5 | `CED_COLD_MISMATCH` |

La primera tanda detectó un flanco no registrado en el scoreboard al leer
EOF. La revisión del testbench limita la lectura al número congelado de
filas y rechaza datos sobrantes sin avanzar el reloj. Se conservó el
diagnóstico; el RTL y la latencia permanecieron iguales.

El banco ampliado detecta la recarga por valor antes del fallo de puertos
esperado por su lanzador. Una tanda corta conserva ese resultado y completa
los controles de recarga y cold con el mismo RTL y scoreboard.
Los bancos positivos previos permanecen vinculados a sus fuentes exactas.

Se revisó explícitamente el aviso VRFC 10-3380: `rq_held` se referencia antes
de su declaración única como registro de módulo. Los escenarios de stall
y las comparaciones completas pasaron. No se aceptaron otros avisos fuera
de los registrados por cada revisión. La declaración se puede ordenar en
una revisión futura; estas evidencias vinculan el RTL exacto ejecutado.

## Evidencia y siguiente trabajo

Las cuatro tandas tienen archivos, manifiestos y fuentes verificados en D:
y NFS. La VM y los procesos externos se conservaron.

| Tanda | Identificador | SHA-256 del archivo |
| --- | --- | --- |
| pcie_symbolic_ced_r1_eof_diagnostic | `20261009T125328Z-V4-ffn-supervised-pcie-symbolic-ced-r1-rtl` | `d84eafeff9cda4b971621ee0c0a872772dae854ee72e48f0094e09e44902b2a3` |
| pcie_symbolic_ced_r1_primary | `20261009T125938Z-V4-ffn-supervised-pcie-symbolic-ced-r1-rtl-r2` | `ae4e7ede4b1d6d7acc9d23f4397fc3ce47d1c5a978ffb62622a73a837b93e5b7` |
| pcie_symbolic_ced_r1_stage_marker_diagnostic | `20261009T131157Z-V4-ffn-supervised-pcie-symbolic-ced-r1-stage-rtl` | `dd7b35967ecd4e498c2172d9b826847e3cafd3a05de047106febe44c82e0c03c` |
| pcie_symbolic_ced_r1_stage | `20261009T132224Z-V4-ffn-supervised-pcie-symbolic-ced-r1-stage-rtl-r2` | `de523813e062ec14dd711adae6d3737224b18eee87d5033e6031cd6aeb2d25a9` |

Siguiente trabajo: caracterización separada C/E/D y conjunta a 4 ns,
I/O cero y sin excepciones, seguida de replay sobre su propia netlist.
R12 conserva su estado como referencia del monitor completo. Las
condiciones físicas y de placa de [VIVADO_QUEUE.md](VIVADO_QUEUE.md)
permanecen abiertas.
