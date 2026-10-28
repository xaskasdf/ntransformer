# Restricciones para rediseñar la tabla de descripciones D

## Problema independiente de B→R

El [ruteo Quick medido](PCIE_SYMBOLIC_MONITOR_PREDECODE_PHYSICAL_DIAGNOSTIC_R1.md)
tiene su peor setup en E→D: una LUT y 17,145 ns de interconexión hacia el
enable de una fila. Los cambios de validación tardía y sus pruebas binarias
no resuelven ese resultado. La lectura comparativa por etapas permitirá
separar ambos problemas antes de elegir otra ejecución física.

Esta nota inventaría el RTL actual; no propone una arquitectura ya validada.
Fuente: `fpga/rtl/ntrpu_pcie_symbolic_descriptions_predecode_r1.sv`, SHA-256
`af28fadbc30779a80fc373ab3f56e803897049926fd65b98593a77924af2f8d1`.

## Accesos y orden que deben conservarse

La tabla contiene 256 descripciones de 51 bits y cinco bits adicionales
por fila para clase de tráfico y atributos: **14.336 bits declarados**,
sin contar contexto retenido ni registros de salida. Esto no es un conteo
independiente de primitivas sintetizadas.

| Operación en un registro de entrada | Sitios de acceso descritos en RTL |
| --- | ---: |
| Consultas de cabecera | 4 lecturas de descripción y atributos |
| Fin de paquete | 4 lecturas de descripción |
| Cálculo de actualización RC | 4 lecturas de descripción |
| Contexto que se retiene | 1 lectura de descripción |
| Commit RC | Hasta 4 valores de actualización seleccionados por fila |
| Admisión RQ | 2 escrituras ordenadas |

Son sitios sintácticos, no una demostración de que los trece accesos sean
simultáneamente independientes en tráfico legal. `command.retained` puede
seleccionar el contexto por valor; esa selección tampoco permite descartar
puertos sin demostrar sus condiciones.

Las lecturas parten del estado anterior del registro de entrada y aplican
el resumen de su comando. Por fila se conserva primero el estado, después
se selecciona el commit RC y finalmente se aplican RQ0 y RQ1. La última
admisión coincidente tiene prioridad. El reset frío limpia todas las filas;
una entrada inválida conserva tabla, contexto y salidas.

## Criterios para un candidato nuevo

- Una partición por bancos debe aceptar los conflictos del contrato: no
  puede asumir tags distintos ni convertirlos en descartes o stalls.
- Añadir una etapa de lectura requiere reenvío de las escrituras aún no
  visibles. Debe cubrir los trece sitios, las colisiones RC/RQ, el contexto
  retenido y reutilización del mismo tag en ciclos consecutivos.
- Los 25 bits superiores se conservan al componer un resumen, pero el
  commit copia la descripción resuelta completa. No tratarlos como memoria
  inmutable por tag sin demostrar el contrato de los comandos retenidos.
- Una propuesta con RAM necesita un mapa explícito de lecturas, escrituras,
  conflictos y reset. Cambiar únicamente el tipo de almacenamiento no
  demuestra que su número de puertos alcance.
- Si cambia la latencia, definir una revisión del contrato y sus oráculos.
  Debe conservar capacidad, orden, caudal de dos RQ/cuatro RC por ciclo,
  drenaje, tratamiento X/Z y cancelación exclusivamente fría.

La selección debe considerar las rutas por etapa y una comparación física
equivalente. El [borrador de distancia al SOP](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_DISTANCE_R1.md)
es un experimento separado de B→R; no habilita por sí mismo este rediseño.
