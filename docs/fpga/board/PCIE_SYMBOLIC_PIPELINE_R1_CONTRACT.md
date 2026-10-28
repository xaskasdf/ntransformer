# Pipeline simbólico R1: siguiente implementación

Candidato del 2026-10-09, **todavía no implementado ni medido**. Parte del
modelo por valor, que pasó los bancos heredados y el banco de vida del
contexto. Este contrato fija una primera partición verificable; no afirma
cierre de timing. Mantiene [todos los requisitos del pipeline](PCIE_INVENTORY_PIPELINE_CONTRACT.md).

## Registros y datos

| Frontera | Responsabilidad | Datos que deben quedar registrados |
| --- | --- | --- |
| S1 | Control crudo y commit simbólico, RC antes de RQ | Cuatro consultas de 51 bits con propiedad/count/TC/atributos; contexto simbólico previo; eventos efectivos EOP; spans RQ; beat crudo, configuración, estado de control resultante y secuencia |
| S2 | Evaluar copias de descripciones | Metadatos y delivered de consultas; contexto previo numérico para validación; contribuciones RC de 13 bits por evento; spans RQ y todo el estado alineado de S1 |
| S3 | Validar, reducir deltas y retirar | Estado público correspondiente al beat, bytes con guarda de rango y fallos acumulados de ese mismo origen |

Primera latencia candidata: entrada aceptada/procesada en el flanco N de S1,
aplicación registrada en N+1, publicación en N+2. La comparación debe usar
el origen del beat; no añadir esperas arbitrarias al banco existente. S1
mantiene un beat RQ2 y RC4 por ciclo, 256 tags, 257 lecturas y capacidad de
1.052.672 bytes. Si alguna frontera no alcanza el período, revisar formato
y latencia explícitamente en una revisión nueva, manteniendo ese ritmo.

## S1 no espera la aritmética

Las consultas y commits usan descripciones completas por valor. La admisión
RQ produce su semilla desde el descriptor crudo; RC concatena consumos sin
aplicar remaining/dirección. TC/atributos siguen separados. Preservar
exactamente la prioridad R3, incluidos punteros/slots malformados; no
reemplazarla por una secuencia ideal de paquetes válidos.

Cada contribución EOP lleva descripción de 51 bits, count de 11 bits,
condición de entrega capturada, terminal y elegibilidad. Puede codificarse
en cuatro slots solo después de comprobar que conserva todas las
contribuciones por lane, también ante alias de punteros y X/Z. El caso
terminal retira original; el no terminal normal retira delivered. La
condición de entrega proviene de la cabecera, no del `owned` borrado en EOP
ni de `normal` alterado por discontinue. Ningún resultado S2 escribe de
vuelta la tabla o el contexto vivo de S1.

## Identidad, wrap y cancelación

El payload retenido conserva su descripción completa; una nueva solicitud
con el mismo tag usa otra copia. No hay ticket a una fila numérica futura.
Los registros pendientes se desplazan con su bit válido; no se direccionan
con el contador de secuencia. Así, el wrap de secuencia no selecciona un
propietario ni puede sustituir una descripción anterior.

Esta regla exige verificar ausencia de referencias a slots reutilizables
y de writeback tardío. Cualquier introducción de punteros obliga a derivar
su máxima vida y ancho de generación. Reset frío cancela todos los válidos
en el mismo flanco y borra estado/fallos; reset de usuario y pérdida de link
conservan obligaciones y generan fallo. No usar la secuencia diagnóstica
como autorización para cancelar o fusionar eventos.

## Contadores, fallos y vacío

Los contadores de tags/lecturas y demás control resultante de S1 se retrasan
hasta su publicación S3. El contador de bytes se actualiza en orden con
los deltas evaluados, sin volver a leer el tag. Mantener anchos, truncamientos,
comparación signed/unsigned y guardas de rango existentes. El fallo de
rango debe acompañar al estado del origen correcto, incluida su retención
sticky; no desplazarlo un ciclo accidentalmente al mover el contador.

Los beats con efectos capturados son obligaciones hasta retiro. Mantener
bits de actividad por etapa y contexto parcial/stalls. Revocar vacío desde
las entradas actuales y los fallos ya detectados en S1. Los snapshots de
inactividad siguen retirándose para diagnóstico, pero no mantienen ocupado
el pipeline eternamente. Especificar `pipeline_pending` desde esas
obligaciones y estados, y comprobarlo junto con la secuencia de retiro.

## Trabajo y gates encolados

1. Implementar y probar S1 contra consultas/commits del modelo por valor;
   congelar el formato exacto de eventos EOP y spans RQ. Medir S1 a 4 ns
   con I/O cero y sin excepciones, admitiendo todos los accesos simultáneos.
2. Implementar S2 y S3 con scoreboard por origen y cancelación explícita.
   Conservar todas las filas tras fallo, capacidades y bancos X/Z. Probar
   wrap y reset con cada etapa ocupada, payload largo/reutilización y
   mutantes reales de omisión, inversión, duplicación y vacío prematuro.
3. Ejecutar el banco completo del monitor integrado, sintetizar y revisar
   caminos propios; comparar su propia netlist binaria. Medir las tres
   fronteras, sin extrapolar el resultado aislado de S1.

El modelo no aporta medidas físicas nuevas. R12 sigue como referencia de
implementación; R15 conserva su regresión documentada. Los requisitos de
CDC/reset, observación nativa, DDR de 512 GiB, pesos reales y placa siguen
abiertos en [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
