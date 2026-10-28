# Productor causal del inventario PCIe

## Alcance de la primera integración

El productor recibe exclusivamente las entradas públicas RQ/RC y la
configuración. Posee su bitmap, tabla de 256 tags, contadores y contextos
de paquetes. Ningún puerto recibe estado de la referencia monolítica.
La referencia se usa sólo en el testbench.

Se separa el compromiso de estado de la validación de payload. El primer
experimento conserva las transiciones de la referencia R7, también tras
fallos, y publica los metadatos realmente usados en cada cabecera RC.
No es todavía un monitor integrado ni produce un certificado de vacío.
Sus fallos tempranos no sustituyen al conjunto completo de validaciones.

## Flancos, orden y propiedad

Antes del flanco N se resuelve RC por lane, incluidos EOP de contexto
retenido; después se resuelve RQ. Cada consulta RC observa el estado
anterior más la última escritura RC precedente del mismo beat. Las
escrituras RQ ocurren después de todas las consultas RC de ese beat.

En N se comprometen conjuntamente bitmap, tabla, contadores y contextos.
También se registran las cuatro respuestas de metadatos, bytes canónicos
entregados, pertenencia y secuencia. En N+1 el validador podrá consumir
esas respuestas junto con el beat capturado en N. La pertenencia previa
al flanco se expone por separado para la captura de la etapa RC existente.

El siguiente beat ve el estado comprometido en N. No se añade una FIFO,
no se inserta una burbuja y no se difieren escrituras para resolver
colisiones. Esta planificación debe medirse; su corrección funcional no
garantiza cumplir 4 ns.

La cabecera terminal libera el tag; su contexto y bytes originales viven
hasta EOP. Ese EOP modifica contadores, pero no vuelve a escribir metadatos
del tag liberado. Las escrituras del productor se resuelven antes del
flanco y no quedan escrituras pendientes que puedan alcanzar un tag
reutilizado. El futuro validador sólo produce diagnósticos. Si esta regla
cambia, será necesario un contrato nuevo de versiones y wrap.

La secuencia de 32 bits es diagnóstico con wrap explícito, nunca una clave
de propiedad. El reset frío borra tabla, contextos y respuesta registrada
en el mismo flanco; el reset de usuario y la pérdida de link no borran
obligaciones. El futuro retiro público debe alinearse con validación y
revocar vacío desde captura, incluyendo fallos y etapas ocupadas.

## Criterios de esta tanda

- Comparar todos los estados canónicos y las 256 palabras después de cada
  una de las 22.339 filas/68 escenarios, sin omitir estados con fallo.
- Comparar cuatro posiciones de respuesta por ciclo y sus contextos,
  incluidas las diferencias de [tráfico inválido](PCIE_INVENTORY_INTEGRATION_GAPS.md).
- Ejecutar omisión RQ/RC y corrupción de tabla/contexto como mutaciones reales.
- Reproducir las entradas binarias y sus salidas en el netlist propio.
  La comprobación jerárquica de las 256 palabras se limita al RTL.
- Medir síntesis a 4 ns, I/O explícitos y sin excepciones; conservar fallos.

RQ2/RC4, 256 tags, 257 lecturas y 1.052.672 bytes permanecen obligatorios.
Después se integrará el validador y se revisará el retiro completo según
el [contrato general](PCIE_INVENTORY_PIPELINE_CONTRACT.md).
