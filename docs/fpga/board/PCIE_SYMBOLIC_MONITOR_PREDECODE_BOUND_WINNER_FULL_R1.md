# Conexión de límites B y retiro winner

## Cambio y comparación

La variante parte del monitor RQ 983/984. Conserva C, E original, D, A,
puertos y cinco relojes de latencia. B registra cuatro límites de consulta
y uno de contexto junto con su registro original; R los entrega al
consumidor winner. Se añaden 75 bits registrados sin otra etapa de pipeline.

La integración espera el PASS del productor B 988 y el nuevo consumidor
winner. El consumidor por prefijo 989 constituye evidencia separada.
Las mejoras de suma E no se incorporan aquí: una futura comparación
contra 984 podrá atribuir el cambio a los límites registrados.

## Pruebas y aceptación

Ejecutar **183.058 filas nuevas** en los 16 bancos completos, comparando
447 bits públicos antes y después del flanco, con 31 sondas de quietud,
seis bancos drenados sin cancelación de tráfico y seis controles negativos
de entrada. Conservar los 34 controles de evidencia ya calificados.

Además, recomputar independientemente los límites mediante sumas de
32 bits desde el B capturado y comparar los **75 bits en la entrada de R**
en cada fase, incluidas las perturbaciones entre flancos. Esta comprobación
evita ocultar desalineación detrás de fallos públicos previamente acumulados.
Exigir contadores propios y ocho controles de corrupción de sus resúmenes.

Compilar 27 fuentes: el B original sigue siendo necesario para las
referencias, además del productor candidato. No sustituirlo globalmente.

## Cola, respaldo y límites

El coordinador espera el cierre completo de la cadena R3 y el PASS
winner; conserva las tandas existentes. Cada ejecución usa snapshot nuevo,
1.800 segundos, dos hilos, revisión independiente y archivos verificados
en D: y NFS. La cápsula R13 incluye sus futuros snapshots.

La receta de fuentes R2 corrige el campo de filas del resumen antes de
congelar o ejecutar; mantiene la comparación y los vectores de R1.
Un PASS habilitará síntesis propia y replay. No acredita timing ni placa.
