# RMSNorm Llama: pruebas nativas y auditoría RTL pendiente

Los 60 casos nativos pasan con MSVC y con GCC usando la biblioteca
matemática instalada de AMD. Ambos producen **78.208 valores FP32 a cero
ULP** respecto a la referencia Python ordenada. Son 44 operaciones
correctas y 16 errores esperados; no es todavía un resultado de Vitis C/RTL.

## Datos y alcance

Se usan los 33 vectores de normalización originales del GGUF Q8_0 de
Llama 3.2 1B Instruct, revisión `b69aef112e9f895e6f98d7ae0949f72ff09aa401`.
Las activaciones son sintéticas. El [contrato y la reproducción](LLAMA32_RMSNORM_R1.md)
fijan dimensiones, epsilon, orden FP64, redondeo, estados de error y tolerancia.
El fixture conserva SHA-256
`66573f15c4070efaf91d5baf1c3c64fb58441a9d84ad4790e115c8d8a9ee3e60`.

La prueba AMD compila sin fast-math y sin contracción de operaciones,
con afinidad de dos CPU. Registró hashes de 50 dependencias de cabeceras
y diez bibliotecas cargadas. Compilación: 4,17 s; ejecución positiva:
0,27 s. Son tiempos de software en esa VM, no latencias del circuito.
Una copia con una referencia alterada terminó con error, como se exigía.
El revisor numérico independiente rechazó siete corrupciones del transcript.

## Evidencia respaldada

Tanda: `20261012T012650Z-V7-llama32-rmsnorm-native-amd-r1`.
Resultado: `PASS_RMSNORM_NATIVE_AMD_R1`. El archivo está verificado en D:
y NFS, con SHA-256
`400d2de0cde9a77957e3056e719650fe0a23f100324eecab02d0bbfbee53efef`.
La revisión tiene SHA-256
`a0413472f7bbbf13f1dd57788be63eaa0f0bfff52b6927b286b8f5d79995e645`.
También se volvió a comprobar el log positivo desde el archivo cerrado.

## Próximos criterios

La tanda HLS espera el replay propio de máscaras de la cola recuperada;
preserva la FFN y las validaciones físicas en curso. Ejecutará C simulation,
síntesis y co-simulación Verilog para VU5P, con 5 ns solicitados y 2.400 s
de límite. Una estimación HLS no equivale a timing físico cerrado.

La auditoría suplementaria está preparada: después del PASS y respaldo
HLS leerá las 60 transacciones reales, estados de retorno, 78.208 palabras
numéricas y latencias desde el archivo. Su preflight leyó 317 transacciones
de un banco HLS histórico y rechazó 22 corrupciones. Las trazas construidas
a partir de resultados nativos prueban el comprobador, no califican RTL.
Las trazas de error vacías no prueban que la RAM haya conservado su contenido.

Quedan integración AXI/reset, atención, KV, logits y validación de inferencia
completa. Ninguna de estas pruebas mide tokens/s en una placa.
