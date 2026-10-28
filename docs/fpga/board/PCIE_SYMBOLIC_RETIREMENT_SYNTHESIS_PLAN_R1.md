# R1: síntesis aislada y reproducción de su netlist

## Dependencias

Ejecutar después de revisar y respaldar la regresión completa de R1:
182529 filas originales, 277 suplementarias y nueve mutantes detectados.
Reutilizar sus fuentes exactas; conservar las pruebas de cuatro estados.
No cambiar la implementación para obtener una cifra de temporización.

## Medición

Primero repetir el banco dirigido de 277 filas contra R7, incluidos el
pico de 1056760 bytes, retorno a 1048572 con fallo 13 persistente y reset
frío. Después sintetizar únicamente `ntrpu_pcie_symbolic_retirement_r1`,
incluidos su validador tardío y las cuatro guardas de cabecera.

Usar `xcvu5p-flva2104-2-i`, `RuntimeOptimized`, jerarquía `rebuilt`, dos
hilos y modo fuera de contexto. Aplicar durante síntesis un reloj de 4 ns
y retardos de entrada/salida de 0 ns, sin excepciones de caminos.
La entrada B contiene 2892 bits; las salidas incluyen control, bytes,
fallos, validez, secuencia y etiqueta de obligación del origen retirado.

Exportar DCP, EDIF, Verilog funcional, XDC aplicado, inventario de
primitivas y reportes de recursos, setup/hold, relojes, comprobación de
temporización, DRC, metodología y excepciones. Revisar advertencias
individualmente y comprobar el grafo combinacional del DCP archivado.
Conservar resultados negativos como diagnósticos.

## Aceptación y siguiente paso

Respaldar fuentes y resultados con hashes verificados en D: y NFS.
Una síntesis correcta permite preparar reproducción binaria de **esa
netlist propia**, comparada con R1 y con los estados R7 de origen. Debe
conservar fallos binarios, vaciados y contabilidad de cancelaciones; no
eliminar selectivamente tráfico problemático. La simulación RTL mantiene
la cobertura X/Z que no representa una netlist binaria.

Esta medición no caracteriza A/B/R juntos ni el monitor completo. No
sumar resultados aislados para anunciar recursos o frecuencia conjunta.
Siguen pendientes quiet inmediato y conservador, integración completa,
mediciones conjuntas, place/route y las validaciones nativas y de placa.
