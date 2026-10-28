# R1: comparación de su netlist funcional propia

## Contrato

Usar el Verilog funcional archivado en la síntesis R1 del 2026-10-09:
SHA-256 `d09195c02f2b491a7caeb8249060c1f27df670a9df3c7236501a59ca7eb9862a`.
La interfaz real tiene 111 puertos de entrada (2894 bits, incluidos reloj
y reset) y 18 de salida (463 bits). Derivar las conexiones de esas
declaraciones; no sustituir el candidato por RTL ni por otra netlist.

La netlist alimenta todas las salidas públicas comparadas con R7 por
origen. Un R1 RTL independiente comprueba además los 463 bits completos.
Las tres guardas internas permanecen como observaciones del RTL de
referencia; sus efectos públicos sí se comparan con la netlist.
Renombrar únicamente identificadores de módulos en catorce referencias,
con hashes y reemplazos registrados. Conservar R7/modelo/S1/C/E/D/A/B,
validación tardía, tablas completas y contabilidad de cancelaciones.

## Casos y aceptación

Ejecutar quince bancos: 174184 filas binarias seleccionadas entre 182806.
Excluir solo escenarios completos con entradas X/Z, aislados por reset
frío; mantener fallos y tráfico binario malformado. Conservar la prueba
RTL de cuatro estados por separado.

Aplicar 200 ns de estabilización GSR, muestreo 1 ns después del flanco y
cinco flancos finales de vaciado. Repetir los testigos de rango en sus
277 filas: pico 1056760, retorno 1048572, persistencia del bit 13 sin
enmascaramiento temprano y borrado por frío.

Seis controles alteran únicamente entradas/reset del candidato: omitir
control, omitir registro, omitir frío, cambiar secuencia, suprimir el paso
de bytes del origen 259 y suprimir la retirada terminal del origen 260.
Los dos últimos deben fallar en filas 264 y 265 respectivamente.

Revisar scripts realmente compilados, semilla, estados, cobertura,
advertencias y controles negativos antes de aprobar. Respaldar también
cualquier diagnóstico. Una simulación funcional correcta no resuelve el
WNS OOC de −3,096 ns ni acredita cierre físico o monitor completo.
