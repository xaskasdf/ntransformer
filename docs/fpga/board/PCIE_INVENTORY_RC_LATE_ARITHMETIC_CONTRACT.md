# Contrato: selección tardía de la aritmética RC

R11 parte de R8 y conserva el selector R3, `newest` de 32 bits, la máscara
dinámica, RQ, propiedad, capacidad y latencia. Ejecutar sólo después de
cerrar R10 y verificar el punto de recuperación de R96.

## Cambio aislado

Renombrar la función aritmética anterior como `rc_prefix_constant`, sin
cambiar su cuerpo salvo el nombre de retorno. La nueva `rc_prefix_state`
calcula sus 16 resultados para máscaras constantes y selecciona después
el correspondiente a la máscara efectiva. No imponer excepciones de timing.

Construir cada bit efectivo con `if (mask[bit])`: X/Z no habilitan una
operación. Conservar signos, conteos cero desalineados, desbordamientos,
propagación de entradas utilizadas desconocidas e independencia de
cantidades no utilizadas. La nueva selección no debe añadir estado,
ciclos, pausas ni entradas procedentes de la referencia.

La estructura RTL expresa una hipótesis de mapeo; Vivado puede compartir
lógica. Revisar recursos y rutas sintetizados antes de atribuir una mejora.

## Validación completa

Comparar el texto de R11 contra R8 para comprobar el alcance exacto del
cambio. Mantener el selector R3 por hash. El banco
`tb_ntrpu_pcie_rc_prefix_mask_four_state.sv` debe conservar las 45.056
comprobaciones aritméticas originales, incluidas 6.624 X/Z, y ejecutar
2.048 adicionales: 1.920 con máscara desconocida y 2.000 con alguna
entrada desconocida. Exigir compilación de ese top, finalización y rechazo
de controles de evidencia alterados. La referencia sigue siendo la
recurrencia secuencial independiente.

Conservar también los bancos de ascendencia 1.153.296 y 32.768,
selector 89.632, RQ 615.536 y 8.850, propiedad integrada 46.356,
prefijos integrados 544, principal 22.339/68 y siete mutantes.
Comparar tablas internas en RTL y todas las filas públicas posteriores
a fallos. No recortar ni sustituir casos heredados.

Exigir síntesis VU5P FLVA2104 -2-i, `RuntimeOptimized`, 4 ns sin excepciones,
DCP acíclico y replay de la propia netlist: 22.330/67 más los tres
suplementos, sin compilar RTL del DUT. Medir área, setup/hold y ruta crítica
frente a R8, R9 y R10. Mantener XSIM sin captura automática de ondas y
respaldar cada resultado, incluidos fallos, localmente y en NFS.

La [cola general](VIVADO_QUEUE.md) mantiene pendientes los criterios
nativos, DDR de 512 GiB, inferencia real y diseño físico/ECAD.
