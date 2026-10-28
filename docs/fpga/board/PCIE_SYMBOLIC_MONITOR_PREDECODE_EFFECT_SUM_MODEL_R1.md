# Etapa E: suma de efectos — modelo R1

La tanda optimizada 974 conserva una ruta C→E de −1,182 ns:
`out_record.rc_data[32]` → `out_e.updates[0].summary[10]/R`, con
19 niveles y siete CARRY8. Es una estimación sin colocación/ruteo.
La revisión identifica el endpoint de reset; su nombre no autoriza una
excepción temporal ni atribuye por sí solo cada celda a una expresión RTL.

## Hipótesis aritmética

`prepare_operation` combina hasta seis efectos de consumo: contexto
retenido, cuatro prefijos y consumo del EOP. Cada palabra tiene 11 bits.
El código actual compone los cuatro prefijos por parejas, incorpora el
retenido y finalmente el consumo añadido. Cada composición limita el
total a 2.049 y el número de consumos a dos.

Para efectos sin reemplazo de metadatos, una alternativa es calcular
la suma completa con 14 bits y saturarla una sola vez. El máximo de seis
operandos es 12.282. Deben conservarse, además, el primer operando activo,
el número de consumos saturado y la bandera de palabras desconocidas.
Si esta última está activa, los bits de suma continúan siendo cero.

## Evidencia y límites

Z3 no encontró diferencias en esos cuatro campos entre ambas
expresiones, bajo el modelo de seis activaciones booleanas y palabras
de 0–2.047. También probó el límite de 12.282. Los tres controles que
cambian la saturación, omiten el consumo final o ignoran desconocidos
sí produjeron contraejemplos.

Prueba `pcie-symbolic-monitor-predecode-effect-sum-r1-proof.json`, SHA-256
`c93201148734f23b24c4947a1dbf11c34cb76b1d4416982f2fd4a5de0a7c95d9`.
Está ligada por hash a las fuentes actuales y al entorno Z3 conservado.

Esto es una identidad del modelo, no una prueba de SystemVerilog con
X/Z. No se creó ni encoló RTL para E. Una futura implementación deberá
respetar las ramas procedurales, `retained`, `tag`, `apply_summary`, el
orden del primer operando y los valores desconocidos del primer word.
Exigirá comparación aislada, regresión integrada y medidas propias.
Mantener este experimento separado de la prioridad RC en curso.
