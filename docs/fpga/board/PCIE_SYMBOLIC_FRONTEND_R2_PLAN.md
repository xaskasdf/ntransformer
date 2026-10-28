# Siguiente partición del frontend: selección y descripciones

Propuesta del 2026-10-09, **sin implementar ni medir**. El S1 R1 funcional
no cumple el período de 4 ns en síntesis OOC: WNS −6,566 ns. Su peor camino
va de `rc_user[76]` a `symbolic_rows_reg[104][5]/D`, con 10,576 ns y 41 niveles
lógicos. Los 7,678 ns de routing son una estimación sin placement; no un
resultado físico. Separar la evaluación numérica no cortó este camino.

## Hipótesis que se debe probar

Separar selección/control crudo, preparación de efectos sin acceso a la
tabla y consulta/composición/commit simbólico.
No repetir R1 cambiando únicamente estrategia de síntesis. La nueva frontera
debe eliminar la dependencia combinacional desde los punteros EOP hasta
el resumen de consumos almacenado. Esto **no garantiza** que cada mitad
alcance 4 ns: las tres requieren medida propia antes de fijar la profundidad.

| Frontera candidata | Estado y responsabilidad |
| --- | --- |
| C: control crudo | Propiedad de tags, contexto de control RQ/RC, admisión, framing, stalls y selección R3. Registra decisiones, beat y estado de control resultante. |
| E: efectos | Precompone los consumos ordenados para cada consulta y actualización a partir de máscaras, counts y condiciones capturadas. No lee remaining/dirección ni la tabla simbólica. |
| D: descripciones | Tabla simbólica y contexto de payload por valor. Procesa un registro E por ciclo, RC antes de RQ; selecciona la base y compone una sola vez con el efecto ya preparado. |
| Etapas posteriores | Aplicación numérica, validación y retiro de bytes; siguen pendientes. |

## Orden e identidad

D mantiene su propia tabla en el orden de los registros recibidos. Los tags
crudos son operandos de la operación de ese origen; nunca deben consultar
el bitmap vivo de C, que ya estará adelantado. Propiedad, máscara de prefijo,
selección del contexto retenido y prioridad de commit deben viajar con el
beat. Para la primera implementación, conservar exactamente las salidas
R3, incluidos sus 256 selects de commit: compactarlas sería otro cambio.

E debe incorporar el consumo del contexto retenido cuando la selección
R3 lo requiere. Su count y condición de entrega capturada viajan desde C;
no se reconstruyen desde el `owned` borrado en EOP. Para un commit RC,
precomponer también el consumo de la cabecera vigente. D no debe repetir
el árbol de cuatro consumos ni encadenar una segunda composición tardía.
La equivalencia de estas expresiones se deriva del álgebra ordenada ya
probada, pero su selección desde tráfico crudo requiere un banco nuevo.

El contexto de payload completo vive en D. Una cabecera captura allí una
copia de la descripción antes de reutilizar el tag. No sustituirla por un
puntero a una fila que pueda cambiar. RC y RQ del mismo registro conservan
su prioridad; el registro siguiente debe ver el commit del anterior.
El contador de secuencia sirve solo para diagnóstico.

Las decisiones EOP deben indicar el contexto/cabecera de su origen y la
elegibilidad original, incluida una contribución válida de cero bytes.
No inferirlas a partir de un bitmap posterior. Preservar alias de punteros,
entradas malformadas y X/Z, sin descartar filas después de un fallo.

## Cancelación y diagnóstico

Reset frío borra las tres etapas y cancela sus registros en el mismo flanco.
Reset de usuario y pérdida de link conservan obligaciones. Retener los
snapshots inactivos para diagnóstico, separados del bit que señala efectos
pendientes. El estado público y sus fallos deben corresponder al mismo
origen, con revocación inmediata de vacío desde las entradas actuales.

## Pruebas antes de integrar

1. Congelar los formatos C→E→D y un scoreboard contra S1 R1 con origen explícito.
   Comparar los 2.498 bits externos y las 256 filas en el origen correcto;
   no corregir diferencias añadiendo esperas al azar.
2. Ejecutar los seis bancos, incluidos los escenarios X/Z, y nuevos casos
   de reset con C/E/D ocupados, reutilización consecutiva y wrap. Inyectar
   omisión, duplicación, propiedad del ciclo equivocado y recarga del tag.
3. Medir C, E, D y el conjunto con 4 ns, I/O cero y sin excepciones; repetir
   sobre la netlist propia. Si D sigue largo, investigar la composición
   del resumen antes de añadir otra latencia de commit con bypass.
4. Solo entonces revisar la latencia total e integrar aplicación/validación.
   Mantener RQ2/RC4 por ciclo, 256 tags, 257 lecturas y 1.052.672 bytes.

Los gates físicos y de placa de [VIVADO_QUEUE.md](VIVADO_QUEUE.md) permanecen
abiertos. Esta propuesta no convierte el resultado aislado de S1 en una
prueba del monitor completo.
