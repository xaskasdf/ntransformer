# R: comprobar persistencia del fallo de rango de bytes

Plan posterior a la prueba corta R1 del 2026-10-09. **No ejecutado**.
El banco principal registró ocho fallos de rango actuales, pero ninguno
de los tres bancos cortos registró `sticky_hits`. Ese contador tampoco
separa un fallo previo de bytes de un fallo temprano de tags/lecturas.
La siguiente prueba debe cubrir específicamente esa distinción.

## Estímulo crudo propuesto

Usar `Vectors` y los constructores RQ/RC existentes; conservar los errores
esperados y las comparaciones independientes contra R7.

1. Después del reset frío y armado, emitir 256 lecturas con tags distintos,
   dirección cero y byte enables completos. Dos usan el count malformado
   2047; las otras 254 usan count 1024. R7 debe conservar sus transiciones
   posteriores al error de descriptor, tal como en los bancos anteriores.
2. Los spans esperados son `2 × 8188 + 254 × 4096 = 1056760` bytes.
   Tags y lecturas quedan en 256, dentro de sus límites; el total de bytes
   supera el límite 1052672. Comprobar esos contadores explícitamente.
3. Completar de forma terminal uno de los tags de 8188 bytes usando un RC
   de un DWORD. El terminal es deliberadamente prematuro y debe conservar
   su fallo correspondiente. Su contribución terminal retira 8188 bytes.
4. El total vuelve a `1048572`, dentro del rango. Comprobar que el bit 13
   siga activo, que la guarda actual de bytes esté inactiva y que el bit 13
   temprano de C/S1 permanezca cero. Añadir ciclos ociosos y cinco flancos
   de vaciado; verificar también el borrado por reset frío.

El modelo Python deja de ser una referencia de contadores al marcar
tráfico malformado como no confiable. Para los dos puntos de comprobación,
usar el cálculo cerrado anterior y el estado R7; no copiar expectativas
del candidato. Registrar las filas y orígenes exactos de los testigos.

## Controles y dependencias

Mantener R1, C/E/D, A/B y el validador tardío inmutables. Derivar un banco
con observaciones adicionales y su manifiesto de reemplazos. Un mutante
real debe eliminar solamente el bit 13 de los fallos previos al actualizar
el registro persistente; la discrepancia esperada es exclusivamente ese bit.
Esta prueba complementa los siete mutantes R ya preparados y todavía sin
ejecutar. La primera prueba corta conserva su condición de prueba parcial.

Después, completar los once bancos restantes: 159286 filas. Junto con las
23243 ya verificadas, cubren las 182529 filas de los catorce bancos con
vaciado de cinco ciclos, sin repetir los tres bancos aprobados si las
fuentes compiladas coinciden exactamente. Añadir el control de rango como
cobertura suplementaria; no sustituye ninguna fila original.

Siguen pendientes quiet, síntesis y netlist propias del conjunto, monitor
completo, cierre físico y los requisitos de [VIVADO_QUEUE.md](VIVADO_QUEUE.md).
