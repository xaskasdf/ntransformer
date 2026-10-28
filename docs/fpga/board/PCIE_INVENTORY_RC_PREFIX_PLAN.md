# Próximo candidato: prefijos de actualizaciones RC efectivas

Estado: **modelo aritmético binario comprobado; transformación RTL pendiente**.
La separación previa de controles se registra en
[el contrato R4](PCIE_INVENTORY_RC_SELECTION_CONTRACT.md). Su síntesis revisada
reduce 6.013 LUT (42.176 frente a 48.189), pero empeora WNS de −9,877 a
−10,047 ns. El productor original sigue siendo la referencia preferida de
timing; R4 aporta una separación de controles para construir el experimento
de prefijos. Exigir su replay propio antes de extenderlo y comparar el
nuevo resultado contra ambos. No combinar este cambio con el contador R3.

## Evidencia disponible

`tools/check_ntrpu_pcie_rc_prefix_math.py` comparó la recurrencia original
de 15 bits con la fórmula de prefijos: **67.108.864 casos** cubren todo
`R[12:0]`, desplazamiento de dirección y `count[10:0]` para un fragmento.
También pasaron **2.097.152 pares límite** y **1.000.000 de prefijos muestreados**
en cadenas de una a cuatro actualizaciones, incluyendo wrap de dirección.
Dos contraejemplos rechazan omitir el caso cero/desalineado y truncar la suma
de counts a 11 bits. El resultado local es
`build/cloud-setup/20261002-board/pcie-inventory-rc-prefix-arithmetic-model.json`;
el hash del verificador es
`5b941ecea54573597afccca47433eff1b51a60928c277750c36edb0ca3040a3e`.

Esta cobertura no es una prueba exhaustiva de cadenas ni valida eventos,
X/Z, RTL, netlist o timing. El selector R4 tiene su banco propio de 89.632
casos binarios; tampoco sustituye la validación del monitor integrado.

## Construir la dependencia de eventos

Extender la selección previa con la procedencia de cada actualización:
encabezado 0–3 o contexto retenido al entrar al beat. Para cada encabezado,
identificar la última escritura previa de su tag y seguir únicamente esa
dependencia. Un encabezado sin EOP no aporta una actualización de tabla.
Un SOP que reemplaza un contexto sin terminar conserva el comportamiento
malformado original; no se deben sumar todos los counts de encabezados del tag.

Cada cadena debe identificar su semilla y los counts de sus escrituras
efectivas. Si no hay escritura previa, la semilla es la tabla propia. Si
la primera escritura procede del contexto retenido, aplicar literalmente
`rc_original-rc_delivered` en 13 bits y `rc_address+rc_delivered` en 12 bits.
Después de fallos, esos registros pueden discrepar de la tabla; no reconstruir
la semilla a partir del descriptor ni asumir que el delta cabe en el restante.

La cadena de controles, tags y selección de semilla no debe leer los bytes
calculados. Cada consulta canónica, delta y dato final se obtiene directamente
desde su semilla y su prefijo; comprobar que no reaparece la misma cadena de
restas saturadas bajo otro nombre.

## Aritmética y semántica

Una cadena vacía devuelve la semilla. En una cadena binaria normal y no
terminal, si su primer count es cero y la dirección inicial no está alineada,
el delta total es el restante inicial. En los demás casos:

`delta = min(R, 4*sum(counts) - (A & 3))`.

Cuatro counts necesitan 13 bits de suma; el término en bytes debe representar
32752 y su resta de desplazamiento sin truncamiento prematuro. Obtener el
restante como `R-delta` y la dirección módulo 4096. Para publicar un delta
individual, comparar los prefijos anterior y posterior correspondientes al
mismo evento; conservar el orden y los anchos de las sumas globales.

El RTL existente usa comparación mixta con y sin signo y ternarios de cuatro
estados. La identidad binaria no autoriza a cambiar la propagación de X/Z
probada. Añadir casos dirigidos con desconocidos en count, dirección, restante,
propietario y contexto retenido; resolver las diferencias en el candidato,
sin filtrar filas fallidas ni sustituir entradas por valores canónicos esperados.

## Criterios para avanzar

Mantener el banco completo de 22.339 filas/68 escenarios y todas las palabras
internas. Añadir cadenas del mismo tag, agotamiento intermedio, ceros alineados
y desalineados, SOP sin EOP, EOP repetidos/fuera de orden, semilla retenida
inconsistente, terminal seguido de reutilización RQ, reset y estados posteriores
a fallos. La selección, los datos, el delta global y los metadatos deben
coincidir; pasar solamente la salida final no es suficiente.

Conservar 256 tags, RQ2/RC4, 257 lecturas, 1.052.672 bytes, latencia pública,
cancelación cold reset y vacío inmediato/eventual, sin pausas. Después del
RTL, sintetizar el monitor completo con las restricciones OOC vigentes,
auditar ciclos y reproducir su netlist propia contra 22.330 filas/67 escenarios
binarios y los nuevos casos aplicables. Registrar regresiones de recursos o
timing; no promover el candidato por pasar solamente la función aritmética.

Las obligaciones nativas, identidad/BDF, CDC/reset de 50 ms, DIMM exacto,
cuatro DDR/512 GiB, pesos reales, ECAD y cierre físico permanecen intactas.
