# Límites registrados en B: resultado y revisión R2, tanda 988

## Resultado

`20261011T185835Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-b-focus-r1`
terminó con código 0 en **144,233 s**. Aprobó **211.606 filas nuevas**,
comparando los **2.967 bits** de B y sus cinco límites de payload antes
y después del flanco. Incluye 131.072 casos aritméticos, 65.536 casos de
controles, 10.892 casos de bits 0/1/X/Z, 4.096 registros pseudoaleatorios
y reset frío 0/1/X/Z. El B original permanece como referencia independiente.

Los seis mutantes reales divergieron: B atrasado, consulta rotada, offset
omitido, límite truncado a 13 bits, desconocido borrado y reset omitido.
La revisión R2 rechazó **20 corrupciones de evidencia**.

## Corrección del revisor

R1 esperaba el fallo del mutante `ignore_cold` en el reset dirigido de
la fila 4. El banco empieza con reset activo: el mutante ya falla en
la **fila 0, fase 1**, con los 2.967 bits X frente a cero. El positivo
sí cumple ese reset. Se conservan los hashes de fuentes, compilación,
logs y archivo original; no se repitió Vivado ni se modificó el RTL.

R2 exige exactamente ese primer testigo y el vector completo X→0.
Mantiene los otros cinco testigos y los 17 controles anteriores; añade
tres corrupciones del testigo de reset. La revisión R1 fallida y los
coordinadores detenidos se conservan. La [recuperación de cola](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_B_QUEUE_RECOVERY_R2.md)
usa coordinadores nuevos para continuar.

## Evidencia y límites

Archivo de **656.407 bytes, 342 archivos**, verificado en D: y NFS,
SHA-256 `aac78d522a8c13cfe3c19db32251b917970b4606b869c25ed2a6062958775def`.
Snapshot `source-20261011T185717Z.zip`, SHA-256
`7567ca8e550bcf527feea9f47ba68de58f069fcf38ca97e53a1284d513eeaea6`.
Revisión `-independent-bound-b-focus-review-r2.json`, SHA-256
`95ddebffea995a536046bd0da0d96870b4cc68b475cea8b315aabab9d21592e5`.
El cierre revisado y respaldado se registró a las **19:07 UTC**.

Se habilita el ensayo aislado del consumidor tardío. Todavía faltan su
conexión al productor, regresión del monitor, síntesis y netlist propias.
No se atribuye una mejora temporal a estos 75 bits adicionales.
