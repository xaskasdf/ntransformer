# Resultado aislado de la bandera de contexto

## Resultado revisado

La tanda 968 `20261011T140543Z-V4-ffn-supervised-pcie-symbolic-late-fault-leaves-winner-r1`
terminó con exit 0 en 149,219 s. Pasaron 47.734 muestras sobre 2.892 bits
de entrada, comparando la salida original y los 32 vectores de 16 bits
por lane contra el padre por prefijo antes del OR.

Incluye 10.240 fronteras SOP/EOP, 17.352 muestras con bits X/Z, 20.000
registros aleatorios y 64 casos de payload desconocido. Hubo 20.397
salidas cero y 27.337 no cero; estado aleatorio final `d8abb5e0`.
Los seis mutantes divergieron en los puntos previstos: contadores RC/RQ,
prioridad SOP/EOP RC/RQ, pérdida de poison y aborto forzado. El revisor
rechazó quince corrupciones de evidencia.

## Evidencia

El archivo terminal de 3.201.341 bytes y 357 archivos quedó verificado
en D: y NFS. Los siete WDB se conservaron por hash, sin interpretar sus
ondas. Se comprobaron las ocho fuentes SV realmente compiladas más glbl
en cada uno de los siete proyectos.

| Artefacto | SHA-256 |
| --- | --- |
| Archivo terminal | `5551e5a7b7c59a499661bb8a43a3368767503fbc58876a9a81566938951de6e0` |
| Fuentes | `85b9daf0c29205face0d70e3a86338b237ea181a49789605c941d1c39a0a3f9e` |
| Evidencia | `d8c7207a8dac6f7bbb2b4585322299936971f3b024d4df17451c3742f94c8c24` |
| Revisión | `bcc1b7d020a33bc76af7f857ca79cd204507d259a33735c10cce7a2ae3f5f5e6` |

La revisión conserva el sufijo histórico `-independent-symbolic-late-fault-leaves-prefix-review-r1.json`;
su contenido, run, configuración y hashes identifican inequívocamente la
variante winner. La cola R1 se detuvo al esperar otro nombre, antes de
crear una tanda integrada. La recuperación R2 enlaza este archivo exacto
y conserva intactos los resultados y estados R1.

## Siguiente validación

La integración `20261011T141336Z` ejecuta las 183.058 filas RTL nuevas.
Síntesis y replay propios dependen de su revisión y respaldo. El ensayo
aislado y la [prueba formal](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_WINNER_R1.md)
no demuestran todavía mejora de timing ni califican el monitor completo.
