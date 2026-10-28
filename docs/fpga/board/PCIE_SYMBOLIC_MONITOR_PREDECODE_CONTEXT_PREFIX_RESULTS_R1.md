# Resultado aislado del contexto por prefijo

## Resultado observado

La tanda 964, `20261011T125805Z-V4-ffn-supervised-pcie-symbolic-late-fault-leaves-prefix-r1`,
aprobó `PASS_SYMBOLIC_LATE_FAULT_LEAVES_PREFIX_DIRECT_R1`.
La revisión y los respaldos en D: y NFS finalizaron el 2026-10-11 a las
13:01 UTC. Se compararon **47.734 muestras** contra la validación tardía
original y la variante por hojas ya aprobada.

El banco incluye 10.240 combinaciones dirigidas de SOP/EOP, contador y
activo 0/1/X/Z, además de 64 casos de payload, 17.352 casos de desconocidos
y 20.000 registros pseudoaleatorios. Compara 32 vectores de 16 bits por
lane antes de reducirlos con OR, además del resultado agregado. Hubo
20.397 muestras con fallo cero y 27.337 con algún fallo.

## Controles que detectaron errores

Los seis mutantes reales divergieron exactamente en los puntos previstos:

| Alteración | Primera muestra | Comparación |
| --- | ---: | --- |
| Contador RC incrementado | 1 | Agregada, `0010 → 0410` |
| Contador RQ incrementado | 6.645 | RQ lane 1, `0400 → 0410` |
| SOP RC gana el empate con EOP | 141 | RC lane 3, `0400 → 0000` |
| SOP RQ gana el empate con EOP | 5.261 | Agregada, `0400 → 0410` |
| Se pierde poison RC | 140 | Agregada, `0000 → 0410` |
| Abort RQ forzado | 65 | Agregada, `0010 → 0000` |

Dos discrepancias requieren la comparación por lane: el OR agregado puede
ocultarlas. El revisor rechazó además quince corrupciones de evidencia y
verificó ocho fuentes SV más `glbl` por cada uno de los siete proyectos.
Los únicos siete avisos corresponden al entorno del compilador XSIM.
Las siete ondas están archivadas; no se afirma haber interpretado su contenido.

## Evidencia y siguiente dependencia

Archivo terminal: **3.091.903 bytes**, 356 archivos; SHA-256
`e0f65e4e55f1c77a7199df35b05dd65b4823a3d898d7a79ba0642a730d744e83`.
Fuentes SHA-256:
`b9c9e2fcb27e854939d46dc4f480f255afe5575fe739207e4d111fd11adbd921`.
Revisión SHA-256:
`cbe633e77ff48f259761c1263e0a2e9f5e08e7aab235338889d784ba5dac648d`.

La aprobación habilitó la integración de 183.058 filas nuevas, tanda
`20261011T130303Z`, en curso al registrar este resultado. Después siguen
su síntesis y replay propios. Este ensayo aislado no demuestra todavía
mejora de timing, equivalencia exhaustiva de cuatro estados ni viabilidad
física de la placa. [Diseño y reglas de contexto](PCIE_SYMBOLIC_MONITOR_PREDECODE_CONTEXT_PREFIX_R1.md).
