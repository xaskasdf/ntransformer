# Replay propio de prioridad RC: tanda 980

## Resultado

`20261011T170725Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-commit-priority-binary-r1`
terminó con salida 0 en **1.320,605 s**. La revisión independiente aprobó
174.436 filas binarias en 16 bancos, comparando 447 bits públicos;
detectó los seis controles negativos de entrada y 34 corrupciones de
evidencia. Las diez sondas binarias aportaron veinte muestras de
perturbación/restauración. No se reutilizaron filas aprobadas de otra
variante.

El netlist simulado procede de la síntesis 979 del candidato RC:
`91857b38f112cc383b2f16d1e97e4087e9d312b6625d18afacb5eb2cd446049b`.
Las referencias y funciones de contabilidad de 971 se conservaron. La
comprobación es funcional binaria, sin SDF; no extiende la aprobación
a X/Z de netlist ni demuestra timing físico.

## Evidencia y respaldo

Archivo de resultados: **80.073.714 bytes, 294 archivos**, verificados
en D: y NFS. SHA-256:
`a63d86cefc8aad14aedf36027c06d93fc39279d97d4de05110d327b8ea255e4c`.

La revisión `-independent-symbolic-monitor-predecode-commit-priority-binary-review-r1.json`
tiene SHA-256
`919c86ac84c59f9b61e1b06537359ef79bec67a4ef54691b3edda17381b91919`.
El snapshot `source-20261011T170544Z.zip` conserva SHA-256
`c7f193258265bf655aed9c4f202718e04b216f34816ffc9d18985316af6a54c9`.
Los certificados se encuentran bajo el mismo ID en
`artifacts/ntrpu-board-vivado/`; el almacenamiento local está en D:.

## Decisión

Se habilita la lectura comparativa por etapas de los DCP 979 y 970,
sin optimización ni ruteo. Permanece la conclusión de síntesis:
**91.660 LUT y WNS estimado −3,704 ns**; la funcionalidad aprobada no
convierte ese incumplimiento en cierre de timing. El experimento RQ
continúa separado, con sus pruebas propias antes de integrarlo.
