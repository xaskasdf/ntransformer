# Consumidor B con límites y winner: tanda 993

## Resultado

`20261011T201441Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-bound-winner-focus-r1`
terminó con código 0 en **167,849 s**. La revisión aprobó **309.884
muestras** sobre entradas B de 2.892 bits, comparando el agregado de
16 bits y 32 contribuciones de 16 bits antes de reducirlas.

Incluye 262.144 casos de límites, 10.240 de prefijos, seis dirigidos,
64 de lanes, 17.352 de desconocidos y 20.000 aleatorios. Los seis
mutantes divergieron en los primeros testigos previstos; el revisor
rechazó 21 corrupciones de evidencia. Productor registrado y consumidor
todavía requieren prueba conjunta: aquí los límites se calculan en el banco.

## Evidencia y respaldo

Resultados: **3.370.077 bytes y 363 archivos**, verificados en D: y NFS,
SHA-256 `f55545666b024e44bdf719c5e700a6cd317650944cb194c29c6f328d57eeedb7`.
El snapshot `source-20261011T201327Z.zip` tiene SHA-256
`c0bd45aa732106885321afece7b4099977b5b0cde23c64222a4dc2fd6bf348b3`.

La revisión `-independent-bound-winner-focus-review-r1.json` tiene SHA-256
`3ab1fbe79a257c643151aff2efd49505b9b5be28991ca6ca45301ce441e3d131`.
El coordinador R3 cerró revisión y respaldo a las 20:18:36 UTC del
2026-10-11, sin repetir Vivado. Registros bajo
`build/cloud-setup/20261002-board/` y certificados en
`artifacts/ntrpu-board-vivado/<tanda>/`.

## Decisión

El resultado habilitó la [integración completa B](PCIE_SYMBOLIC_MONITOR_PREDECODE_BOUND_WINNER_FULL_R1.md),
que comenzó como tanda `20261011T202003Z`. Esa prueba compara 447 bits
públicos y 75 bits internos de alineación en 183.058 filas, manteniendo E
original para aislar el efecto B. Síntesis y replay propio dependen de
su cierre revisado y respaldado. El PASS aislado no califica integración,
netlist, timing físico ni placa.
