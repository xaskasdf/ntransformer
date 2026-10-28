# Monitor PCIe predecode R1: regresión RTL completa

La variante E/D completó **183.058 filas en dieciséis bancos** contra la
referencia original. El resultado combina las 9.948 filas del
[filtro revisado](PCIE_SYMBOLIC_MONITOR_PREDECODE_GATE_R1_RESULTS.md) con
173.110 filas adicionales, usando exactamente los mismos 22 archivos SV.
La segunda tanda duró 494,667 s y terminó con código 0.

## Cobertura comprobada

| Grupo | Filas |
| --- | ---: |
| Filtro: monitor, directed, controls, range_return y rq | 9.948 |
| main, ownership, prefix, lanes y context originales | 71.270 |
| Seis bancos con drenaje completo | 101.840 |
| **Total** | **183.058** |

Se compararon **447 bits públicos**, antes y después de cada flanco,
con `!==`: 183.138 muestras por fase, incluidos cinco ciclos finales
por banco. Permanecen los valores X/Z originales, los oráculos R7/S1,
la inspección de las 256 filas en la referencia y la contabilidad por
origen. El candidato se compara por sus puertos públicos. Los seis bancos
drenados cancelan **cero filas de tráfico original**.

El filtro conserva 31 sondas entre flancos y sus 31 restauraciones, además
de 27 mutantes reales. La revisión completa rechazó 32 corrupciones de
evidencia; el filtro había rechazado 34 y otras seis específicas del primer
testigo de `omit_raw_rc`. Los 22 avisos de esta tanda corresponden a los
dos diagnósticos previstos por banco. No hubo errores ni avisos críticos.

## Evidencia y alcance

- Tanda nueva: `20261011T080044Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-full-r1`.
- Archivo verificado en D:/NFS: `fa68928c7b9d6e38072cf4a69fb13c8fd64b05b9d06c4d1a2975702d64a0e1f1`.
- Fuente compartida con el filtro: `eab2cfb34c3a257674988f3dd7adfd2234c8879ef196b19a4a6e5f561754f6b7`.
- Revisión independiente: `b98e3afafade1b852ea6a534a645d03412b99cebe62808dbc1a756eb5c565422`.

Esto habilita la síntesis propia del candidato. No demuestra mejora de
timing, equivalencia formal, funcionamiento PCIe nativo ni inferencia.
Siguen pendientes recursos/temporización de síntesis, replay de la netlist
propia, implementación física y las dependencias de integración de placa.
