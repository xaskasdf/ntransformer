# Contador de admisiones C: síntesis propia, tanda 999

## Resultado medido

`20261011T215539Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-admission-count-synth-r1`
terminó con código 0 en **440,348 s**. La auditoría EDIF, revisión
independiente y comparación con RQ 984 están aprobadas y respaldadas.
Solo cambia `allocate_count`; E, B y retiro conservan la referencia.

| Métrica | RQ 984 | C 999 | Cambio |
| --- | ---: | ---: | ---: |
| LUT | 92.913 | 93.025 | +112 |
| FF | 27.560 | 27.531 | −29 |
| CARRY8 | 568 | 568 | 0 |
| WNS estimado, ns | −2,768 | −3,857 | −1,089 |
| TNS estimado, ns | −1.472,519 | −962,186 | +510,333 |
| Endpoints setup incumplidos | 1.439 | 1.475 | +36 |
| WHS estimado, ns | −0,075 | −0,075 | 0 |
| THS estimado, ns | −148,104 | −145,139 | +2,965 |
| Endpoints hold incumplidos | 2.068 | 2.026 | −42 |

Hay 71.007 endpoints, 150 menos que en RQ. Los doce controles de
cobertura temporal dan cero faltantes. No hay latches, BRAM, URAM ni
DSP. Se mantienen dispositivo, builds, reloj de 4 ns, retardos E/S cero
y `RuntimeOptimized/rebuilt`. El preflight comparó 529 filas y el
contador interno, además de las 31 sondas quiet.

## Ruta crítica y decisión

La peor ruta ahora parte del puerto **`rc_user[78]`** y termina en
`c_stage/registered_ops_reg[eops][0][mask][2]/D`: **32 niveles**, un
CARRY8 y 7,867 ns de datos, repartidos en 1,683 ns de lógica y 6,184 ns
de interconexión estimada. El reporte archivado atraviesa el selector RC,
`header_prefix_mask`, `selected_prefix_mask` y la máscara EOP registrada.
Los nombres describen la netlist resultante; no prueban una causa aislada
para el cambio de mapeo.

**Se mantiene RQ como referencia.** La mejora de TNS no compensa por sí
sola el peor WNS, los 36 endpoints setup adicionales y las 112 LUT extra.
Se completa el replay propio ya encolado y la comparación por etapas para
conservar un diagnóstico válido; no se promueve ni combina esta variante.
La matriz entre registros no incluye este origen de entrada: debe leerse
junto con el reporte global de E/S. No se cambia la restricción de entrada
para ocultar esa ruta.

## Evidencia y alcance

Archivo local/NFS: **35.336.837 bytes, 184 miembros**, SHA-256
`5e6eb2a0d4340a4ecc882bbef6fc0cf600068ced1b26d987b6029bb069d3fb11`.
Snapshot `source-20261011T215408Z.zip`:
`82c8377c8675717c43e57e7122c97a42b475773b1f87b359ccd7a27e0ad4ffe8`.
Revisión: `18f785c44f0a4d4a4aacdae6b26e40723d7b6720af468f068ea2707f189039de`.
Comparación: `25a99b0ef6edf3da454da6c76cc2cd0215f03f9596b101cd1a9e1c41b98b2964`.
DCP: `b36e5ff1b906be78531ad635fa3b729a230037f5cc7f3abc88f808548ba6d733`.
Netlist funcional: `43f13f10c468647f0acd0b7667d67e6a128f21c330be748c1236fcdc0899e564`.

Certificados bajo `artifacts/ntrpu-board-vivado/<tanda>/`; reportes y
revisiones en `build/cloud-setup/20261002-board/`. Los avisos emitidos se
revisaron individualmente; dos categorías alcanzan el límite de 100
mensajes del fabricante. La equivalencia binaria propia aún está pendiente;
estos resultados tampoco califican timing físico, throughput ni placa.
