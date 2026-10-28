# Distribución por conectividad de los comparadores de bancos

## Objetivo

Ensayo del 6 de octubre de 2026 UTC, posterior al
[Pblock único que agotó el ruteo](PEER_RX_OVERLAP_FLOORPLAN.md). Se conservan
el RTL `compact`, el intervalo N+4, ocho bancos y las restricciones de
200 MHz. Se evalúan grupos pequeños por banco, manteniendo libre la lógica
compartida. La geometría es una hipótesis de colocación, no una predicción
de timing.

## Selección por conexiones

La extracción lee el checkpoint ruteado verificado
`20261006T030021400903Z-route` y no lo modifica. Exporta primitivas y
9.493 pines de entrada de 1.431 celdas combinacionales del asignador.
El análisis sigue las conexiones desde los registros de estado de cada
banco y los dos buses de generación entrantes. Selecciona lógica alcanzada
por un bus entrante que depende de un solo banco. Excluye combinaciones
entre bancos y dependencias combinacionales externas no examinadas.

El resultado tiene **954 celdas**: 528 registros de estado y 426 celdas
comparadoras o derivadas, distribuidas entre ocho grupos. Un segundo
recorrido comprueba los registros de origen de cada celda seleccionada.
Los 128 registros de generación entrantes quedan sin asignar a Pblocks,
porque alimentan comparadores de varios bancos.

| Banco | Celdas | Región inicial SLICE |
| --- | ---: | --- |
| 0 | 119 | X57Y248:X62Y267 |
| 1 | 118 | X65Y272:X70Y291 |
| 2 | 119 | X65Y248:X70Y267 |
| 3 | 118 | X73Y248:X78Y267 |
| 4 | 121 | X57Y272:X62Y291 |
| 5 | 120 | X49Y272:X54Y291 |
| 6 | 120 | X73Y272:X78Y291 |
| 7 | 119 | X49Y248:X54Y267 |

Las regiones iniciales son disjuntas y tienen separaciones de dos columnas
y cuatro filas. La asignación minimiza desplazamiento agregado en
coordenadas SLICE entre ocho posiciones candidatas; esas unidades no son
retardos eléctricos. `IS_SOFT=TRUE`, `EXCLUDE_PLACEMENT=FALSE` y
`CONTAIN_ROUTING=FALSE` permiten salir de las regiones y compartir recursos.
AMD recomienda evitar [Pblocks superpuestos al tratar congestión](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Addressing-Congestion).

## Validación y estado

La colocación parte de la síntesis original de `compact`. Se mantiene
`xcvu5p-flva2104-2-i`, período de 5 ns, incertidumbres de setup/hold de
0,100/0,050 ns y BUFGCE en `BUFGCE_X0Y60`. Se revisan hashes, conectividad,
miembros de cada región, recursos, cobertura interna, DRC y rutas completas.
Un ruteo terminado requiere además setup, hold y ancho de pulso válidos
para acreditar timing interno. Después se comparan las 28 salidas con
el RTL en los 16 escenarios establecidos, sin compensar latencia.

La colocación `20261006T042105519153Z-place` termina en 330,890 s:
setup preliminar +0,388 ns, hold −0,170 ns y ancho de pulso +1,958 ns.
Usa 6.789 LUT, 16.022 registros y 15 bloques BRAM. Los 954 miembros y
sus tipos se conservan, pero sólo 449 permanecen dentro de las regiones
sugeridas; 505 quedan fuera, permitido por `IS_SOFT`. Son resultados
anteriores al ruteo, sin acreditación de timing.

Al crear las regiones, Vivado normaliza los límites derechos X62 a X61
y X78 a X77 en los bancos 0/4 y 3/6. Las regiones aplicadas tienen
100 o 120 sitios SLICE. El revisor admite únicamente esa contracción
de hasta una columna y exige igualdad entre la geometría aplicada
antes y después de implementar. Comprueba capacidad, miembros y
restricciones de reloj. Cinco controles negativos rechazan una región
desplazada, un cambio posterior de geometría, una región rígida, un
miembro ajeno y una relajación del reloj.

El ruteo `20261006T043115634715Z-route` termina en 291,573 s y pasa
`ROUTED_INTERNAL_TIMING_PASS`: **setup +0,054 ns, hold +0,006 ns y pulso
+1,958 ns**. Las 17.827 redes están completamente ruteadas, sin errores.
TNS y THS son cero. Recursos y miembros coinciden con la colocación.
La estimación final del ruteador era +0,034 ns; se adopta el STA final
de +0,054 ns, sin mezclar ambos informes.

Frente al mejor `compact` anterior de −0,244 ns, mejora el peor setup
en 0,298 ns, con cuatro LUT adicionales, dos registros menos y la misma
BRAM. La tanda `20261006T043637637991Z-netlist` termina en 155,778 s:
**pasan los 16 escenarios y las 28 salidas**, con 5.247 comparaciones
ciclo a ciclo, 129.088 bytes, 92 trabajos completados y cinco cuarentenas
esperadas. Los 13 controles negativos de referencia y las siete
eliminaciones de trazas ruteadas son rechazados. Es comparación funcional
sin SDF, no equivalencia formal ni simulación con retardos.

Se selecciona este checkpoint como nueva referencia experimental
`compact` N+4 con cierre interno a 200 MHz. Se conservan la referencia
N+7 acreditada y los checkpoints anteriores, sin sustituir RTL fuente
ni alterar la cola de validación de placa.

La peor ruta va de `active_generation_reg[1][14]` al enable de
`bank_state[5].generation_reg[5][52]`: 12 niveles, 4,542 ns de datos,
de los cuales 3,402 ns corresponden al cableado. `grant_5` tiene fanout
132 y 0,951 ns de retardo. La mejora de esta implementación no demuestra
que cada región individual sea beneficiosa: seis bancos tienen miembros
fuera de sus regiones, y los bancos 1/5 quedan completamente fuera.
El margen de 54 ps es pequeño; integrar interfaces y otros bloques
requiere una nueva implementación y su propio STA.

## Reproducción y evidencia

```bash
source /ruta/Vivado/settings64.sh
# Exportar conexiones y metadatos de un checkpoint con hash conocido.
python3 -B tools/run_ntrpu_overlap_connectivity.py /probe/output \
  --checkpoint /referencia/physical_route.dcp --expected-sha256 HASH_VERIFICADO
python3 -B tools/prepare_ntrpu_overlap_bankplan.py /probe/output /plan
cp -a /probe/output /plan/probe_evidence
python3 -B tools/run_ntrpu_overlap_bankplan.py /place/output \
  --stage place --prior /compact_synth/output --plan /plan
python3 -B tools/check_ntrpu_overlap_bankplan.py /place/output --review /place/REVIEW.json
python3 -B tools/run_ntrpu_overlap_bankplan.py /route/output \
  --stage route --prior /place/output
python3 -B tools/check_ntrpu_overlap_bankplan.py /route/output --review /route/REVIEW.json
# Sólo con ruteo terminado y revisado:
python3 -B tools/run_ntrpu_overlap_bankplan.py /netlist/output \
  --stage netlist --prior /route/output
python3 -B tools/check_ntrpu_overlap_bankplan.py /netlist/output --review /netlist/REVIEW.json
```

Cada etapa se revisa y respalda antes de continuar. Se usan dos CPU,
prioridad baja y 900 s por etapa física; la extracción se limita a 300 s.
El retorno correcto del revisor indica informe generado; para timing se
exige expresamente `ROUTED_INTERNAL_TIMING_PASS`.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-overlap-bankplan\20261006T040018739552Z`.
Checkpoint ruteado SHA-256:
`e9e93ef08348f136be89be9922727a3cbef270f3c7ac5d210cd38f6984f4fc18`.
Netlist SHA-256:
`119f51a4235f0bb23e30ce61d27132c1863ef65b4a1e0a5328497f527e14f76a`.
El respaldo es incremental por SHA-256 en
`D:\AI\artifacts\ntrpu-content-store`, sin conservar ZIP de transferencia.
Manifiestos finales: `overlap-bankplan-r2-place-20261006`,
`overlap-bankplan-r2-route-20261006`, `overlap-bankplan-r2-netlist-20261006`
y `overlap-bankplan-handoff-20261006`. Los intentos previos se conservan
con sus propios diagnósticos; no se confunden con las tandas aceptadas.
Los pines sin conexión se registran como tales. Se conservan diagnósticos
del ejecutor: extracción inicial detenida ante uno de esos pines y
comprobación previa detenida por una diferencia lista/tupla al leer JSON.
Una primera colocación terminó, pero la exportación de utilización falló:
Vivado admite un solo Pblock por llamada. La revisión corregida genera
un informe por banco y guarda antes el checkpoint; la colocación se repitió
con la misma selección y geometría. Los intentos conservan índices y
fuentes separados.

El alcance es este bloque fuera de contexto, sin acreditación de retardos
externos, placa PCIe/HBM, CDC ni recuperación entre épocas.

## Integración posterior con IP AXI

El [ensayo con dos fronteras AXI registradas de AMD](PEER_RX_OVERLAP_AXI.md)
pasa los 16 escenarios y 15 perfiles de tráfico; conserva N+4 y añade
dos ciclos por lote de 128 transferencias sin pausas. La síntesis integrada
cambia la selección de primitivas, por lo que este plano no se trasplanta.
La colocación libre termina, pero su ruteo agota 900 s con congestión.
Ese conjunto no hereda el cierre documentado aquí. Queda preparado un
mapa actualizado de 928 celdas como propuesta para el siguiente ensayo.
