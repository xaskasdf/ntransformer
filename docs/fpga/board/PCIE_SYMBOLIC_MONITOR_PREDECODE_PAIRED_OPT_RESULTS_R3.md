# Optimización comparable del monitor — revisión R3

Tanda 974, `20261011T153534Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-paired-opt-r1`,
terminó con código 0. Se aplicó `opt_design` ordinario a los DCP revisados
966 (prefijo) y 970 (bandera del último evento). Mismos VU5P, reloj de 4 ns,
E/S de 0 ns y dos hilos; sin colocación, ruteo ni inserción de reloj.

## Mediciones revisadas

| Medida | Prefijo 966 optimizado | Bandera 970 optimizada |
| --- | ---: | ---: |
| LUT ajustadas por Vivado | 104.843 | 104.534 |
| Primitivas LUT lógicas | 112.078 | 111.768 |
| FF | 27.078 | 27.078 |
| CARRY8 | 2.114 | 2.096 |
| WNS global estimado | −2,546 ns | −2,546 ns |
| B→R setup estimado | −2,528 ns | −2,508 ns |
| C→C setup estimado | −2,407 ns | −2,407 ns |
| C→E setup estimado | −1,182 ns | −1,182 ns |
| E→D setup estimado | +0,742 ns | +0,742 ns |

Ambos tienen 1.413 endpoints de setup y 2.233 de hold incumplidos;
WHS −0,075 ns. Cada optimización elimina 456 FF, sin añadir registros ni
cambiar INIT de los supervivientes. Esto exige un replay propio antes de
atribuir equivalencia funcional al resultado optimizado.

La peor ruta global parte de `rc_user[76]` y termina en
`c_stage/registered_ops_reg[commit_slot][422]/D`: 27 niveles, siete CARRY8,
6,556 ns de datos, de los cuales 4,912 ns son ruteo estimado. La ruta
realimentada C→C parte de `allocated_tags[51]` y conserva 29 niveles.
El fallo físico E→D de la tanda Quick anterior continúa abierto; la
estimación positiva de esta tabla no lo sustituye.

## Correcciones del revisor y evidencia

R1 exigía erróneamente que el uso ajustado de LUT coincidiera con la suma
de primitivas. R2 separó ambos conteos, pero seleccionó el índice del
informe en vez de la sección `8. Primitives`. R3 corrige esa selección.
Se conservaron las revisiones fallidas; **no se repitió Vivado**.

R3 contrasta primitivas con EDIF e inventarios y uso ajustado con el
informe jerárquico. Rechaza 16 corrupciones; los dos grafos son acíclicos
y sus controles negativos pasan. No reconstruye el empaquetado físico.
La distinción concuerda con [AMD UG949](https://docs.amd.com/r/en-US/ug949-vivado-design-methodology/Disable-LUT-Combining)
y [UG835](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/report_utilization).

Archivo de 114.274.971 bytes, verificado en D: y NFS:
`390177d0915e14d56028b8cd1e1969e70eb54ac12de90e1c9b990e2742457280`.
Revisión `-independent-paired-opt-review-r3.json`, SHA-256
`087a538b50fd3f8f3b430c54d358227acd4c507cca4ce1bdf59598ec4018592b`.

## Decisión

Seleccionar el candidato 970 optimizado para replay binario completo:
menor área estimada y mismo peor setup global. Conservar 966 como
referencia. Priorizar el análisis de C antes de integrar otra variante
que solo modifique B→R. El experimento de distancia constante 973 queda
aprobado únicamente en su banco aislado. No iniciar otro ruteo Default.
