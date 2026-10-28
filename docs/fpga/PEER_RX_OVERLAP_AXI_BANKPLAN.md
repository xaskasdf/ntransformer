# Plano de bancos para el transporte con IP AXI

**Resultado: mapa verificado, colocación completada y ruteo detenido por
tiempo. El montaje con IP AXI continúa sin acreditar 200 MHz.**

Este ensayo continúa [la integración AXI](PEER_RX_OVERLAP_AXI.md) usando
su misma lógica y sus dos IP AMD AXI Register Slice. Cambia únicamente
la colocación física. No modifica el RTL de producción, la cola del
agente principal ni la referencia anterior sin IP, que cerró a 200 MHz.

## Comprobación antes de colocar

La sonda `20261006T063520917953Z-preflight` abre el checkpoint sintetizado
verificado, ejecuta `opt_design` en una instancia aislada y exporta un
checkpoint nuevo. Termina en **61,070 s**. El mapa candidato de la tanda
`20261006T053023018152Z-connectivity` coincide en este punto exacto:

- 928 celdas: 528 registros de estado y 400 comparadoras/derivadas.
- 2.912 conexiones de entrada en los conos examinados, sin cambios de
  conductores, miembros ni tipos de primitivas.
- 128 registros de generación compartidos permanecen fuera de los grupos.
- Ocho regiones disjuntas, con 100 o 120 sitios SLICE reales por región.

Vivado normaliza cuatro bordes inferiores: X62 pasa a X63 y X78 a X79.
La revisión compara el conjunto exacto de sitios solicitado con el
aplicado, consultando el inventario del dispositivo. También contempla
columnas ausentes dentro de una región: su capacidad no se calcula
simplemente multiplicando ancho por alto. No se amplían regiones para
ocultar diferencias.

Cuatro controles negativos rechazan una celda ausente, un conductor
alterado, un sitio omitido y un cambio del periodo de reloj.
`POST_OPT_PLAN_PREFLIGHT_PASS` habilita el ensayo; no acredita timing.

## Implementación y criterios

La colocación usa exactamente el checkpoint posterior a `opt_design`,
verificado por SHA-256, sin repetir esa optimización después de comprobar
la conectividad. Los Pblocks siguen siendo blandos: pueden orientar la
colocación sin obligar a que todas las celdas permanezcan dentro.

Se conservan reloj de 5 ns, incertidumbres de setup/hold de 0,100/0,050 ns
y BUFGCE_X0Y60. Cada etapa utiliza dos CPU, prioridad baja y un máximo de
900 s. La revisión comprueba miembros, tipos, sitios reales, restricciones,
DRC y cobertura interna; respalda cada tanda antes de habilitar la siguiente.

La acreditación exige ruteo completo sin errores, márgenes internos no
negativos y comparación funcional contra la netlist del montaje con IP.
Los puertos externos, PCIe, HBM, CDC y recuperación en caliente siguen
fuera de esta acreditación.

## Resultado de colocación

La tanda `20261006T063806457290Z-place` termina en **401,984 s** y pasa
la revisión de miembros, tipos, sitios y cobertura interna. Conserva
18.529 registros y 15 bloques BRAM; utiliza 7.684 LUT.

Solo 51 de las 928 celdas quedan dentro de su región asignada; las otras
877 se colocan fuera, permitido por `IS_SOFT`. La existencia de ocho
Pblocks no demuestra que se haya conseguido separar físicamente los bancos.
El setup preliminar es **−1,417 ns**, frente a −0,707 ns con colocación
libre; hold −0,168 ns. Esto no constituye una mejora ni STA final.

La peor ruta previa al ruteo conecta el contador de la cola W
`receiver/boundary[0].w/count_reg[1]` con `status_reg[30][0]/CE`.
Atraviesa 24 niveles de lógica, incluidos retorno, detección de errores,
despacho y admisión. Sus 6,114 ns de datos incluyen 4,289 ns de cableado
estimado. El ruteo determinará cuánto persiste este problema. Si persiste,
la siguiente hipótesis será acortar esa cadena de control en una variante
aislada, verificando de nuevo latencia, contrapresión y propiedad de bancos;
no basta con aumentar la cantidad de buffers.

## Resultado del ruteo

`20261006T064514896795Z-route` alcanza el límite en **900,018 s**, con
15 superposiciones pendientes en la tercera iteración global. Las dos
anteriores llegaron a cero superposiciones, pero sus márgenes de setup
intermedios fueron −2,771 y −1,906 ns. Esas instantáneas no sustituyen
un checkpoint y STA finales.

Se conserva `TIMEOUT_DIAGNOSTIC`: no hay `physical_route.dcp`, informe
final de timing ni comparación funcional de netlist ruteada. La revisión
y el respaldo se completaron antes de detener la secuencia. La variante
no se promueve como mejora física; repetir el mismo plano no es el
siguiente ensayo propuesto. La referencia sin IP conserva su validación
independiente de +0,054 ns de setup.

## Próxima variante propuesta

Investigar una separación registrada entre devolución/validación y
admisión/reserva, en un generador experimental independiente. No registrar
simplemente `fault` o `ready`: una devolución inválida debe seguir impidiendo
asignaciones indebidas, cada banco debe conservar un único propietario y
generación, y las transacciones AXI ya expuestas deben poder drenarse.

Si el corte añadiera un ciclo no solapable al intervalo actual N+4, bloques
de 4 KiB pasarían de 68 a 69 ciclos: aproximadamente 1,45 % menos capacidad
por ciclo en ese caso ideal. Es una estimación condicional, no un resultado
implementado; los bloques pequeños tendrían un coste proporcional mayor.
La variante deberá repetir los 16 escenarios funcionales y 15 perfiles de
rendimiento con las mismas IP, antes de síntesis y ruteo. No se ha modificado
la lógica para introducir ese corte en esta tanda.

## Reproducción y evidencia

```bash
python3 -B tools/run_ntrpu_overlap_axi_preflight.py /preflight/output \
  --checkpoint /axi_synth/output/suite/physical_synth.dcp \
  --expected-sha256 SHA256 --plan /candidate_plan
python3 -B tools/check_ntrpu_overlap_axi_preflight.py /preflight/output \
  --plan /candidate_plan --review /preflight/REVIEW.json
# Tras revisar y respaldar la sonda:
python3 -B tools/run_ntrpu_overlap_axi_bankplan.py /place/output \
  --stage place --reference /qualified_compact_route/output \
  --prior /axi_synth/output --plan /candidate_plan --preflight /preflight/output
python3 -B tools/check_ntrpu_overlap_axi_bankplan.py /place/output \
  --review /place/REVIEW.json
```

Continuar con `route` y, si pasa su revisión, `netlist`, usando la etapa
anterior como `--prior`. Usar directorios nuevos y cargar previamente
`settings64.sh` de Vivado 2026.1.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-overlap-axi-bankplan\20261006T063520917953Z`.
Respaldo incremental SHA-256/gzip en
`D:\AI\artifacts\ntrpu-content-store`, sin ZIP de transferencia retenidos.
Manifiestos de esta tanda: `overlap-axi-bank-preflight-20261006`,
`overlap-axi-bank-place-20261006`, `overlap-axi-bank-route-20261006` y
`overlap-axi-bank-handoff-20261006`.

## Variante del corte de control

La implementación aislada de devoluciones registradas, sus garantías y
resultados se documentan en [Devoluciones registradas](PEER_RX_RETURN_CUT.md).
