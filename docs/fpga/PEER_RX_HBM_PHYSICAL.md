# Implementación física del transporte y arranque HBM nativo

## Alcance

Continuación de la [auditoría CDC](PEER_RX_HBM_CDC_AUDIT.md), sobre
`xcvu47p-fsvh2892-2-e`. Se conserva el RTL de las 21 pruebas funcionales,
la configuración de las IP y las restricciones de los cinco estados
asíncronos. Un wrapper añade dos buffers
[`BUFGCE`](https://docs.amd.com/r/en-US/ug572-ultrascale-clocking/BUFGCE-Clock-Buffers)
con habilitación constante: transporte de 5 ns y memoria de 2,222 ns.
El colocador elige sus sitios; no se reutilizan ubicaciones de otra FPGA.

El diseño físico es un bloque fuera de contexto: todavía no contiene
la instancia HBM nativa ni PCIe. Los puertos externos conservan su
frontera de auditoría; faltan presupuestos de entrada/salida de placa.
Un resultado temporal interno favorable no acreditaría una tarjeta completa.

## Arranque nativo completado

La tanda `20261006T225151995329Z-boot` obtiene **`BOOT PASS`**.
XSIM termina a **48,6567 µs simulados**, tras 710,108 s reales.
Los dos `apb_complete` y `init_seq_complete` están altos. El banco espera
32 ciclos APB adicionales después de disponibilidad y comprueba CATTRIP.
No se fuerza ninguna señal ni se omite la configuración del fabricante.

La pausa de configuración observada entre 42 y 45 µs era un polling
de `STAT_DFI_INIT_COMPLETE`, no un atasco definitivo: el secuenciador
consulta `0x110034` y después continúa. La ROM contiene esa consulta
y el [mapa de registros AMD](https://docs.amd.com/r/en-US/pg276-axi-hbm/Memory-Controller-Register-Map)
identifica el estado de inicialización PHY en el desplazamiento `0x10034`.

Se conservan 51 muestras, incluida la repetición de la última al detectar
el fin de simulación. Este muestreo es diagnóstico, no captura exhaustiva
de transferencias APB. La duración real es coste de simulación, no latencia
de arranque de una tarjeta. El modelo es el
[respondedor interno admitido por XSIM](https://docs.amd.com/r/en-US/pg276-axi-hbm/Simulation).
**Todavía no se prueba tráfico del transporte contra esta instancia nativa.**

## Implementación y controles

La síntesis usa 12.109 LUT, 25.621 registros y 15 BRAM36 equivalentes,
además de los dos buffers globales. Los márgenes iniciales son:

| Dominio | Setup | Hold |
| --- | ---: | ---: |
| Transporte, 5 ns | +1,336 ns | −0,128 ns |
| Memoria, 2,222 ns | −0,335 ns | −0,154 ns |

Se mantienen 31 cruces CDC informativos y 21 registros del supervisor
con `ASYNC_REG`; no hay extremos internos sin reloj/restricción ni DRC
reportados por la auditoría de síntesis. El alcance DRC es fuera de contexto.

La colocación y el ruteo se revisan por separado, incluyendo setup, hold,
estado de rutas y restricciones de desfase de las FIFO asíncronas.
La tanda de ruteo `20261006T232147456054Z-route` termina en 290,919 s:
**30.716 de 30.716 redes enrutables conectadas, cero errores de ruteo**.

| Etapa | Dominio | Setup | Hold |
| --- | --- | ---: | ---: |
| Colocado | Transporte | +0,476 ns | −0,188 ns |
| Colocado | Memoria | −0,167 ns | −0,134 ns |
| Ruteado | Transporte | **+0,211 ns** | **+0,006 ns** |
| Ruteado | Memoria | **+0,025 ns** | **+0,009 ns** |

Son márgenes internos positivos pero estrechos. La peor ruta de memoria
ahora sale del convertidor de reloj hacia `adapter/guard/allowed_reg[1]`;
la ruta de reset identificada en síntesis ya no es la peor. El diseño
conserva el objetivo original; no se relajaron períodos ni incertidumbres.

La colocación `20261006T230933509594Z-place` termina en 589,497 s y se
respalda antes del ruteo. Usa 11.830 LUT y 25.571 registros, con la misma
BRAM. El buffer de transporte ocupa `BUFGCE_X0Y23`, el de memoria
`BUFGCE_X0Y0` y el control de reset, `BUFGCE_X0Y74`.

Después de optimizar quedan **27 cruces CDC informativos**: 15 CDC-3 y
12 CDC-9. El inventario pasa de 313 a 291 registros `ASYNC_REG`:
desaparecen cuatro cadenas de cuatro registros `cdc_wr_rst_busy` y seis
registros `rst_d4`, todos internos a los convertidores AMD. Las entradas
supervivientes y los 21 registros del supervisor permanecen idénticos.
Se conserva `CDC_OPTIMIZATION_DIFF.json`. Esta revisión estructural no
sustituye una simulación funcional del netlist optimizado.

El DRC previo a ruteo informa `RTSTAT-13` por rutas aún incompletas y doce
advertencias `PDCN-1569` sobre entradas de LUT de las FIFO AMD que no
participan en su ecuación. Se registran sin silenciarlas ni declarar DRC
limpio. Después del ruteo desaparece `RTSTAT-13`, pero persisten doce
`PDCN-1569` y aparece un `RTSTAT-10` por redes sin cargas enrutables:
**13 advertencias DRC pendientes**, sin errores ni hallazgos DRC críticos.

También quedan advertencias de contexto: no se definieron `HD.PARTPIN_LOCS`
para las conexiones externas ni `HD.CLK_SRC` para sus orígenes de reloj.
Los buffers internos sí tienen ubicación y rutas; no se acredita con ello
el reloj aguas arriba ni el timing de los puertos. Siguen faltando retardos
en 1.073 entradas y 1.150 salidas. Las advertencias no se ocultan con waivers.

La auditoría independiente `20261006T232732272410Z-audit`, 74,862 s,
abre el DCP ruteado sin modificarlo y confirma **12 restricciones de bus
skew satisfechas**: límite 2,222 ns, máximo observado 0,548 ns y margen
mínimo +1,674 ns. El resumen temporal verifica cero extremos fallidos en
setup, hold y ancho de pulso; el peor margen de pulso es +0,579 ns.
Se revisan 76.016 extremos de setup, 75.963 de hold y 27.917 de pulso,
dentro del conjunto de restricciones suministrado. Esto no cubre los
puertos sin contexto ni equivale a una simulación funcional del netlist.

Checkpoint ruteado SHA-256:
`e04bc6fc8b5aedfee9e87a94fdd962b6e1b3b0f6d1b2ebd3065bf48ed9b71ef5`.

Dos correcciones del driver quedan registradas. La preparación de IP
reutilizaba la variable `stage` y nombró el primer DCP `physical_ip.dcp`:
se revisó su estado sintetizado y se reutilizó el mismo hash sin resintetizar.
La primera colocación terminó, pero una comprobación exigía exactamente
dos buffers globales y rechazó el buffer de reset añadido por Vivado para
11.091 cargas. No llegó a guardar un DCP; esa tanda se conserva como
diagnóstico rechazado. La corrección distingue relojes de controles y
guarda el checkpoint antes de los reportes.

## Reproducción y respaldo

```bash
python3 -B tools/run_ntrpu_hbm_physical.py /sintesis/output --stage synth \
  --prior /auditoria-sintesis/output --review /auditoria-sintesis/REVIEW.json
python3 -B tools/check_ntrpu_hbm_physical.py /sintesis/output --review /sintesis/REVIEW.json
python3 -B tools/run_ntrpu_hbm_physical.py /colocacion/output --stage place \
  --prior /sintesis/output --review /sintesis/REVIEW.json
python3 -B tools/check_ntrpu_hbm_physical.py /colocacion/output --review /colocacion/REVIEW.json
python3 -B tools/run_ntrpu_hbm_physical.py /ruteo/output --stage route \
  --prior /colocacion/output --review /colocacion/REVIEW.json
python3 -B tools/check_ntrpu_hbm_physical.py /ruteo/output --review /ruteo/REVIEW.json
python3 -B tools/run_ntrpu_hbm_route_audit.py /auditoria/output \
  --prior /ruteo/output --review /ruteo/REVIEW.json
python3 -B tools/check_ntrpu_hbm_physical.py /auditoria/output --review /auditoria/REVIEW.json
python3 -B tools/run_ntrpu_hbm_boot_window.py /arranque/output \
  --donor /snapshot-precompilado/xsim --manifest /SNAPSHOT_INPUTS.json
```

Cada proceso usa dos CPU, prioridad baja y un límite de 900 s; la auditoría
de ruteo, 300 s. El snapshot HBM se verifica antes y después del uso.
Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-hbm-physical\20261006T224923841669Z`.
Respaldo incremental SHA-256/gzip en `D:\AI\artifacts\ntrpu-content-store`,
con objetos compartidos entre tandas y sin archivos ZIP de transferencia
retenidos. Los DCP, reportes y fuentes están locales; el ejecutable XSIM
compilado permanece como caché en la VM y se puede regenerar con el flujo
de arranque previo.

## Siguiente integración

La [regresión del netlist ruteado](PEER_RX_HBM_NETLIST.md) ya supera los
21 escenarios; las 13 advertencias DRC están inventariadas y clasificadas,
sin waivers. Definir el contexto físico real, revisar de nuevo DRC y confirmar márgenes antes de
integrar tráfico HBM nativo. Después, validar escrituras, respuestas y
lectura de comprobación en los pseudo-canales 00 y 16, y conectar el
transporte completo. Esta prueba usa HBM de 16 GB; no acredita 512 GB,
coherencia entre tarjetas ni rendimiento PCIe/BAR. El productor compartido
de payload de 512 bits a 200 MHz sigue limitado aritméticamente a 12,8 GB/s;
más bancos no aumentan ese ancho de banda por sí solos.
