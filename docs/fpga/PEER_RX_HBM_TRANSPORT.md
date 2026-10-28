# Transporte con adaptación HBM y enlace nativo

## Resultado y alcance

Continuación aislada del [adaptador de escritura](PEER_RX_HBM_ADAPTER.md).
Se conserva por SHA-256 el transporte de devoluciones registradas y se
conectan dos adaptadores AMD de 512 a 256 bits. La composición pasa la
simulación; se miden sus recursos y se elabora, separadamente, un enlace
con la IP HBM real sobre una VU47P exploratoria.

| Evidencia | Resultado | Límite |
| --- | --- | --- |
| Transporte + dos adaptadores | `HBM_TRANSPORT_FUNCTIONAL_PASS` | Destinos AXI3 conductuales, un reloj |
| Síntesis de esa composición | 10.622 LUT, 22.290 FF, 15 BRAM equivalentes | Sin colocación ni ruteo |
| Enlace con controlador HBM | `NATIVE_HBM_BINDING_ELABORATED` | Elaboración RTL, sin inicialización ni tráfico |

Esta tanda no modifica la cola de placa, el RTL previamente acreditado ni
`AGENTS.md`. No se acredita todavía la cadena completa transporte–HBM físico,
PCIe, coherencia entre tarjetas ni recuperación en caliente.

## Contrato de integración

`ntrpu_hbm_transport.sv` conserva dos canales y ocho bancos de payload:
dos dedicados por canal y cuatro de reserva. Cada adaptador mantiene cuatro
solicitudes padre; una escritura de 4 KiB produce ocho fragmentos AXI3.
Los bancos de payload y los créditos de respuestas siguen siendo recursos
distintos: devolver un banco no confirma la escritura remota.

`stop` cierra la publicación en el transporte. Los adaptadores reciben
`admit=rst_n`, para permitir que crucen las AW de trabajos comprometidos
antes de la parada. Cada adaptador conserva su cierre local por fallo.
`drained` exige tanto transporte drenado como ambos adaptadores vacíos.
Los fallos de adaptador llegan al transporte y cierran nuevas publicaciones;
el destino sano puede terminar su trabajo comprometido.

La ventana experimental sigue siendo `[0, 16 GiB)`, comprobada antes de
reducir direcciones. No es un mapa BAR autorizado ni sustituye el objetivo
de capacidad DDR4. Reset requiere una época fría coordinada.

## Pruebas de composición

`20261006T202249749858Z-sim`, **48,065 s**, pasa 17 escenarios:

- 102 solicitudes publicadas, 97 completadas, 350 fragmentos y 159.936 bytes
  comprobados desde el descriptor hasta las escrituras de 256 bits.
- Cinco solicitudes sin completion son deliberadas: tres descriptores
  retenidos por fallos y dos escrituras cuya respuesta tiene BID ajeno o EXOKAY.
- Parada con descriptor en ingreso, fetch o payload pendiente; cercas
  retenidas hasta recibir las completions externas; destino bloqueado
  mientras avanza el otro; datos anteriores a AW y rechazo local de ventana.
- Cuatro padres y **32 fragmentos simultáneamente pendientes** en un puerto.
  Al liberar las respuestas, los ocho trabajos previamente aceptados drenan
  aunque la parada siga enclavada.
- Trece alteraciones de trazas rechazadas por reproducción independiente.
  Dos mutaciones RTL aisladas también fallan: cerrar el adaptador con `stop`
  impide drenar; desconectar su fallo permite publicaciones indebidas.

El transporte genera strobes completos. La cobertura de strobes parciales
pertenece al banco anterior del adaptador. Los datos de las escrituras
cuarentenadas no se revierten ni se consideran confirmados.

## Coste y timing

`20261006T202511127248Z-synth`, **186,482 s**, usa
`xcvu5p-flva2104-2-i` para mantener una referencia de recursos:

| Recurso | Composición |
| --- | ---: |
| LUT totales / LUTRAM incluidas | 10.622 / 384 |
| Registros | 22.290 |
| RAMB36 / RAMB18 | 14 / 2 |
| DSP / URAM / latches / cajas negras | 0 / 0 / 0 / 0 |

Frente a la síntesis anterior: +2.344 LUT, +3.595 FF y la misma capacidad
BRAM. No es una comparación de colocaciones equivalentes: esta medición
OOC aún no incorpora el wrapper físico de reloj de la referencia.

A 5 ns, con incertidumbres de setup/hold de 0,100/0,050 ns, las estimaciones
internas son **−1,259 ns / −0,037 ns**. La peor ruta va de `b_head_reg`
a `state_reg` dentro del receptor: 23 niveles y 6,141 ns de datos estimados.
No hay errores DRC, bucles combinacionales ni extremos internos sin reloj
o restricción. Las 1.073 entradas y 1.135 salidas carecen de retardos externos
intencionadamente: **no hay cierre temporal acreditado**.

Checkpoint SHA-256:
`dd1cf7e6ade76a1ca52ede64637956bcb4c3ffdf84d7a0130729046348d6c8b9`.

## Enlace nativo HBM

La instalación dispone de `hbm:1.0` y partes VU35P, VU37P y VU47P. Se usa
`xcvu47p-fsvh2892-2-e` como referencia exploratoria, sin cambiar el objetivo
de la placa. La selección por defecto activa un stack de 8 GB; la configuración
explícita activa **dos stacks, 16.384 MiB, los 16 controladores y ambos switches**.
La familia HBM y sus capacidades se describen en la
[tabla de AMD](https://www.amd.com/en/products/adaptive-socs-and-fpgas/fpga/virtex-ultrascale-plus-hbm.html).

`20261006T203404859513Z-link`, **66,878 s**, genera la IP y elabora
`ntrpu_hbm_native_link`: conecta escrituras a **AXI_00 y AXI_16**, una entrada
por stack. Comprueba 1.084 conexiones contra la plantilla exportada:
dirección de 34 bits, datos de 256, strobes de 32, ID de 6 y LEN de 4.
Las lecturas y los otros puertos de escritura quedan inactivos; los relojes,
resets y señales de gestión se exponen explícitamente. Se calcula paridad
por byte y se comprueban dos conexiones deliberadamente alteradas.

Los puertos de gestión requieren inicialización y supervisión externas.
`apb_complete`, las alarmas térmicas y el reset del controlador no se tratan
como una parada ordinaria del transporte. Véanse
[puertos y mapa HBM](https://docs.amd.com/r/en-US/pg276-axi-hbm/Port-Descriptions)
y [dominios de reloj](https://docs.amd.com/r/en-US/pg276-axi-hbm/Clocking).

El intento de configurar los parámetros de reloj AXI del ejemplo a 200 MHz
fue rechazado por Vivado, que admite 225–450 MHz en ese perfil. Se conserva
el diagnóstico y se usa **450 MHz** en el enlace nativo. Esto no demuestra
un límite universal de frecuencia del bloque físico ni acredita timing a
450 MHz. El transporte de 200 MHz no se conecta directamente a ese dominio.

La elaboración tiene cero errores y cero avisos críticos. Conserva 217
avisos de IP AMD: 12 de parámetros convertidos a locales, 5 de lógica
secuencial eliminada, 100 de redes sin driver y 100 de puertos sin carga.
No proceden del wrapper generado; deben revisarse en síntesis completa.
La elaboración no sustituye esa revisión ni una simulación de memoria.

## Siguiente frontera y reproducción

La continuación de esta etapa se registra en
[arranque y transporte entre relojes](PEER_RX_HBM_CDC.md), con pruebas
funcionales CDC independientes del ensayo de arranque HBM nativo.

Preparar CDC AXI antes de los adaptadores, con inventario y transferencia
de fallo/drenaje entre dominios; integrar arranque APB y autorización de
tráfico. Después, simular la cadena con el modelo HBM y realizar síntesis,
colocación, ruteo y comprobación de netlist sobre la parte seleccionada.
El reparto de direcciones debe favorecer los pseudo-canales próximos a
cada entrada; el direccionamiento global no elimina la contención.

El puerto compartido de payload de 512 bits a 200 MHz tiene un techo
aritmético de **12,8 GB/s**, antes de pausas. Dos puertos de 256 bits a
450 MHz permitirían 28,8 GB/s en esa frontera, pero no aumentan por sí solos
el caudal del productor. Añadir buffers no resuelve ese límite; para escalar
hacen falta más productores o mayor ancho/frecuencia con timing validado.

```bash
python3 -B tools/run_ntrpu_hbm_transport.py /tanda/output --stage sim \
  --reference /return-cut-rtl/output
python3 -B tools/check_ntrpu_hbm_transport.py /tanda/output --review /tanda/REVIEW.json
python3 -B tools/run_ntrpu_hbm_transport.py /sintesis/output --stage synth \
  --reference /return-cut-rtl/output --prior /tanda/output
python3 -B tools/run_ntrpu_hbm_native_probe.py /nativo/output --stage link
```

Los runners usan dos CPU, prioridad baja, fuentes congeladas, manifiestos
SHA-256 y un máximo de 900 s por etapa. Se preservan los diagnósticos:
el revisor de síntesis necesitó comparar semánticamente el orden de claves
de `COMPOSITION.json`; todos los HDL permanecen idénticos por bytes.

Evidencia local:
`C:\Users\xasko\artifacts\ntrpu-hbm-transport\20261006T201829889995Z`.
Respaldo incremental por contenido en `D:\AI\artifacts\ntrpu-content-store`,
con restauración verificada y sin ZIP de transferencia retenidos.
