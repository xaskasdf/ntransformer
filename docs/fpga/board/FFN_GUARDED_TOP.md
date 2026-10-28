# Top FFN con filtro de capacidad DDR

## Diseño y contrato

`fpga/rtl/ntrpu_ffn_guarded_fabric.sv` conecta el motor supervisado existente
con `ntrpu_axi_capacity_guard`, después del arbitraje compartido. Conserva
direcciones host de 64 bits; el puerto externo tiene 37 bits para un canal
de 128 GiB. Ambos módulos originales permanecen sin cambios.

El top contiene únicamente conexiones: 112 puertos del motor, 35 señales
AXI intermedias y 74 conexiones del filtro. `capacity_guard_idle` describe
el filtro; no acredita que MIG esté inactivo. Se conserva el contrato de
reset coordinado. Esta conexión no resuelve el reinicio DDR con tráfico ni
garantiza el vaciado de transacciones si se pierde un reloj.

## Simulación del top real

Los bancos `20261004T100229Z-V7-ffn-supervised-guarded-top-fast` y
`20261004T100239Z-V7-ffn-supervised-guarded-top-slow` terminaron con salida 0
en 397,866 y 415,855 segundos. Fuente `source-20261004T100200Z.zip`, SHA-256
`6a668f1d9d0580820fdd0ff0a05ff431e7af1cdb829e120827265ff8e8254f1f`.

| Medida por perfil | Host 4 ns | Host 6,2 ns |
| --- | ---: | ---: |
| Pruebas de acceso / descriptores DECERR | 36 / 12 | 36 / 12 |
| Lecturas de control sin corrupción | 21 | 21 |
| FFN D32/H32 / valores FP32 exactos | 3 / 96 | 3 / 96 |
| Ciclos de cómputo por FFN | 2122 / 2121 / 2132 | 2102 / 2149 / 2292 |
| Eventos idénticos al banco anterior | 14698 | 15799 |

Se usan el MIG generado, su modelo genérico, HLS y CDC AMD. El monitor
observa pasivamente las entradas del filtro dentro del top sintetizable.
Se conserva exactamente el estímulo y los verificadores del
[banco de errores](DDR_AXI_GUARD_ERRORS.md). Por perfil se rechazan 62
alteraciones de trazas y siete de cableado. La advertencia conocida de
declaración adelantada de `probing` permanece documentada, sin supresión.

Ambas tandas están respaldadas en el volumen NFS y en
`artifacts/ntrpu-board-vivado/<run_id>/`, con todos los miembros y la copia
de fuentes verificados. SHA-256 de los archivos de resultados:

- Rápido: `6537d81fc1eab427a39a9eaeb02a23abd0d1b441bc881949f3b5aeab4c6270a8`.
- Lento: `dee2a31d9099a6a26ffab18109dd5e96bd024ff88e88e3382fbd5fc0fe15640a`.

## Implementación y límites

`tools/ntrpu_board_ffn_guarded_synth.tcl` prepara la síntesis OOC con los
relojes existentes de 5, 4 y 10 ns. Reubica bajo `compute/` las 51
excepciones CLR explícitas; conserva las restricciones de las IP.
`tools/ntrpu_board_ffn_guarded_audit.tcl` revisa conectividad, anchos,
inicialización, CDC y timing del checkpoint resultante.

La síntesis `20261004T101702Z-V7-ffn-supervised-guarded-top-synth` terminó
en 608,930 s. La auditoría `20261004T103636Z-…-audit-r3` pasó sobre el mismo
checkpoint, SHA-256
`e466cb42604c6169d698e46ded2c0c33ac2e9dff41ce82719a831e013c58c062`.

| Métrica del conjunto | Resultado de síntesis |
| --- | ---: |
| LUT / FF / DSP | 56.488 / 51.484 / 153 |
| BRAM36 / BRAM18 | 48 / 3 |
| Incremento frente a FFN supervisada anterior | 1.126 LUT / 171 FF |
| WNS / WHS | +0,173 / +0,013 ns |
| Peor margen entre 20 rutas de dirección revisadas | +2,935 ns |
| Extremos internos sin restricciones / sin reloj | 0 / 0 |
| Bits de entrada / salida sin retardos externos | 1.408 / 1.291 |

Se comprobaron 128 bits registrados de dirección, incluidos los 54 bits
altos que llegan a estados de rechazo; los puertos DDR conservan 37 bits.
Siguen presentes las 51 etapas CLR, 131 registros persistentes de IRQ,
tres relojes, 17 grupos de skew y 21 waivers de proveedor. No se añadieron
excepciones globales. El revisor rechaza 14 alteraciones de evidencia.

Vivado reconstruye la jerarquía y mueve lógica entre módulos. Los intentos
de auditoría `102723Z` y `103344Z` fallaron al exigir nombres de pines/redes
RTL que ya no existían; están respaldados como fallos. El inventario
`103010Z` explica la transformación. La revisión final sigue los registros
del árbitro y su conectividad con
[`all_fanout`](https://docs.amd.com/r/en-US/ug835-vivado-tcl-commands/all_fanout),
sin modificar ni volver a sintetizar el circuito.

Síntesis, auditoría final y los tres diagnósticos están respaldados con
verificación de todos los miembros. Faltan el DIMM exacto,
reset DDR caliente, PCIe/DMA completo, propiedad de buffers, cuatro canales,
retardos de placa y hardware. Son simulaciones de FFN sintética con arranque
frío; no son medidas físicas ni inferencia Llama completa.

## Ruteo y auditoría física del conjunto

Actualizado a las 11:15 UTC del 2026-10-04. El ruteo
`20261004T104259Z-V7-ffn-supervised-guarded-top-route` terminó en 1.350,013 s,
con cuatro hilos. **Falla setup**; la salida 0 indica que Vivado terminó.
La auditoría conjunta `20261004T110541Z-…-route-audit` revisó el mismo
checkpoint en 209,462 s, sin cambiar sus restricciones.

| Medida después del ruteo | Resultado |
| --- | ---: |
| LUT / FF / DSP | 56.313 / 51.153 / 153 |
| BRAM36 / BRAM18 | 48 / 3 |
| WNS / TNS / extremos con fallo de setup | −0,176 ns / −5,375 ns / 144 |
| WHS / extremos con fallo de hold | +0,010 ns / 0 |
| Margen de ancho de pulso / fallos | +1,468 ns / 0 |
| Redes que requieren ruteo / completamente ruteadas | 100.722 / 100.722 |
| Errores de ruteo / huecos de relojes padre | 0 / 3 |
| Menor margen de 20 rutas de dirección | +1,537 ns |
| Menor margen de 17 grupos de skew | +2,218 ns |

La peor ruta es comparador FP64 del MAC → habilitación de BRAM de
activaciones: 4,509 ns de datos, incluidos 3,224 ns de interconexión y 14
niveles lógicos. El filtro de direcciones no es esa ruta crítica.

Se conservan 128 bits registrados de dirección, 51 etapas CLR, tres
registros de reset externo y 131 INIT de IRQ. La polaridad física de los
204 pines inspeccionados coincide con el contrato lógico. Se revisaron
los cambios de diagnóstico CDC tras empaquetar inversores y los 21 waivers
originales. Las 1.315 redes del aviso de cargas ausentes corresponden a 1.270 salidas
externas y 45 redes internas sin uso del convertidor AMD; no se aceptaron
redes de lógica propia sin clasificar. El revisor rechazó 20 alteraciones
de evidencia. Un texto anómalo en el eco de `report_drc` quedó registrado;
la tabla y las propiedades de la infracción confirman `RTSTAT-10`, sin waiver.

Ambas tandas tienen copia local y NFS con todos sus miembros verificados:

- Ruteo, SHA-256: `576cdee9cbb4f6afce019e7d6752c311118336e9c934cf2d53df8591ff5a537e`.
- Auditoría, SHA-256: `8d9eaf6ea1cc39ea42463dcc2635ed5b10186ba657cb2468a15c40dc93c822e5`.
- DCP ruteado: `37e70ccdabea95346c0143c6c0f18f66ed01358bd1983c5fc5cb44ab47a8a09e`.

Tras verificar esas copias se lanzó
`20261004T111431Z-V7-ffn-supervised-guarded-top-physopt`: una pasada
`phys_opt_design -directive Explore`, seguida automáticamente por la misma
auditoría de solo lectura. Conserva relojes de 5/4/10 ns, excepciones y
sitios supuestos `BUFGCE_X0Y60/61/62`. El resultado se detalla a continuación.

Todo este ruteo es OOC: permanecen tres huecos de relojes padre, cuatro
contratos externos de señales de disponibilidad y 1.402 entradas/1.291
salidas sin retardos de placa. No cierra la implementación de la tarjeta.

## Optimización revisada y replay funcional

Actualizado a las 11:35 UTC del 2026-10-04. La optimización `111431Z`
terminó en 590,371 s y la auditoría `112432Z` en 215,766 s. **Pasa el timing
interno OOC:** WNS +0,010 ns, WHS +0,010 ns, WPWS +1,468 ns; cero extremos
con fallo de setup, hold o pulso. Son márgenes estrechos bajo el contexto
exploratorio anterior, no una calificación de frecuencia de la placa.

El conjunto ocupa 56.321 LUT, 51.153 FF, 153 DSP, 48 BRAM36 y tres BRAM18.
Las 100.730 redes que requieren ruteo están completas, sin errores. La
auditoría conserva los 128 bits de dirección, las 51 etapas CLR, 131 INIT,
17 grupos de skew y 21 waivers originales. El mínimo margen de dirección
sigue en +1,537 ns y el de skew en +2,218 ns. Se rechazaron 24 alteraciones
de evidencia, incluidas identidad y restricciones del DCP de entrada.

El nuevo banco `tb_ntrpu_ffn_public_ddr.sv` conserva el estímulo, los modelos
MIG y los verificadores de bytes del banco anterior. Omite los monitores
internos de fase, notificador y entrada del filtro, cuyos nombres pueden
desaparecer en la implementación. Observa los puertos de host y memoria.
El control RTL `20261004T112246Z-V7-ffn-supervised-public-rtl-fast` pasó en
385,541 s: 14.498 eventos públicos idénticos, incluidos tiempos, a la
referencia `100229Z`. Los 201 eventos internos excluidos no se reconstruyen.
El verificador rechazó 16 alteraciones de datos, direcciones, DECERR,
strobes, tiempos, IRQ y lecturas; conservó 36 pruebas y tres FFN/96 FP32.

Las tres tandas tienen respaldo local/NFS con todos sus miembros verificados:

- Optimización: `a8e74d278e9f024e6e7e07dbd48fa4fb5c3ba88abb32cd0af24a8ae634d98428`.
- Auditoría: `781ed21443ebdf10d7ce9ebed598305c65c6698d153d1cada7b2ae5f46768f7b`.
- Control RTL público: `72ee2009e1998ebf9d3221052bbb64a607f4066eefecba18e5c09cf4449de37e`.

## Netlist funcional y segundo reloj del host

Actualizado a las 12:01 UTC del 2026-10-04. El netlist
`20261004T113319Z-V7-ffn-supervised-public-netlist-fast` **pasa la revisión
funcional independiente** en 670,243 s. Usa el DCP optimizado
`3cd72700cb203417a1a4245bee0b63fead73cf72b366e018bca6b45291da861a`,
exportado con `write_verilog -mode funcsim`; no compila el RTL original
del motor como sustituto. El modelo MIG/DDR y el estímulo se conservan.

Hay tres FFN D32/H32, 96 FP32 exactos, 36 pruebas de acceso, 12 DECERR,
21 lecturas de control y actividad en cuatro ranks. El verificador
reconstruye bytes y exige direcciones, respuestas, ID/LAST, strobes,
payload retenido, flancos de reloj y orden MMIO/IRQ. Rechaza 78 alteraciones
de evidencia. Los registros CSR informan 2122/2121/2132 ciclos de cómputo;
este banco no observa los pines internos para medirlos independientemente.

La traza completa **no es temporalmente idéntica** a RTL: algunas salidas
son indeterminadas a 1 ps y se estabilizan a 101 ps bajo reset. El modelo
FDRE instalado incluye un retardo de salida de 100 ps. También cambian
transiciones de reset/disponibilidad y una muestra previa a la admisión.
Los demás grupos de eventos coinciden con la referencia. Se exige estado
conocido después de 1 ns y se conserva la diferencia, sin normalizar tiempos.
Las dos definiciones de `glbl` son idénticas y la elaboración tiene una
sola raíz global. Continúan documentadas dos aserciones anidadas del
calibrador AMD que XSIM no soporta.

El control RTL lento `20261004T113749Z-V7-ffn-supervised-public-rtl-slow`
pasa en 423,739 s: 15.599 eventos públicos idénticos a `100239Z`, incluidos
tiempos. Conserva las tres FFN y las 36 pruebas. El verificador independiente
también pasó los controles RTL rápido/lento y netlist rápido, rechazando
62 alteraciones en cada traza; las otras 16 comprueban proyección pública,
IRQ y ausencia de observaciones internas inventadas.

Ambas tandas tienen fuentes, resultados y todos los miembros verificados
en `artifacts/ntrpu-board-vivado/<run_id>/` y NFS:

- Netlist rápido, 2.459 archivos: `4d8156f8b03406fdf701b56fb9e00c76267649611cc2b1d45caca14cbb28a8d6`.
- RTL lento, 833 archivos: `9defbc1e5b9ca7bfcf3f08a36532a92ce3b806d11da3b6072f6389f0a6082190`.

El netlist lento `20261004T120039Z-V7-ffn-supervised-public-netlist-slow`
también **pasa**, en 677,005 s, con host a 6,2 ns y el mismo DCP. Sus
15.608 eventos públicos permiten verificar tres FFN/96 FP32, 36 pruebas,
12 DECERR y 21 lecturas de control; se rechazan 78 alteraciones. Los CSR
informan 2096/2139/2284 ciclos y se observan 93/78/78/115 activaciones por
rank. Cambian tiempos de DMA, MMIO, IRQ y memoria respecto al RTL lento;
la validación numérica y de protocolo no exige equivalencia temporal.
El respaldo local/NFS contiene 2.459 archivos verificados, SHA-256
`9f3e3c55bfb2fcf081d947ecfe34398f678c015d67beaa1dcf51446211519c41`.
Fuente `source-20261004T115648Z.zip`:
`2286cee45e38fd196213c1202800b70345ecdd18d9a99a1cf98f25e507ed44af`.
Son pruebas funcionales sin SDF, con arranque frío y DIMM genérico;
no califican timing de placa, reset caliente ni inferencia completa.

## Próxima integración PCIe y DDR

Estado del banco integrado y sus diagnósticos: [PCIe + DDR + FFN](PCIE_DDR_FFN.md).
La tanda `123643Z` terminó y está respaldada: transferencias y 32 FP32
exactos, 2.429 ciclos UI. `130150Z` está respaldada y correlaciona todos los
avisos con lecturas periódicas internas en el tramo completado. Una última
respuesta quedaba después de `$finish`; `133315Z` la observa con una espera
final pasiva y pasa la revisión estricta, incluidas 94 alteraciones rechazadas.
`140242Z` conserva un fallo del banco al clasificar `idle_stopped` como otro
error; está respaldada. `144107Z` pasa los siete casos con la comprobación
corregida, sin cambios en el RTL del dispositivo ni en las IP: dos DECERR,
lecturas de control exactas y recuperación de ambos canales. La revisión
independiente rechaza 163 alteraciones y cierra las 233 lecturas periódicas.
El banco `152800Z` también pasa once descriptores en el límite de 128 GiB:
rechazo de la porción inválida, ausencia de alias y recuperación; 193
alteraciones rechazadas. Un C2H fallido publica 64 bytes válidos antes del
error: no consumir resultados sin finalización correcta. Propiedad de
buffers y reset DDR caliente siguen pendientes.

El banco PCIe actual (`ntrpu_pcie_ffn_app.sv`) usa RAM conductual,
`ntrpu_ffn_async_fabric` y un reloj de motor generado por el banco.
La siguiente tanda debe reutilizar su root PIPE/XDMA y los verificadores
CQ/RC, sustituyendo la aplicación por este top y el MIG ya probado:

| Conexión | Contrato pendiente de probar en conjunto |
| --- | --- |
| XDMA `user_clk` → `host_clk` | AXI4 512/64 bits y AXI-Lite 32 bits mediante los CDC existentes |
| MIG `ui_clk` → `clk` | Motor y puerto DDR en el reloj real del controlador; dirección de 37 bits solo después del filtro |
| IRQ y `usr_irq_ack` | Mantener identidad ABI2 y separación entre ACK de transporte y servicio MMIO |
| Reset y disponibilidad | Obtener confirmaciones de los pares reales; no sustituirlas por constantes ni confundir enlace activo con transacciones drenadas |

Primero exigir elaboración, enlace, calibración y admisión sin tráfico
prematuro; después H2C, una FFN D32/H32, C2H y MSI-X con referencia de
bytes independiente. Conservar los dos trabajos inválidos del banco PCIe
y verificar que las respuestas del filtro llegan a los estados XDMA.
Ensayar resets durante tráfico requiere cerrar el contrato de recuperación
DDR; no queda validado por el éxito del arranque frío.
