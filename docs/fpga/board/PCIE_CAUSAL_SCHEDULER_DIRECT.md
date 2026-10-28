# Inventario PCIe: actualización directa del fragmento

## Cambio bajo prueba

`fpga/rtl/ntrpu_pcie_inventory_scheduler_r1.sv` conserva puertos, flanco
de compromiso y capacidad del [productor inicial](PCIE_CAUSAL_SCHEDULER_RESULTS.md).
Sustituye solamente el cálculo de dirección y tamaño restantes al terminar
un fragmento RC normal, propio y no terminal. La selección y el orden de
escrituras no cambian; tampoco cambia el estado posterior a errores.

La ruta anterior pasaba por el mínimo de bytes entregados y después por
otra resta/suma. Con tamaño `R`, dirección `A` y longitud en DWORD `C`:

```text
span = R + (A & 3)
capacity = 4*C
partial = (C != 0 or (A & 3) == 0) and span > capacity
R_next = partial ? span - capacity : 0
A_next = partial ? (A & ~3) + capacity : A + R   (mod 4096)
```

El caso `C=0` y dirección desalineada conserva la comparación mixta de
signos de la referencia: el disponible negativo selecciona todo `R`.
Para un paquete retenido se reconstruye `C=(rc_expected-3) mod 2048`;
una cabecera nueva usa directamente sus once bits de longitud. Si algún
operando contiene X/Z, la simulación conserva las expresiones originales
de resta/suma. El dominio del netlist sigue siendo binario.

## Evidencia funcional

- Comprobación local de las **67.108.864** combinaciones de `R`, offset
  y `C`, más **134.217.728** comprobaciones de dirección con las partes
  altas 0 y 4092. Las demás partes altas trasladan ambas expresiones por
  el mismo múltiplo de cuatro, módulo 4096.
- Dos alteraciones de fórmula rechazadas: omitir la regla del disponible
  negativo y omitir la alineación de dirección.
- RTL `20261008T050043Z`: **22.339 filas/68 escenarios**, 5.718.784
  comparaciones de palabras, todos los estados, contextos y consultas
  idénticos a la versión anterior, incluidas filas con fallo.
- Cuatro mutaciones reales y doce controles del revisor rechazados.
  Duración RTL: **86,734 segundos**. La advertencia del `force` de tabla
  conserva el alcance documentado en la versión inicial.

La comprobación matemática está ligada al texto del helper y sus hashes;
no se presenta como prueba formal de equivalencia secuencial RTL.

## Síntesis y replay

La tanda `20261008T050519Z` mide el mismo VU5P, 4 ns e I/O de 0 ns, sin
excepciones temporales. **La variante regresa y no se promueve.**

| Medida | Inicial | Actualización directa |
| --- | ---: | ---: |
| LUT | 43.932 | 44.179 |
| FF | 9.760 | 9.760 |
| CARRY8 | 119 | 122 |
| WNS setup | −10,451 ns | **−11,162 ns** |
| TNS setup | −126.328,086 ns | −100.749,453 ns |
| WHS hold | −0,075 ns | −0,075 ns |
| Extremos setup fallidos | 15.863/28.035 | 15.866/28.036 |

Aunque mejora TNS, empeora la peor ruta en **0,711 ns** y añade 247 LUT.
La ruta ahora va de `remaining_reg[51][3]` a `remaining_bytes_reg[20]`:
61 niveles y **15,144 ns**, con 3,548 ns de lógica y 11,596 ns de ruteo
estimado. No hay latches, BRAM, URAM ni DSP. La auditoría de 64.808
primitivas/68.244 nets no encontró ciclos y rechazó ocho controles.
La síntesis duró 640,624 segundos; siguen fallando setup y hold.

El RTL binario `20261008T051716Z` y su netlist exacto `20261008T052102Z`
pasaron **22.330 filas/67 escenarios**, con nueve controles del revisor
rechazados en cada banco. Duraron 45,721 y 93,563 segundos, respectivamente.
Todas las salidas públicas, contextos y consultas coinciden; las 256
palabras internas sólo se comparan en RTL. No se compiló el productor RTL
como DUT del replay mapeado, no se aplicó SDF y se verificó la igualdad
de ambas definiciones `glbl`.

Se reutilizó la instantánea inmutable `source-20261008T045912Z`, verificando
los hashes de cada entrada, para evitar transferencias repetidas. El productor
inicial permanece como opción medida preferida; estos resultados se
conservan para no repetir la hipótesis descartada. Ninguna estimación
fuera de contexto demuestra tiempos físicos tras colocación/ruteo.

Los IDs completos añaden `-V4-ffn-supervised-pcie-causal-scheduler-direct-`
y `rtl`, `synth`, `binary-rtl` o `binary-netlist`, respectivamente.
Cada archivo está verificado bajo `artifacts/ntrpu-board-vivado/<ID>/`
con respaldo en D: y NFS. Revisores, fórmulas, hashes y cola R86 se guardan
en `build/cloud-setup/20261002-board/` y en el checkpoint siguiente.

## Integración siguiente

Conectar la validación RC a los metadatos y bytes canónicos y recuperar
la validación de payload/longitud RQ. Capturar el contexto previo del
productor junto al beat permite comprobarlo sin consultar la referencia.
La identidad esperada debe proceder de la configuración retenida al armar,
nunca de la cabecera RC ni de un BDF fijo de pruebas.

El validador RC anterior no se conecta sin adaptación: usa direcciones y
tamaños declarados en la cabecera y algunas condiciones estrictas de
X/Z difieren del comportamiento procedural de la referencia. Hay que
comparar el conjunto completo de fallos, además del estado, por secuencia.
El retiro debe cancelar etapas con reset frío, ejercer wrap y revocar
vacío inmediatamente ante tráfico o validación pendiente. La mejora de
tiempos continúa abierta; esta integración funcional no califica 4 ns,
el endpoint nativo, 512 GiB ni la implementación física.
