# Tamaños de origen y destino con extremos de 65 bits

## Cambio y contrato

Experimento aislado del 2026-10-08, continuación del
[guard de intervalos](PEER_READ_INTERVAL.md). En su checkpoint colocado,
19 de los 20 peores caminos de setup de transporte partían de `source_size`.

`ntrpu_peer_read_admission_size.sv` reutiliza `request_source_end` y
`request_target_end` para comprobar tamaños de origen y destino. Para entradas
binarias sin signo de 64 bits, `offset <= size && bytes <= size - offset`
equivale a `{0,offset} + {0,bytes} <= {0,size}`. La suma de 65 bits conserva
el acarreo; un extremo exactamente igual al tamaño sigue siendo válido.
El rechazo separado de longitud cero permanece intacto.

Solo cambian esas dos condiciones combinacionales. No se añaden registros,
ciclos, buffers ni excepciones de timing. Se conservan prioridad de errores,
configuración, permisos, watermark, arbitraje, autenticación y retención.
La equivalencia aritmética no constituye una prueba formal de toda la máquina.
El guard anterior, las lanes, los lectores y la cola principal permanecen
protegidos. `SIZE_PROFILE.json` conserva su procedencia sin heredar calificación.

## Validación funcional

El diferencial compara los 2.519 bits de salidas públicas contra el guard de
intervalos anterior. Conserva sus 324 casos y añade 64 casos dirigidos:
16 condiciones para cada región y canal. Incluye límite exacto, un byte fuera,
tamaño y longitud cero, tamaño máximo, extremo igual a 2^64, regiones de distinto
tamaño y prioridad HANDLE/PERMISSION/REPLAY frente a RANGE y SPAN.

Resultado revisado: 388 casos, 3.878 ciclos, 391 solicitudes, 33 admisiones y
358 rechazos. Detectó ocho alteraciones de traza. También pasaron los 15 casos
existentes del guard y sus ocho controles negativos.

La composición compartida pasó sus seis casos, con 27 retiros exitosos,
46.848 bytes exitosos y cuatro reservas fallidas retenidas. Sus 6.663 líneas
`SG`/`SF` coinciden con el donante, incluidos datos y tiempos; detectó quince
controles negativos.

Se detectaron los seis mutantes RTL. Para cada región,
se introduce una comparación estricta, truncamiento del extremo a 64 bits y
uso del tamaño de la otra región. Se observaron diferencias públicas, sin timeout ni fallos de compilación.
Los casos de detección fueron 326/336 (límites exactos), 70/349 (truncamiento)
y 26/10 (tamaño intercambiado). Los casos nuevos detectan errores que pasan
los 324 casos anteriores. XSIM puede devolver cero ante `$fatal`; el auditor
exige el fallo observado y ausencia de marca de éxito.

## Medición física

La síntesis terminó en 618,923 s y pasó la revisión de procedencia, cobertura
interna y CDC. No aparecieron latches, cajas negras ni endpoints internos sin
restricciones. Se conservan las 13 observaciones CDC-3 y ocho CDC-9 del donante.

| Recurso sintetizado | Intervalos de 65 bits | Tamaños de 65 bits | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 22.197 | 21.889 | -308 |
| Registros | 21.712 | 21.713 | +1 |
| CARRY8 | 1.441 | 1.405 | -36 |
| LUT de RAM distribuida | 3.036 | 3.036 | 0 |
| BRAM equivalentes a 36 Kb | 0 | 0 | 0 |

El RTL no declara estado adicional; la diferencia de un registro corresponde
al resultado de síntesis. Se conserva `xcvu47p-fsvh2892-2-e`,
periodos de 5,000/2,222 ns e incertidumbres y restricciones CDC originales.
Ambas tandas usan ocho CPU, prioridad baja y límite de 1.200 segundos,
previa comprobación de capacidad disponible en la VM compartida.

El ruteo agotó 1.200,093 s durante la iteración global 0. Detectó congestión
inicial de nivel 5; el último contador observado fue de 36 nodos con conflictos.
No produjo checkpoint ruteado. El último resumen intermedio mostraba setup
+0,286 ns y hold -0,190 ns; no son resultados finales ni califican timing.
Las fuentes y el donante permanecieron intactos. La tanda incompleta también
fue respaldada, restaurada y revisada.

Una inspección de 82,281 s abrió el checkpoint colocado sin modificarlo y
exportó 80 caminos. Comparación de estimaciones colocadas:

| Slack colocado (ns) | Intervalos de 65 bits | Tamaños de 65 bits |
| --- | ---: | ---: |
| Transporte, setup | +0,053 | +0,177 |
| Transporte, hold | -0,212 | -0,226 |
| Memoria, setup | +0,039 | +0,163 |
| Memoria, hold | -0,102 | -0,092 |

Ninguno de los veinte peores caminos de setup de transporte parte ahora de
`source_size`: diecisiete van del guard a las lanes y tres parten de
`lease.source_base`. El peor va de `compact[3].source_offset[20]` al enable
de `lane[1].context1.bytes[1]`. Atraviesa selección, dirección inicial,
dirección final y admisión: 15 niveles, ocho CARRY8 y 4,343 ns de datos
(1,178 ns de lógica y 3,165 ns estimados de interconectado).
El enable final conserva un fanout de 1.032.

Los endpoints comparados cambian. Mejoran los peores setup colocados, pero
empeora ligeramente el hold de transporte. Este pasa del FIFO de cruce de
reloj a la RAM distribuida del frame. No se demuestra mejora de frecuencia
o ancho de banda y no se ejecutó simulación de un netlist ruteado calificado.

El cierre requiere informes finales de setup, hold, pulse width, bus skew,
cobertura, CDC y DRC. Los caminos colocados son estimaciones; HBM nativa,
PCIe, E/S y placa siguen necesitando evidencia propia.

## Siguiente experimento propuesto

Evaluar por separado una etapa elástica entre selección de solicitud y
admisión de la lane, para separar la aritmética de direcciones del enable
que carga el contexto. El camino observado y su fanout motivan esa prueba;
todavía no hay implementación ni mejora física demostrada.

Mantener todas las comprobaciones y las cuatro reservas globales. La etapa
debe sostener datos bajo backpressure y conservar revocación, errores,
watermark y retención de reservas fallidas. Como cambiaría la latencia,
comparar transacciones y orden con un scoreboard que tolere la latencia
definida; añadir pruebas de stop, revoke y fallo con la etapa ocupada.
No reutilizar sin adaptación la comparación ciclo a ciclo de este experimento.
El hold de los frames y el ruteo completo siguen siendo criterios separados;
no ampliar buffers ni relajar clocks para considerar cerrada esta variante.

## Evidencia y reproducción

Directorio: `D:/AI/artifacts/ntrpu-peer-size/20261008T055115217515Z`.
Cada tanda conserva fuentes privadas, hashes, resultados y revisión. El almacén
incremental `D:/AI/artifacts/ntrpu-content-store` se restaura y el auditor se
repite por tanda; conservar juntos `objects/` y `manifests/`.
Las doce tandas, incluida la de ruteo incompleto, fueron restauradas y revisadas.
`HANDOFF` reúne procedencia, documentos, comparación física y recibos.

Herramientas: `prepare_ntrpu_peer_size.py`, `run_ntrpu_peer_size.py`,
`check_ntrpu_peer_size.py` y `fpga/tb/tb_ntrpu_peer_size.sv`. Los bancos existentes
y el flujo físico se reutilizan sin cambiar sus fuentes.

Ejemplo para restaurar el diferencial y repetir su auditor localmente;
el directorio de destino debe ser nuevo:

```powershell
$store = "D:/AI/artifacts/ntrpu-content-store"
$review = "D:/AI/artifacts/review-size-diff1"
python -B tools/ntrpu_artifact_store.py restore $store peer-size-20261008T055425695091Z-diff1 --destination $review
python -B "$review/auditor.py" "$review/output"
```

Esto revisa evidencia guardada. Para repetir XSIM o Vivado, usar las fuentes
privadas restauradas en Linux con Vivado 2026.1 y un directorio nuevo por tanda.
