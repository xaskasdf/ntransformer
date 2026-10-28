# Límites de intervalo de 65 bits en admisión

## Cambio y contrato

Experimento aislado del 2026-10-08 sobre la composición con
[validación DMA/ACK paralela](PEER_READ_PARALLEL.md). Sus 20 peores caminos
colocados de setup de transporte parten de `lease.window_offset` y terminan
en enables del watermark. Esta variante intenta reducir las restas dependientes
de las comprobaciones de ventana y región publicada.

`ntrpu_peer_read_admission_interval.sv` guarda dos extremos exclusivos de
65 bits al aceptar una configuración válida: `window_offset + window_bytes`
y `published_offset + published_bytes`. Cada solicitud compara su extremo,
también de 65 bits, con esos valores. Conserva el límite inferior explícito.
Una configuración rechazada o bloqueada no actualiza los extremos.

Para entradas binarias sin signo de 64 bits y `offset >= base`, la diferencia
`d = offset - base` no desborda. Exigir `d <= size` y `bytes <= size - d`
equivale a `offset + bytes <= base + size` si ambas sumas conservan 65 bits.
Si `offset < base`, ambas versiones rechazan. Esto justifica la transformación
aritmética; no demuestra formalmente la equivalencia de toda la máquina RTL.

Permanecen intactos orden y códigos de error, rangos de origen/destino, SPAN,
watermark, arbitraje, autenticación del retiro, retención y latencia observable.
Se declaran dos extremos de 65 bits; síntesis puede eliminar longitudes de
lease que dejan de utilizarse. El cambio neto de registros debe medirse.
El guard de referencia, los lectores, las lanes y la cola principal no cambian.
`INTERVAL_PROFILE.json` conserva los hashes del donante y su perfil nativo;
ninguna calificación anterior se hereda.

## Validación funcional

El diferencial compara los 2.519 bits de salidas públicas, ciclo por ciclo,
contra el guard compacto anterior. Incluye 64 casos dirigidos en ambos canales,
cuatro escenarios de configuración y 256 combinaciones seudoaleatorias
reproducibles. Los casos dirigidos cubren límites exactos, un byte fuera,
rangos vacíos, sumas que exceden 64 bits y prioridades HANDLE/PERMISSION/REPLAY
frente a errores de rango. Configuración cubre rechazo, cambio válido,
revocación y bloqueo mientras hay reservas activas.

Resultado revisado: 324 casos, 3.252 ciclos, 327 solicitudes, 19 admisiones y
308 rechazos. Los seis controles negativos del auditor fueron detectados.
También pasaron los 15 casos existentes del guard, con ocho controles negativos.
El intento inicial `diff1` quedó archivado como rechazo del auditor: su tabla
Python tenía valores numéricos incorrectos para cuatro códigos. La repetición
`diff2` usa los valores del paquete y pasó; el RTL del banco usa sus constantes.

La composición completa pasó sus seis casos: 27 retiros exitosos, 46.848 bytes
exitosos y cuatro reservas fallidas retenidas. Sus 6.663 líneas `SG`/`SF` coinciden
con el donante, incluidos tiempos y datos, y detectó quince controles negativos.

Se detectaron los cinco mutantes RTL: truncar cada
extremo a 64 bits, omitir cada límite inferior y actualizar extremos tras una
configuración rechazada. Los fallos fueron diferencias públicas reales: casos 9, 25, 3, 19 y 65,
respectivamente. El mutante de configuración pasó los primeros 64 casos. XSIM
devolvió código cero ante `$fatal`; los auditores exigieron el fallo observado
y ausencia de marca de éxito, no solamente un código de salida.

## Medición física

La síntesis terminó en 601,441 segundos y pasó la revisión de procedencia,
CDC y cobertura interna: sin latches, black boxes, bucles ni extremos internos
sin restricciones. Conserva 13 entradas CDC-3 y ocho CDC-9 informativas revisadas.
Las E/S externas siguen sin calificación: 6.077 entradas y 5.684 salidas sin
delays, además de dos entradas APB con cobertura parcial.

| Recursos sintetizados | Guard anterior + lane paralela | Intervalos de 65 bits | Diferencia |
| --- | ---: | ---: | ---: |
| LUT | 22.624 | 22.197 | -427 (-1,89 %) |
| Registros | 21.709 | 21.712 | +3 |
| CARRY8 | 1.493 | 1.441 | -52 (-3,48 %) |
| LUT de RAM distribuida (incluidas arriba) | 3.036 | 3.036 | 0 |
| BRAM36 equivalentes | 0 | 0 | 0 |

El ruteo terminó por timeout a los **1.200,088 segundos**, retorno `-15`,
con ocho CPU y prioridad baja. Mantiene `xcvu47p-fsvh2892-2-e`, periodos
5,000/2,222 ns, incertidumbres y restricciones CDC originales. Conservó
`optimized.dcp` y `placed.dcp`, pero no generó `routed.dcp`.

La congestión estimada inicial siguió en nivel 5. Alcanzó la iteración global 4;
el último resumen intermedio mostró WNS -0,320 ns y TNS -128,475 ns, sin
resultado de hold. Llegar a cero solapamientos en una iteración no equivale
a un ruteo final aprobado. Los informes de congestión identifican conexiones
`lane_request`, `result_context`, bytes del guard y lógica de admisión de una
lane. No son una explicación exhaustiva de todos los obstáculos de ruteo.

La inspección del checkpoint colocado terminó en 82,136 segundos y exportó
80 caminos, sin modificarlo. Comparación de estimaciones colocadas:

| Slack colocado (ns) | Guard anterior | Intervalos de 65 bits |
| --- | ---: | ---: |
| Transporte, setup | -0,027 | +0,053 |
| Transporte, hold | -0,194 | -0,212 |
| Memoria, setup | +0,056 | +0,039 |
| Memoria, hold | -0,127 | -0,102 |

El antiguo camino de ventana no aparece entre los 20 peores de setup de
transporte. El primero ahora va de `guard/issue_ptr` al enable de `lane[0]/sent`,
atravesando la selección de solicitud, aritmética de direcciones y admisión de
la lane. Tiene 16 niveles (ocho CARRY8), 4,626 ns de datos: 1,102 ns de lógica
y 3,524 ns estimados de interconectado. Una red de enable del camino tiene
fanout 1.032. Los otros 19 caminos parten de `guard/lease.source_size` y llegan
a enables de reservas compactas.

Los endpoints comparados son diferentes. La mejora del peor setup colocado
no se extiende al hold de transporte ni al setup de memoria; **el timing final
ruteado continúa pendiente**. Todavía no se demuestra una mejora de frecuencia
o ancho de banda y no se ejecutó simulación de un netlist ruteado calificado.

El cierre requiere reportes finales de setup, hold, pulse width, bus skew,
cobertura, CDC y DRC. HBM nativa, localidad entre stacks, PCIe, E/S y placa
necesitan validación propia.

## Siguiente experimento propuesto

Reutilizar los extremos de solicitud de 65 bits ya calculados para los límites
de tamaño de origen y destino del guard. Para valores sin signo,
`offset <= size && bytes <= size - offset` equivale a
`{0,offset} + {0,bytes} <= {0,size}`. Esto permitiría reemplazar las restas
restantes sin agregar estado. Los 19 caminos desde `source_size` motivan esa
comparación; todavía no hay implementación ni mejora física demostrada.

Probar la variante por separado, incluyendo cero, tamaño máximo, límites
exactos, un byte fuera, desbordamiento y prioridad de errores, y repetir los
bancos y restricciones actuales. Mantener los controles completos de las lanes.
El camino entre guard y lane y su enable de alto fanout requieren una revisión
separada; no combinar una nueva etapa de pipeline con el cambio aritmético.

## Evidencia y reproducción

Directorio: `D:/AI/artifacts/ntrpu-peer-interval/20261008T045909464100Z`.
Las tandas usan copias privadas, manifiestos de hashes y el almacén incremental
`D:/AI/artifacts/ntrpu-content-store`. Las doce tandas, incluido el intento inicial rechazado por el auditor,
fueron restauradas y revisadas. `HANDOFF` reúne documentos, procedencia y
recibos. Conservar juntos `objects/` y `manifests/`.

Herramientas: `prepare_ntrpu_peer_interval.py`, `run_ntrpu_peer_interval.py`,
`check_ntrpu_peer_interval.py` y `fpga/tb/tb_ntrpu_peer_interval.sv`.
Los bancos existentes y el flujo físico se reutilizan sin modificar sus fuentes.
