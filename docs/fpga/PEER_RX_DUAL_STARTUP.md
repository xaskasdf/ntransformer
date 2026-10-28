# Arranque frío dual: corrección posterior al ruteo

Fecha local: 2026-10-04; tandas identificadas en UTC del 2026-10-05.

## Estado y alcance de la tanda original

La variante aislada corrige el bloqueo del receptor R4 integrado después
del ruteo: pasa la reproducción del fallo, STA interno, auditoría CDC y
los 56 comandos del banco con dos MIG. **El RTL compartido del repositorio
conserva la versión original**; la corrección revisable está en
`D:\AI\artifacts\ntrpu-dual-mig-startup\candidate_startup.patch`.
Los 57 archivos protegidos, incluido `AGENTS.md`, permanecen intactos.

Esto valida el receptor y su supervisor hasta las interfaces AXI de
los MIG, con simulación de primitivas sin SDF. No acredita una placa,
PHY DDR4 implementada, PCIe/P2P, HBM ni caudal sostenido.

## Fallo observado y cambio

La captura del núcleo original muestra BVALID/RVALID indeterminados
antes del primer flanco UI útil: 173,998 ns y 174,729 ns, respectivamente.
Las respuestas pasan a cero en ese mismo instante, pero los contadores
registran `X` 100 ps después. `held` también pasa a `X`; `rearm_ready`
queda indeterminado a 676,2 ns y la admisión permanece cerrada.

El contador original solo se limpiaba por reset del receptor o por
respuestas B/R pendientes. La variante añade dos condiciones locales:

```systemverilog
if (ui_rst[g] || !calibrated[g] || memory_reset_n[g] ||
    memory_bvalid[g] || memory_rvalid[g]) begin
    count <= 0;
    held <= 0;
end
```

Así, la comprobación de memoria vacía comienza después de la calibración
y de salir del reset UI. El rearmado espera a ambos controladores;
se conserva una sola apertura por configuración y no se habilita un
reinicio en caliente.

## Evidencia que distingue ambas versiones

- Captura: `20261005T013239641809Z-capture`, primeros 2 µs del banco
  original. Los 471 archivos del snapshot coinciden con el respaldo
  previo; se recompiló únicamente con trazas habilitadas.
- Original: `20261005T014230039206Z-replay-original`, permanece con
  `rearm_ready=X`, `admit=0`, incluso tras calibrar.
- Variante: `20261005T014706224442Z-replay-candidate`, mismo estímulo,
  `admit=0` antes de calibrar y `admit=1`, `running=1` después.
- Banco completo: `20261005T014715976083Z-dual-netlist`, 13 casos RTL y 56 comandos a través
  del núcleo ruteado; revisión independiente de payload, orden,
  completions, límites de dirección, reservas y progreso independiente.

Se compararon 18.816 bytes escritos y 35.840 bytes leídos, con 11
drenajes y 17 controles negativos del revisor. Las 720 advertencias
del modelo DDR4 en el puerto 0 corresponden a cinco lecturas periódicas
BL8 de direcciones aún no escritas; se verificó su causalidad y que no
se filtraran al payload. El puerto 1 no emitió esas advertencias.

El replay usa entradas registradas durante 2 µs y una continuación
sintética con calibración a 2,2 µs; termina a 3,8 µs. VCD no conserva
el orden entre ciclos delta: el replay cambia relojes antes de actualizar
respuestas mediante NBA. El banco completo confirma el resultado con
los modelos MIG originales. La captura inicial sin información de
trazas falló y también se conserva; no cuenta como validación.

## Implementación y coste

Tanda física: `20261005T013351160432Z-dual-physical`, xcvu5p-flva2104-2-i, Vivado 2026.1.
Medición OOC con entradas/salidas registradas y cuatro BUFG.

| Dominio | Periodo (ns) | Setup (ns) | Hold (ns) |
| --- | ---: | ---: | ---: |
| SOURCE | 5 | +0.251 | +0.007 |
| MANAGEMENT | 10 | +8.231 | +0.012 |
| MEMORY0 | 5 | +1.167 | +0.009 |
| MEMORY1 | 5 | +0.854 | +0.006 |

Núcleo: 7.158 LUT, 17.951 FF, 15 BRAM, 238 CARRY8 y cero DSP.
23.224/23.224 redes ruteadas, sin errores DRC, latches, blackboxes ni
endpoints internos sin restricciones. No aumentan BRAM ni FF frente
al original. La diferencia de LUT y de márgenes incluye los efectos
del nuevo placement/ruteo, no una ganancia de arquitectura.

La ruta crítica ahora va de `aw_done_reg[1]` a `first_reg[31]/D`:
4,563 ns de datos, 97,195 % de ruteo. El hold mínimo es 6 ps; hace
falta repetir STA en la integración física completa. Siguen sin
calificar los retardos IO, pines y fuentes físicas de reloj.

Auditoría `20261005T014907003875Z-dual-audit`: 75 registros sincronizadores, cadenas de
tres etapas co-localizadas, 20 restricciones de bus skew y margen
mínimo de 4.366 ns. Se mantienen las
restricciones específicas y pasan cinco controles negativos.
La advertencia CDC-26 del reset externo sigue abierta sin waiver;
se verificó su conexión directa al PRE del sincronizador.

El revisor MIG conserva una limitación histórica de su banco RTL que
afirma que no existe P&R/CDC. `SCOPE_RESOLUTION.json` aclara que, para
esta variante, los informes físicos y CDC anteriores aportan esa
evidencia mediante hashes exactos. Se preservan el reporte y el revisor
originales. La auditoría de modelos se ajustó al netlist ruteado:
espera cero avisos de FIFO comportamental, como el original ruteado;
el banco RTL anterior emitía diez. Se conserva la versión inicial
del auditor y la justificación en `MODEL_AUDIT_EXPECTATION_REVIEW.json`.

## Reproducción, respaldo y siguiente paso

Los launchers en `D:\AI\artifacts\ntrpu-dual-mig-startup` crean
snapshots nuevos; no ejecutarlos encima de resultados existentes.
`SOURCE_VARIANT.json` identifica el único cambio RTL. Los verificadores
del repositorio permanecen intactos. `audit_models.py` admite ese cambio
solo desde el snapshot físico y verifica la procedencia de los modelos.
`FINAL_REVIEW.json` reúne resultados y límites; el ZIP
`20261004-dual-mig-startup-handoff.zip` incluye las siete tandas,
scripts, patch, documentación y respaldo anterior con el ejemplo MIG.
El archivo `.VERIFIED.json` registra SHA-256, CRC y hashes por miembro.

El siguiente paso previsto al cerrar aquella tanda era integrar el patch y
medir la ocupación y las pausas por destino antes de ampliar reservas:
R4 conserva dos bancos dedicados por destino más cuatro compartidos,
de 4 KiB cada uno. La entrada compartida de 512 bits a 200 MHz tiene
un techo bruto de 12,8 GB/s, todavía no medido. Los 16 KiB compartidos
equivalen a 1,28 µs ideales de reserva a ese techo; aumentar buffers
solo ayuda mientras haya pausas absorbibles y capacidad de drenaje.

## Integración autorizada el 2026-10-05

Se integró la condición de arranque en `fpga/rtl/ntrpu_dual_mig_cold_start.sv`,
preservando los finales de línea LF del repositorio. Su contenido coincide
con la candidata probada al normalizar CRLF/LF; el único cambio funcional
es la condición documentada arriba. `INTEGRATION.json` en
`D:\AI\artifacts\ntrpu-dual-mig-performance` registra ambos hashes y
la procedencia. La medición del pool conserva el mismo núcleo ruteado.
Los seis escenarios, su alcance y el siguiente paso están en
[Rendimiento dual R4](PEER_RX_DUAL_PERFORMANCE.md).
