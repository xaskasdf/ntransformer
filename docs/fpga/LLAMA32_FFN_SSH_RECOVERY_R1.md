# Recuperación de supervisión de la FFN real

## Incidente y ejecución conservada

El 12 de octubre de 2026, a las 03:05 UTC, la conexión SSH del
coordinador R4 terminó con código 255 (`Connection reset`). El banco
`20261012T010512Z-V7-llama32-full-ffn-r3` siguió ejecutándose: se
comprobaron sus procesos originales, directorios y tiempos de nacimiento
en `/proc`. La desconexión no constituye un resultado de Vivado.

Se conserva el runner remoto 3926126, iniciado a las 01:05:15 UTC, y
su límite original de 14.400 segundos. No se reinició XSIM, no se
modificaron fuentes, pesos, estímulos ni criterios numéricos. El ZIP de
fuentes mantiene SHA-256
`fb70fe6402884a62d67417d8c133e3dc9b0ae12d18f1511521cea2f1ced027b0`.

## Observador y dependencias

El primer observador R5 falló porque intentó leer `output/seed-1.log`
antes de su exportación al finalizar. Su diagnóstico se conserva.
R6 consulta `console.log` mientras la simulación está activa, identifica
el runner por PID y tiempo de nacimiento y tolera desconexiones de
transporte sin declarar terminada ni relanzar la tanda.

La prueba previa ejecutó el sondeo real y tres controles de transporte:
timeout y código 255 mantienen la espera; un fallo del programa remoto
conserva el diagnóstico. Al terminar, R6 exige metadatos de cierre y
ausencia de procesos propios, reutiliza la revisión numérica R4 y
verifica el archivo respaldado antes de liberar la dependencia.

Las 14 dependencias detenidas no habían iniciado nuevas tandas. La
recuperación R2 conserva ese hecho y el orden: ocho etapas físicas y
de máscaras; RMSNorm HLS y revisión de transacciones; RoPE; residual;
norma→Q8. La auditoría de métricas FFN depende directamente del cierre
de la FFN y puede ejecutarse localmente mientras avanza la cola física.

## Evidencia y límites

En `build/cloud-setup/20261002-board/`:

- `llama32-ffn-ssh-adoption-r2-preflight.json`: sondeo real y controles.
- `llama32-full-ffn-r6-package.json`: identidad e insumos conservados.
- `pcie-symbolic-monitor-predecode-ssh-recovery-r2-preparation.json`:
  14 verificaciones de arranque y 24 auxiliares derivados. Los estados
  de padres todavía no creados se suministraron sólo en memoria.
- `llama32-ssh-recovery-r2-prelaunch-backup.log`: cápsula previa en D:
  y NFS, con SHA-256 y verificación de todos los miembros.

La preparación R1 de dependencias nunca se lanzó. Ninguna observación
en vivo ni prueba del coordinador acredita un PASS RTL, inferencia
completa, frecuencia implementada o rendimiento de DDR físico.
