# Concurrencia del retorno HBM: prueba de los conversores

Resultado del 2026-10-07, previo al barrido A de [retorno HBM](HBM_READ_RETURN.md). **Los conversores AMD usados actualmente permiten al menos ocho fragmentos de lectura pendientes en esta configuración.** La limitación a uno está en nuestro adaptador. Su sustitución todavía necesita RTL, validación de fallos, timing y simulación HBM nativa; esta prueba no cierra esas etapas.

**Resultado posterior del mismo día:** [W1: seguimiento concurrente](HBM_READ_WINDOW.md) completó esas pruebas en una composición privada y conserva ocho créditos como candidato. Con HBM nativa, dos lecturas de 4 KiB tardaron 920 ns frente a 940 ns con cuatro. El resultado físico de ocho tiene solo +5 ps de margen interno; el adaptador de referencia permanece intacto.

## Alcance de la prueba

Se copiaron los wrappers y configuraciones de la revisión 3 calificada, verificando sus hashes. Las dos instancias se extrajeron sin cambios de `ntrpu_hbm_read_adapter.sv`: downsizer 512→256, `MAX_SPLIT_BEATS=16`, `FIFO_MODE=0`; conversor AXI4→AXI3, `TRANSLATION_MODE=2`. No se modificó ese adaptador.

El banco conecta esas IP a un respondedor sintético ordenado, ID cero, con límite de 1/2/4/8 transacciones aceptadas. Ambos extremos usan un reloj de 2,222 ns. Quedan fuera CDC, origen de 200 MHz, frame, comprobador de respuestas, HBM y PCIe. Las cifras siguientes son ventanas finitas desde la primera AR de origen hasta el último dato, sin calentamiento ni CPL; no son caudal sostenido del sistema.

La documentación permite múltiples transacciones en el downsizer y advierte restricciones adicionales cuando el conversor AXI3 debe dividir una transacción. Aquí el downsizer ya entrega fragmentos de hasta 16 beats; se comprobó el comportamiento efectivo de la cadena. [PG059: downsizer](https://docs.amd.com/r/en-US/pg059-axi-interconnect/AXI-Downsizer), [conversión AXI4/AXI3](https://docs.amd.com/r/en-US/pg059-axi-interconnect/AXI4-to-AXI3-Converter).

## Resultados

Pasaron **28 escenarios y 15 controles negativos del ledger**. Incluyen saturación con la primera respuesta retenida, cuatro comandos consecutivos de 4 KiB, consumidor detenido/intermitente, SLVERR en el último beat y longitudes de 1/8/9/64 beats de 64 bytes. Se comprobaron direcciones, cobertura, datos, IDs, RLAST, RRESP, orden y estabilidad bajo contrapresión. El ledger reconstruye además la trayectoria de operaciones pendientes, agrupando altas y bajas del mismo flanco.

Ventana en ciclos para cuatro comandos de 4 KiB, un canal, consumidor disponible:

| Latencia sintética AR→primer R | 1 crédito | 2 créditos | 4 créditos | 8 créditos |
| --- | ---: | ---: | ---: | ---: |
| 48 ciclos | 2.050 | 1.042 | 562 | 562 |
| 64 ciclos | 2.562 | 1.298 | 690 | 578 |
| 96 ciclos | 3.586 | 1.810 | 946 | 610 |

La prueba de saturación observó exactamente 1/2/4/8 AR aceptadas antes de la primera respuesta. A 48 ciclos, cuatro créditos cubren la espera de este modelo; ocho retienen más operaciones sin reducir la ventana. A 96 ciclos, ocho reducen la ventana de 946 a 610 ciclos frente a cuatro. **No hay un número óptimo independiente de la latencia y del consumidor.** No interpretar el metadato `NUM_READ_OUTSTANDING=2` del wrapper como una medición del límite efectivo de fragmentos.

Para nueve beats de origen, la IP produjo fragmentos nativos de **2+16**, no 16+2. El primer auditor suponía incorrectamente fragmento máximo primero; se corrigió para comprobar cobertura contigua y límites legales, conservando todas las comprobaciones de datos y orden. La tanda inicial de 20 casos quedó guardada con el auditor corregido; la tanda final incorpora ese cambio desde su preparación.

El SLVERR se conserva en el beat ancho correspondiente. Este banco permite que los conversores entreguen el prefijo anterior al error: **no comprueba la protección de frame completo**. Esa propiedad corresponde al lector ya calificado y debe mantenerse al introducir concurrencia.

## Decisión de diseño para el próximo candidato

Conservar el frame de 4 KiB por canal y parametrizar únicamente el seguimiento de fragmentos a 1/2/4/8. Comparar especialmente cuatro y ocho con HBM nativa antes de elegir. En el perfil actual, una lectura máxima se divide en ocho fragmentos: ampliar el seguimiento por encima de ocho no ofrece más trabajo de ese único frame.

El candidato debe distinguir estos estados, sin reutilizar un único indicador `busy`:

1. **Reservado:** el conversor entregó una dirección y existe una entrada para ella antes de ofrecerla a HBM.
2. **Ofrecido:** ARVALID y sus campos permanecen estables hasta aceptación, incluso si aparece un fallo.
3. **Aceptado:** HBM adquirió la obligación; las respuestas se asocian en orden a sus longitudes reales.
4. **Validado:** el último beat superó RID/RLAST/protocolo. Solo entonces vuelve el crédito de seguimiento; los datos en colas posteriores siguen contando para `pending` y drenaje.

Mantener los seis bits de RID en la comprobación, las dos etapas registradas y la cuarentena. Nuevos fallos cierran admisiones; no borran direcciones ofrecidas ni respuestas ya ofrecidas al conversor. RREADY debe depender de estado registrado. Una cola llena, stop, pérdida de inicialización, errores tardíos y pausa de memoria deben probarse con varias obligaciones simultáneas.

Estos créditos son **entradas de seguimiento, no nuevos buffers de payload**. El segundo frame por canal es otro experimento: eleva la reserva de lectura de 8 a 16 KiB totales y permite llenar mientras se entrega. No comparte automáticamente las reservas del escritor.

El timing aislado vigente conserva solo +4 ps en memoria. La concurrencia aumenta control y conexiones; ninguna ganancia de esta prueba autoriza conservar aquel resultado físico sin volver a implementar el candidato.

## Evidencia y reproducción

Tanda final: `D:/AI/artifacts/ntrpu-read-concurrency/20261007T151041144201Z/`. Compilación, elaboración y simulación sumaron 19,729 s usando dos CPUs con prioridad reducida. `REVIEW.json` contiene métricas y trayectorias; `output/FILES.json` identifica fuentes, wrappers, configuración y logs por SHA-256. La referencia y los bloques anteriores permanecieron intactos. Respaldo incremental en `D:/AI/artifacts/ntrpu-content-store/` con restauración y repetición del auditor.

```bash
# DONANTE es el output de una preprueba revisada del retorno, revisión 3.
# SALIDA debe ser nueva; escoger dos CPUs disponibles.
python3 -B tools/run_ntrpu_read_converter_probe.py \
  --donor DONANTE --output SALIDA \
  --settings /ruta/2026.1/Vivado/settings64.sh --cpus 26,27
python3 -B tools/check_ntrpu_read_converter_probe.py SALIDA
```

El ejecutor verifica el donante antes y después. La prueba usa las bibliotecas AMD instaladas; no distribuye ni sustituye la instalación de Vivado. La selección eléctrica DDR/HBM y la integración peer siguen siendo decisiones y validaciones separadas.
