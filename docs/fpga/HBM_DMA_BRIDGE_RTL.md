# Puente HBM–DMA: RTL experimental y banco local

Implementación aislada del [contrato](HBM_PCIE_BRIDGE_CONTRACT.md), usando el
[modelo Python](HBM_DMA_BRIDGE_MODEL.md) como referencia de transacciones.
La tanda local es `D:/AI/artifacts/ntrpu-hbm-dma-rtl/20261009T005545469986Z`.
No cambia la cola principal ni conecta todavía HBM o XDMA nativos.

## Organización del RTL

| Archivo en `fpga/experimental/` | Responsabilidad |
| --- | --- |
| `ntrpu_hbm_dma_bridge.sv` | Lease configurada, admisión serializada, watermark común, cuatro contextos y reset frío |
| `ntrpu_hbm_dma_bridge_lane.sv` | Dos reservas por canal; asociación descriptor/datos/CPL/DMA/ACK; retención ante fallos |
| `ntrpu_bridge_async_fifo.sv` | Colas de cuatro entradas con punteros Gray y sincronizadores de dos etapas |

Cada canal tiene colas separadas de comandos, payload y finalización. Como
solo puede haber dos contextos activos por canal, las cuatro entradas de
eventos alcanzan para retener sus dos pulsos Done cuando se detiene el reloj
del consumidor. Payload y resultados mantienen VALID y su contenido durante
contrapresión. El descriptor se carga únicamente con Run, Ready y enlace activo.

La entrada usa `lease_t`/`request_t` internos. El wrapper comprueba identidad,
generación, permisos, rangos, publicación, solapamiento y spans de 64–4096
bytes. La lease procede de un plano de control autorizado. Falta conectar
el mapa HBM nativo, la publicación real de pesos y la ABI PCIe de control.

## Diferencias explícitas respecto al modelo abstracto

- El modelo elige el primer slot libre; el RTL alterna un puntero. El revisor
  compara rango, exclusión y vida del writeback, sin exigir el mismo slot.
- El ACK se acepta cuando la indicación de último beat enviado ya cruzó al
  dominio del lector. El integrador debe presentar el ACK en esa frontera;
  todavía no existe un receptor que retenga ACK mientras llega esa indicación.
- El resultado RTL es una oferta estable. Si un fallo llega mientras se
  espera READY, conserva el resultado ofrecido y **retiene la reserva** incluso
  si se acepta. El consumidor debe considerar también el estado de fallo;
  un resultado OK aislado no autoriza reciclar recursos tras esa condición.
- Un reset frío borra el estado local. Solo es legal con aislamiento externo
  y reinicio coordinado de ambos extremos; este bloque no produce tal certificado.

Los errores DMA se reciben como una indicación ya decodificada. La lectura y
habilitación de los registros XDMA, la autenticación del ACK y el writeback
real quedan fuera de este perfil.

## Banco y comparación

`fpga/tb/tb_ntrpu_hbm_dma_bridge.sv` usa relojes de 200/250 MHz, pausas de
consumidor y fases variables, incluidos flancos coincidentes. Un receptor
independiente comprueba cada descriptor y byte transmitido, los bits altos
de direcciones, los slots de writeback y la estabilidad de interfaces.

La ejecución completa contiene 25 escenarios: ambos canales, dos Done
consecutivos con reloj del lector detenido, seis ordenaciones CPL/DMA/ACK,
reutilización y vueltas de punteros/generaciones, errores parciales, datos
malformados, ACK obsoletos/duplicados/prematuros, identidad sobre 32 bits,
remapeo ocupado y pérdidas de enlace. El escenario normal se repite con
otras 23 semillas: 48 escenarios ejecutados en total.

El revisor reproduce ocho escenarios exitosos de la corrida completa y
los 23 normales adicionales sobre el modelo Python: 110 transferencias y
108.416 bytes. Comprueba las imágenes de memoria con direcciones completas.
Ocho controles deben rechazar trazas alteradas. Los escenarios de fallo
dependen de las aserciones del banco RTL y de sus marcadores completos;
no se afirma equivalencia formal ni exhaustiva del diseño.

## Reproducción y cierre pendiente

Icarus Verilog 13.0 se extrae de un paquete de MSYS2 dentro de la tanda.
Las DLL requeridas se copian al directorio privado y se registran sus hashes;
no cambia PATH ni la instalación global. El ejecutor exige salida nueva y
verifica fuentes/herramientas antes y después. Avisos de sensibilidad a arrays
se conservan en el log; cualquier otra advertencia exige revisión.

```powershell
python -B tools/run_ntrpu_hbm_dma_bridge_rtl.py D:/AI/artifacts/bridge-rtl-nuevo --toolchain D:/AI/artifacts/ntrpu-hbm-dma-rtl/20261009T005545469986Z/toolchain
```

El respaldo debe incluir fuentes, logs, ejecutores, verificador y toolchain.
La prueba de restauración recompila y repite el banco desde esos archivos.
Esto valida RTL conductual; no es síntesis, timing, análisis físico de CDC,
enumeración PCIe, simulación AMD PIPE ni calificación de placa.

El siguiente paso es revisar/elaborar este RTL con Vivado y definir constraints
CDC de los punteros y buses de memoria. Después conectar el perfil XDMA VU47P
real, medir `axi_aclk` y su reset, ejercitar ambos canales y unir el HBM nativo.
La discrepancia de metadatos 250/125 MHz del wrapper AMD sigue abierta.
