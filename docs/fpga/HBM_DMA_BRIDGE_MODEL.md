# Banco local del contrato HBM–DMA

Primera etapa ejecutable del [contrato del puente](HBM_PCIE_BRIDGE_CONTRACT.md),
posterior a la [generación XDMA VU47P](HBM_PCIE_GENERATION.md). Es un modelo
Python de eventos y colas acotadas. No ejecuta HDL, IP AMD, TLP ni primitivas
CDC; la frecuencia efectiva del endpoint y su discrepancia 250/125 MHz siguen
pendientes de medición. No modifica fuentes ni trabajos del principal.

## Decisiones comprobables

- Dos canales C2H, dos contextos por canal y watermark compartido de identidad.
  Una ruta previamente autorizada fija handles, generación, epoch y rangos.
  La autorización/publicación HBM completa sigue perteneciendo al gestor de regiones.
- Descriptores y payload conservan direcciones de 64 bits. Cuatro slots de
  writeback separados del payload permanecen reservados hasta retirar el contexto.
- Un span alineado de 64–4096 bytes por descriptor/paquete; una cola acotada
  por canal aplica contrapresión sin intercalar paquetes.
- Se reservan dos créditos de finalización por canal antes de emitir
  descriptores. Los eventos sobreviven aunque el dominio del lector se detenga.
- Retiro solo después de CPL local, evento DMA atribuido y ACK válido.
  Fallos, remapeos ocupados y recuperación del enlace conservan las reservas;
  este perfil no implementa rearmado ni liberación por timeout.

La emisión usa el subconjunto conservador `Run && Ready` para pulsar `load`.
PG195 permite un descriptor adicional inmediatamente después de retirar
`Ready`; este modelo no utiliza esa posibilidad. Queda pendiente contrastar
las secuencias con el endpoint real.
[PG195: descriptor bypass](https://docs.amd.com/r/en-US/pg195-pcie-dma/Descriptor-Bypass-Mode?contentId=~056AtKnLlttouhSRUzXQg).

## Pruebas y límites

El banco contiene 17 pruebas. Incluye 24 planificaciones reproducibles de
cuatro transferencias sobre ambos canales: 96 retiros y 107.520 bytes
comparados. El productor avanza en pasos de 5 ns y el consumidor en pasos
de 4 ns con distintas fases y pausas. Es planificación lógica de eventos;
no demuestra comportamiento eléctrico de CDC ni caudal sostenible.

Un receptor independiente contrasta direcciones completas, canal, datos,
longitudes, límites y slots de writeback con un manifiesto de transferencias.
Los destinos de ambos canales comparten bits bajos y difieren por encima
de 4 GiB. Seis alteraciones de descriptor/payload deben ser rechazadas.
El writeback del receptor es sintético; no constituye tráfico DMA observado.
Un escenario adicional compara los bytes y la publicación remota con el
modelo de transporte ya existente.

También se prueban las seis ordenaciones de CPL/DMA/ACK, pulsos Done
consecutivos, consumidor detenido, descriptor sin Run/Ready, ACK prematuro,
duplicado u obsoleto, slot reutilizado, rango/overflow, último beat inválido,
error DMA parcial y pérdida de enlace en siete fronteras. Los contextos
fallidos conservan propiedad y reportan cero bytes válidos.

La entrada de error DMA representa el resultado de una política externa;
todavía falta implementar y probar la lectura de registros de error. El
modelo tampoco define la ABI ni autentica criptográficamente el ACK. No es
una prueba de equivalencia con los ledgers RTL existentes.

## Reproducción y evidencia

```powershell
python -B tests/test_ntrpu_hbm_dma_bridge_model.py
python -B tools/run_ntrpu_hbm_dma_bridge_model.py D:/AI/artifacts/bridge-run-nuevo
```

El ejecutor exige un directorio nuevo, registra hashes antes/después,
resultados por planificación y un manifiesto de salida. El respaldo de esta
tanda conserva una copia congelada de las fuentes; la restauración debe
repetir las 17 pruebas y comparar hashes de trazas y memoria por semilla.
Directorio de la tanda:
`D:/AI/artifacts/ntrpu-hbm-dma-bridge/20261009T004033292852Z`.

El siguiente entregable es el RTL experimental y su banco de equivalencia
frente a estos escenarios, incluyendo estabilidad VALID/READY, créditos,
identidad y reset. Después corresponde el perfil PIPE VU47P real y la unión
con el HBM nativo. Las reservas de pines, potencia y placa siguen abiertas.
