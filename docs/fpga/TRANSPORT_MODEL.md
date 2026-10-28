# Modelo funcional del transporte entre dos NTRPU

Implementado y probado localmente el 2026-10-04 UTC. El modelo Python
comprueba una parte del [contrato de transporte](NTRPU_TRANSPORT.md):
regiones, ventanas retenidas, créditos, orden de publicación y recuperación
después de escrituras parciales. No requiere CUDA, Vivado, red ni hardware.

La suite pasó **25 pruebas**, incluida una prueba con **20 semillas** de
intercalado bidireccional. Esto valida el comportamiento del modelo;
no demuestra DMA PCIe, visibilidad AXI física ni velocidad de un enlace.

## Archivos y ejecución

- [tools/ntrpu_transport_model.py](../../tools/ntrpu_transport_model.py):
  modelo determinista dirigido por eventos, con biblioteca estándar.
- [tests/test_ntrpu_transport_model.py](../../tests/test_ntrpu_transport_model.py):
  comprobaciones de datos, estados, errores y conservación de créditos.

Desde la raíz, con Python 3.10 o posterior:

```bash
python -B tests/test_ntrpu_transport_model.py -v
```

La salida termina en `OK` y devuelve cero solo si todas las pruebas pasan.
Las comprobaciones de unittest y TransportError permanecen activas con
optimización de Python. La prueba es independiente de CMake/CTest para
mantener aislada esta exploración; no se modificó el simulador C++ ni RTL.

## Evidencia local

La [tanda del 2026-10-04](../../artifacts/ntrpu-transport-model/20261004T021032398739Z/result.json)
registra Python, plataforma, comando, duración y hashes de las fuentes
comprobadas antes y después de ejecutar. Las 25 pruebas pasaron; la salida
completa está en [stderr.txt](../../artifacts/ntrpu-transport-model/20261004T021032398739Z/stderr.txt).

El [respaldo ZIP](../../artifacts/ntrpu-transport-model/20261004T021032398739Z.zip)
contiene fuentes, documentación de entrada y resultados. Se verificaron
los ocho miembros por contenido y CRC, con hashes SHA-256 para los archivos
del manifiesto. Su [registro de verificación](../../artifacts/ntrpu-transport-model/20261004T021032398739Z/VERIFIED.json)
identifica el comprimido con SHA-256
`de7e55bed1def05abd10c7a80e7ed3c30b2091c1bd035d15d0e4d6e5f5b58f3a`.

## Modelo de recursos

Hay dos nodos, una cola por dirección y una ventana lógica por receptor.
Cada nodo anuncia por defecto 512 GiB virtuales; solo se materializan
bloques escritos de 64 bytes, hasta 1 MiB por nodo. Registrar una región
grande no asigna toda esa capacidad.

Valores iniciales: cuatro entradas de cola por dirección, dos créditos
de fragmento por receptor, 16 regiones por nodo, fragmentos de 64 bytes
y transferencias de hasta 1 MiB. El tamaño de fragmento es configurable;
admite un fragmento final menor, siempre múltiplo de 64. No representa MPS,
MRRS ni una transacción AXI/TLP. La traza conserva los últimos 4096 eventos.

Los accesos públicos requieren offset y longitud alineados a 64 bytes.
Una región nueva no contiene datos válidos aunque sus bytes físicos
conserven una escritura anterior. `host_write` inicializa rangos; SUCCESS
publica los rangos recibidos. `debug_read_raw` permite observar datos
inválidos exclusivamente para diagnóstico de las pruebas.

Los lectores retienen sus intervalos de origen. Las escrituras en destino
excluyen accesos solapados; PUT distintos pueden avanzar sobre intervalos
disjuntos de una misma ventana. La liberación de región y la reasignación
de ventana esperan el retiro de todas las referencias correspondientes.
Intentar UNREGISTER en uso marca la región como cerrándose y devuelve
REGION_BUSY: las admisiones nuevas se rechazan y el trabajo anterior drena.

## Fases y finalización

1. `submit` valida sesión, secuencia, época, handles, ventana, permisos,
   límites, validez del origen y conflictos. Reserva una entrada de cola.
2. `send` toma un crédito del receptor y captura un fragmento del origen.
3. `receive` entrega ese fragmento a un buffer lógico, sin escribir memoria.
4. `commit` escribe sus bytes y devuelve el crédito. Aún no publica el rango.
5. `confirm_remote` exige todos los fragmentos escritos y publica SUCCESS.
6. `retire` consume la finalización y libera la entrada y sus referencias.

La confirmación no consume créditos de datos; su espacio queda reservado
en la entrada del trabajo. La suite completa un trabajo mientras otro
ocupa el último crédito. El orden de llegada y las finalizaciones pueden
diferir del orden de envío.

FENCE captura la última secuencia admitida de una dirección. Espera esos
trabajos, no los posteriores; un fallo anterior a su límite persiste tras
retirar la finalización. El modelo exige secuencias estrictamente crecientes
por emisor, una restricción mayor que simplemente no reutilizarlas.

## Caídas y escrituras tardías

`fail` detiene admisiones y publica errores con cero bytes válidos para
trabajos incompletos. Los paquetes emitidos siguen existiendo: pueden
llegar y escribir incluso después de retirar esa finalización de error.
Las referencias y los destinos afectados quedan retenidos y no publicados.

`recover` rechaza tráfico sin drenar y finalizaciones sin retirar. El banco
puede drenar los paquetes o usar `revoke_inflight`, que modela una
**revocación ya confirmada por el transporte**. Esta llamada es una hipótesis
del entorno; no demuestra cómo detener un DMA PCIe real.

La recuperación avanza la sesión e invalida registros y ventanas anteriores.
No borra la memoria física simulada ni vuelve válidos los bytes parciales.
La sesión inicial es 1 y el contador vive dentro de una instancia del
modelo: la negociación persistente tras reiniciar un proceso real sigue
pendiente. Los handles son identificadores para un controlador confiable,
no capacidades criptográficas ni un mecanismo de aislamiento entre usuarios.

## Cobertura y límites

Las pruebas incluyen direcciones superiores a 4 GiB y el último bloque
de 512 GiB; overflow, permisos y descriptores inválidos sin escritura;
generaciones obsoletas; colas llenas; ventanas retenidas; duplicados;
fragmentos y completions fuera de orden; escrituras tardías tras timeout;
revocación; agotamiento de memoria y corrupción detectada.

Cada semilla aleatoria intercala seis PUT de 128 bytes entre ambos nodos.
Una referencia de bytes independiente comprueba los destinos y un contador
externo verifica créditos libres más fragmentos en vuelo. Los pasos están
acotados para detectar ausencia de progreso.

Quedan fuera de este modelo: codecs binarios, cálculo de CRC, retransmisión,
temporización, latencia/ancho de banda, arbitraje físico, BAR/IOVA reales,
PCIe/Aurora/AXI, caches, CDC y adaptación a FFN ABI2. La detección de
corrupción se inyecta como evento. `confirm_remote` representa una barrera
de visibilidad cumplida; su implementación debe probarse en cada adaptador.

El [modelo de adaptador PCIe/AXI](PCIE_AXI_TRANSPORT_MODEL.md) ya separa
confirmación del puente, handshakes AW/W/B y visibilidad final. Sus 18 pruebas
y las 25 regresiones del protocolo pasan localmente. El hook de finalización
de fragmentos se extrajo para que ambos modelos compartan créditos y sesiones;
la evidencia anterior conserva su versión original de fuentes.
El [receptor RTL aislado](PEER_RX_GUARD.md) añade 74 casos de simulación de
señales para buffering, AW/W/B, cuarentena y rearme. Su
[síntesis y replay de netlist](PEER_RX_SYNTHESIS.md) están revisados.
La integración RTL completa y la validación de los IP siguen pendientes.
