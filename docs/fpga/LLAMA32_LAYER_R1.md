# Capa completa Llama 3.2 1B: integración R1

Fecha: 2026-10-16. **Control aprobado en Icarus; síntesis y protocolo XSim
encolados. La aritmética conjunta todavía no está validada.**

## Composición y propiedad

[`ntrpu_llama32_layer_r1.sv`](../../fpga/rtl/ntrpu_llama32_layer_r1.sv)
conecta entrada/RMSNorm/QKV → RoPE/KV/atención → cuantización/O →
residual/RMSNorm/Q8 → FFN/residual. Reutiliza los bloques reales existentes
y una sola autoridad de caché KV; no instancia otro motor de atención en
la cola de salida. Su generador fija los hashes de los seis hijos directos.

La [admisión global](LLAMA32_LAYER_PREFLIGHT_R1.md) captura y verifica
24 regiones antes de iniciar el productor. La entrada original alimenta
también el residual posterior a O. El padre conserva las reservas durante
toda la operación, aunque ya haya liberado la respuesta del validador.
La consulta de prefijo continúa apuntando a la capa retenida mientras
avanza FFN. La publicación del nuevo prefijo corresponde al productor.

Hay cuatro interfaces AXI anteriores y una interfaz `tail_axi` compartida
por O, post-O y FFN. El propietario sólo cambia en la admisión del bloque
siguiente. La confirmación pública libera primero post-O/FFN; después de
su drenaje libera O y finalmente la entrada. La memoria de atención puede
liberarse antes, una vez consumidas sus 2.048 palabras por cuantización.
Las regiones externas siguen reservadas hasta el drenaje final.

Cada límite comprueba identidad completa, estado y tamaño del resultado.
Los fallos transitorios quedan retenidos. Un fallo posterior a una respuesta
correcta invalida inmediatamente `result_valid`; conserva el estado ya
presentado y activa `completion_revoked`. Una identidad ajena no recibe ACK.

## Validación nativa y límites

El banco R5 usa el validador y el handoff post-O/FFN reales, más cuatro
pares aritméticos **sintéticos**. Pasaron 55 trabajos por semilla (1, 7, 42):

- Secuencia completa, lectura de 2.048 palabras y reutilización de prefijos
  entre posiciones 0 y 1 sin reinicio de la vinculación KV.
- Rechazo de cada región mal dimensionada antes del primer productor.
- Certificados incorrectos, cancelación y cambio de época en cada etapa.
- Contrapresión hasta timeout, identidades de 64 bits alteradas y fallos
  tardíos de caché, O, post-O y FFN.
- Captura de descriptores frente a cambios del host, conexión de todos los
  campos del multiplexor AXI, retención y orden de confirmación/drenaje.

Doce mutaciones RTL provocaron aserciones semánticas explícitas; no se
aceptaron expiraciones como prueba de detección. El revisor rechazó 36
transcripciones corruptas. Son **165 trabajos de control**, no inferencias
ni medidas de ancho de banda. Los patrones AXI comprueban conexiones.

La jerarquía completa también compiló con 83 archivos, ocho interfaces
HLS sin ejecutar y dos adaptaciones de declaraciones para Icarus. Persisten
22 advertencias de enumeraciones implícitas en controles heredados;
esa compilación sólo comprueba integración de interfaces. Vivado conserva
las fuentes originales y debe elaborar/sintetizar la aritmética real.

Nueve tandas, incluidos diagnósticos, tienen archivos y fuentes restaurados
y verificados en D:, con respaldo NFS. R1 del banco falló por el nombre
`bind_capacity`; R2 exigía equivocadamente las regiones completas donde
el handoff entrega subregiones limitadas. La primera tanda de mutaciones
motivó una aserción directa del orden de ACK de O. El RTL candidato no cambió.

Revisión: `build/cloud-setup/20261002-board/llama32-layer-native-r1-review.json`,
SHA-256 `a9e4816070a2fd160be914db74df98eda395f570c07056cf1b15bc3bb95c773f`.

## Turno Vivado y siguiente paso

Paquete `20261016T091236Z-llama32-layer-vivado-r1`, PID 11872, añadido
después del validador de memoria. Conserva los 27 propietarios anteriores.
Sus 238 fuentes de hardware contienen 75 RTL y 163 Verilog HLS de ocho
familias; el banco de control utiliza un proyecto separado de cinco archivos.
Límite: dos CPU, 3.600 segundos, reloj OOC de 5 ns, VU5P. No hay reintento
automático. Tanto éxito como diagnóstico deben archivarse y restaurarse
antes de liberar el slot.

ZIP fuente SHA-256:
`730b016a8b56488590b11fd45af9d1e8656fb8aed245cd6a9e65ae89e84848f0`.

Falta ejecutar esta tanda y la [comparación aritmética completa](LLAMA32_LAYER_REAL_R1.md),
ya preparada y encolada detrás de ella, desde la entrada al residual final. También faltan el arbitraje de
memoria externa, las 16 capas y logits, y el cierre físico. **Esta integración
no aporta SAIF, consumo nuevo, generación de texto ni timing ruteado.**
