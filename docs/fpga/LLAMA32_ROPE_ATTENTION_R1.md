# RoPE, caché KV y atención bajo una admisión común

Estado 2026-10-16 05:13 UTC: **admisión comprobada en Icarus; síntesis
completa encolada; aritmética conjunta todavía pendiente**.
El [wrapper R1](../../fpga/rtl/ntrpu_rope_attention_r1.sv) une el
[frente RoPE/captura/despacho](LLAMA32_ROPE_QKV_DISPATCH_R1.md) con el
[root de 32 cabezas](LLAMA32_ATTENTION_ROOT_XSIM_SYNTH_RESULTS_R2.md).
Incluye el append, caché, lector KV, guardia y controladores reales.
No modifica sus implementaciones previamente archivadas.

Actualización 2026-10-16 05:40 UTC: el [banco aritmético conjunto](LLAMA32_ROPE_ATTENTION_REAL_R1.md)
quedó compilado y encolado después de esta síntesis, PID 18736. Usará ambos
HLS originales y KV escrito por el diseño: ocho trabajos, cinco comandos
exitosos previstos y 160 cabezas. El revisor pasó 168 corrupciones
sintéticas; **la aritmética conjunta continúa pendiente de XSim**.

## Interfaz y propiedad

El [generador](../../tools/build_ntrpu_rope_attention_r1.py) clasifica
151 puertos públicos y verifica dirección y ancho de 59 conexiones entre
hijos. La interfaz ofrece tres maestros AXI independientes: `rope_axi_*`
para Q/K/coeficientes, `v_axi_*` para V y `kv_axi_*` para lectura/escritura
de caché. El interconector de memoria aún pertenece a la integración superior.

Una vinculación pendiente tiene prioridad sobre un comando nuevo. Una vez
admitido el comando, ninguna vinculación entra hasta liberar atención,
captura y fuente RoPE, incluido el drenaje posterior al ACK público.
`busy` y `cache_exclusive` también cubren mantenimiento y respuestas de
vinculación retenidas. `bind_completion_*` publica exclusivamente respuestas
de vinculación; los ACK de append permanecen dentro del frente de despacho.

Las épocas de vinculaciones exitosas deben crecer estrictamente. Se puede
presentar una solicitud futura mientras el frente está ocupado, manteniendo
su descriptor estable; `expected_epoch` debe actualizarse después del drenaje
y antes de aceptar esa vinculación. Cambiarlo durante un trabajo revoca la
autoridad de ese trabajo. El reset exige quiesce coordinado de los tres
maestros, memoria y todos los hijos; no se ha validado reset unilateral.

## Prueba de admisión y alcance

El [banco R2](../../fpga/tb/tb_ntrpu_rope_attention_admission_r2.sv) ejecutó
tres semillas: **18 comandos rechazados**, 45 vinculaciones, 42 ACK de
vinculación, 18 ACK públicos y seis resets coordinados. Comprueba prioridad
simultánea, respuesta de vinculación retenida, exclusión durante trabajo,
respuesta pública y drenaje, capacidad inválida, época repetida y reset
con una vinculación aún sin ACK.

Los comandos se rechazan por identidad fuente, posición fuera de rango o
fuente inválida, antes de leer memoria. Los 25 módulos RTL de producción
están presentes. Dos centinelas de prueba abortan si se inicia cualquier
núcleo aritmético: **esta tanda no ejecuta RoPE ni atención numérica** y no
contiene comandos de inferencia exitosos.

El [revisor independiente](../../tools/check_ntrpu_rope_attention_admission_r1.py)
reconcilia identidades, orden, estados y ciclos de retención/exclusión.
Rechazó 189 corrupciones. Cinco variantes RTL defectuosas fallaron por
aserciones explícitas. Una sexta, que retira el chequeo externo de
`cache_idle`, pasó también el revisor: el despachador interior mantiene
ese mismo chequeo. Se registra como superviviente; no se cuenta como fallo
detectado ni como prueba general de equivalencia.

El primer banco reutilizó una época exitosa y esperaba aceptación. El motor
la rechazó correctamente. R2 corrige el estímulo a épocas crecientes y añade
el rechazo explícito de la época repetida. El RTL del wrapper permanece igual.

## Evidencia y cola

Tres tandas con fuentes y resultados restaurados en D:, y copias verificadas
en NFS:

| Tanda | SHA-256 del TAR |
| --- | --- |
| `20261016T050148Z-V7-rope-attention-admission-r1` | `944254b33e5bb4bf70c4161ecbc2ff37378016d9b5c459ed1d087cbf7c8c80f5` |
| `20261016T050405Z-V7-rope-attention-admission-r2` | `9207c2a023c516633fdbc8c5b743213e1ba34193661c6177791c0ff44b8ff8ac` |
| `20261016T050653Z-V7-rope-attention-mutants-r1` | `a238caded4867e3fab11cb7116cc20dd673300d9df5b74cf18100b190b82897d` |

Revisión: `build/cloud-setup/20261002-board/rope-attention-admission-r1-review.json`,
SHA-256 `9f6304fc0f4dee79cf05dec811be29717a2070cc075364e245fd044fa3e2023f`.

Síntesis encolada después de reutilización KV, PID 26844:
`20261016T051132Z-rope-attention-synth-r1`, 105 archivos, 66 fuentes
compilables y diez IP. Contiene los **41 Verilog HLS originales** de RoPE y
atención; no compila bancos ni centinelas. ZIP SHA-256
`dff21e658713a2ba4d4c5cec09ad4825f70d71ab182ab06f98f6124aea6e2231`.
VU5P, 5 ns OOC, dos CPU, límite total de 30 minutos. Recursos y WNS serán
estimaciones de síntesis; no cierre físico. Cada resultado o diagnóstico
se respalda y restaura antes de liberar el slot; no hay reintento automático.

Falta simular la ruta positiva completa con ambos HLS, comparar las salidas
numéricas y las lecturas KV con los Q/K/V realmente publicados, y unir este
frente con proyección O/FFN. DDR R1 conserva su turno tras BRAM; sigue sin
SAIF ni estimación nueva de consumo.
