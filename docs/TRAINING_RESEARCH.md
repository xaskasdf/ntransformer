# Investigación: diseñar el entrenamiento desde sus dependencias

El objetivo es reducir el tiempo para alcanzar una calidad verificable, con un
presupuesto de memoria y energía explícito. Para Brandon conservamos el contrato
actual mientras eliminamos materializaciones. Para otra arquitectura necesitaremos
demostrar primero qué función cambia, cómo se deriva y qué coste se desplaza.

La investigación produjo implementaciones de GQA directo, RoPE y pérdida fusionada, y comprobaciones
numéricas pequeñas de las ecuaciones siguientes. Los resultados de entrenamiento
están en [direct-gqa/RESULTS.md](benchmarks/h100-training-2026-10-13/direct-gqa/RESULTS.md)
y [fused-rope/RESULTS.md](benchmarks/h100-training-2026-10-13/fused-rope/RESULTS.md).
Los dos diseños de bloque de las secciones finales **todavía no se entrenaron**.

## 1. Qué optimizamos

La métrica principal de una arquitectura será el tiempo hasta una pérdida y una
puntuación de instrucciones prefijadas, junto con memoria máxima y energía medida.
Un modelo que procesa más tokens pero necesita más para aprender puede perder.
Para cambios de implementación usamos además el mismo trabajo y la auditoría
numérica completa de 100 pasos. Esos 100 pasos no prueban convergencia de otra red.

Una cota orientativa del paso es

\[
t_{paso}\ge\max(F/P,\;B_{HBM}/W_{HBM},\;B_{enlace}/W_{enlace},\;t_{ruta\ crítica}).
\]

Es una cota, no un modelo suficiente de latencia: los recursos compiten, las
dependencias impiden solapamientos y los tamaños de GEMM afectan su eficiencia.
Los bytes lógicos inventariados en nuestras trazas no son mediciones de HBM.

Con las dimensiones actuales, las matrices de atención y FFN suman 3.428.352
parámetros por bloque. Las 30 capas más el clasificador contienen 107.569.152
coeficientes usados en GEMM; los 35.136 restantes son escalas RMSNorm. El cómputo
forward/backward de esas matrices es aproximadamente **645,4 MFLOP por token**
(6 por coeficiente), sin contar productos de atención, softmax, normalización,
optimizador ni recomposición. Dentro del bloque, FFN representa aproximadamente
74,2% de los coeficientes de esas matrices. Por ello conviene inspeccionar
activaciones/FFN y pérdida además de atención.

## 2. GQA: sustituir una expansión por un índice

Brandon tiene \(H_q=9\), \(H_{kv}=3\) y \(r=3\). Para la cabeza \(h\),
\(g(h)=\lfloor h/r\rfloor\):

\[
P_h=\operatorname{softmax}(Q_h K_{g(h)}^T/\sqrt{d_h}+M),\qquad O_h=P_hV_{g(h)}.
\]

Sea \(R\) el operador que repite cada cabeza KV tres veces. Materializar \(RK\)
y \(RV\) sólo representa ese índice mediante copias. Su adjunto es una suma:

\[
\overline K_g=\sum_{h:g(h)=g}\overline{(RK)}_h,
\qquad \overline V_g=\sum_{h:g(h)=g}\overline{(RV)}_h.
\]

Eso prueba equivalencia sobre números reales. En BF16,
\(\operatorname{round}(\sum a_h)\) puede diferir de
\(\operatorname{round}(\sum\operatorname{round}(a_h))\). Por eso medimos las
capas 0 y 29 con gradientes reales, y luego el modelo completo. La nueva opción
`--direct-gqa` mantiene RoPE y sus frecuencias, proyecciones, máscaras y caché,
y entrega tres cabezas KV directamente a SDPA. El backend efectivo se verifica
en las trazas; no basta con activar `enable_gqa` para demostrar ahorro.

El [artículo GQA](https://arxiv.org/abs/2305.13245) es el antecedente de la
arquitectura. Aquí Brandon ya era GQA: el experimento cambia su representación
en ejecución, no el número de cabezas ni los parámetros.

## 3. Siguiente trabajo que puede desaparecer sin cambiar la arquitectura

| Candidato | Trabajo prescindible | Resultado o comprobación pendiente |
|---|---|---|
| RoPE fusionado, implementado | Materializaciones FP32 y temporales complejos usados sólo para rotar pares BF16. | Seis continuaciones estrictas exactas; +5,45% habitual. El pico estricto/P2P sube unos 0,40 GiB: sigue pendiente aislar esa diferencia. |
| Pérdida fusionada, implementada | Log-softmax y buffers FP32 de NLL completos. | Exacta en seis continuaciones estrictas; reduce el pico estricto 1,153 GiB/rank. Velocidad pequeña y variable. |
| Pérdida por bloques de vocabulario, experimental | Logits y probabilidades completos en memoria global. | Probada: falla el diagnóstico de gradientes por tensor (6,33% frente al 2%); 1,91% de discrepancia de actualización a 100 pasos. Sólo 0,260 GiB adicionales sobre la fusión; no se promueve. |
| Clasificador sólo en posiciones supervisadas | En SFT se calculan logits de filas que luego reciben peso cero. | Mantener todo el contexto en el backbone y seleccionar sólo las filas para el head. No ahorra filas en pretraining sin máscara. |
| Plan de ejecución y buffers persistentes | Reconstrucción del mismo grafo, asignaciones de objetos y consultas al host. | Datos futuros preparados, direcciones estables y eventos del último consumidor; medir primero el tiempo de host expuesto. |
| Prefetch y persistencia | Preparación tardía de entradas y guardados que bloquean. | Instantánea coherente del estado, memoria reservada y coste de evaluación/guardado separado del paso. |

RoPE aplica a cada par \((u,v)\) la matriz
\(R_\theta=\begin{bmatrix}c&-s\\s&c\end{bmatrix}\). Con frecuencias fijas,
su backward es \(R_\theta^T\overline y\): no necesita conservar la activación
de entrada. Un kernel puede leer BF16, efectuar los productos FP32 y escribir
BF16. Fusionar las instrucciones con FMA también cambia el redondeo frente a
productos separados. La implementación comprobó el comportamiento de referencia
y usa explícitamente `fma(u,c,-round(v*s))` y `fma(u,s,round(v*c))`. La misma
expresión con seno negado produce el backward. Los controles FP32/BF16, de
gradientes reales y de estado completo pasaron; no se asume esa equivalencia
para cualquier compilador o hardware futuro sin repetir la conformidad.

La pérdida ofrece otra eliminación. Para \(z_j=h^Tw_j\),
\(\ell=\log\sum_j e^{z_j}-z_y\). Se puede recorrer el vocabulario por bloques
manteniendo sólo el máximo \(m\) y la suma escalada \(s\):

\[
m'=\max(m,\max_{j\in C}z_j),\qquad
s'=e^{m-m'}s+\sum_{j\in C}e^{z_j-m'},\qquad LSE=m+\log s.
\]

Una segunda pasada recupera \(p_j=e^{z_j-LSE}\) por bloques y acumula
\(\overline h=\sum_j(p_j-1_{j=y})w_j\) y
\(\overline w_j=(p_j-1_{j=y})h\). Para una media enmascarada se multiplica cada
fila por su máscara y se divide por el total supervisado. Esta derivación no
filtra probabilidades pequeñas. El ahorro de memoria puede costar recomposición;
los órdenes de reducción siguen sujetos a la auditoría BF16/FP32.

El antecedente es [Cut Cross-Entropy](https://arxiv.org/html/2411.09009v1).
Sus variantes con filtrado de gradientes añaden una aproximación que habría que
evaluar por separado. No importamos sus ganancias para vocabularios mayores al
vocabulario 8.192 de Brandon. Nuestro head produce 256 MiB BF16 por microbatch y
su representación FP32 ocupa 512 MiB; esos buffers motivan una medición propia.

[FlashAttention-3](https://arxiv.org/abs/2407.08608) también es una referencia
para solapar transferencias y cómputo dentro de Hopper. Cambiar el kernel de
atención requeriría otro control numérico; no trasladamos sus cifras publicadas
al paso completo de Brandon.

## 4. Candidato de arquitectura: ramas paralelas con entrada común

Sean \(A\) atención normalizada y \(G\) FFN normalizada. Un bloque secuencial es

\[
s=x+\alpha A(x),\qquad y_s=s+\beta G(s).
\]

El paralelo sería
\(y_p=x+\alpha A(x)+\beta G(x)\). Su ventaja de ejecución es que ambas ramas
leen la misma versión de la entrada. Permite fusionar proyecciones y planificar
sus ramas juntas. Las [capas paralelas de PaLM](https://jmlr.org/papers/volume24/22-1144/22-1144.pdf)
son un antecedente explícito; este mecanismo no sería una invención nuestra.

La diferencia funcional que hay que pagar es

\[
y_s-y_p=\beta\{G(x+\alpha A(x))-G(x)\}
=\alpha\beta J_G(x)A(x)+O(\alpha^2).
\]

Si el Jacobiano de \(G\) es Lipschitz con constante \(L_J\) en el segmento,
el resto tras el término lineal está acotado por
\(|\beta|L_J\alpha^2\|A(x)\|^2/2\). El término de primer orden significa que
no podemos simplemente reinterpretar el checkpoint como si fuera la misma red.

Para RMSNorm sin escala, \(N(x)=x/\rho\),
\(\rho=\sqrt{\|x\|^2/d+\epsilon}\):

\[
J_N=I/\rho-xx^T/(d\rho^3),\qquad
\|J_{\gamma N}\|_2\le\|\gamma\|_\infty/\rho.
\]

Si además se comparte la escala RMSNorm entre ramas, se puede producir una sola
entrada normalizada y proyectar \([Q,K,V,gate,up]\) con una matriz empaquetada.
Compartir esa escala es una segunda decisión de arquitectura; la primera
ablación debe conservar escalas independientes para aislar el efecto paralelo.
Se pueden compartir sus reducciones sobre el mismo \(x\), pero las entradas
escaladas aún son distintas. No se asumirá solapamiento perfecto: las ramas
compiten por Tensor Cores y ejecutarlas juntas puede aumentar el pico de memoria.

Las escalas residuales \(\alpha,\beta\) permiten estudiar la magnitud del
acoplamiento omitido. Una cota local del Jacobiano paralelo es
\(1+|\alpha|\|J_A\|+|\beta|\|J_G\|\); no demuestra convergencia ni permite
elegir escalas óptimas sin medir. Agregar el término JVP de corrección recuperaría
la aproximación de primer orden, pero exige trabajo adicional: podría anular
la ganancia buscada. No lo incorporamos sin comparar su coste.

## 5. Candidato de arquitectura: dos estados reversibles con puntos de restauración

La pregunta es si podemos reconstruir activaciones cuando llega el backward.
Los antecedentes son [RevNet](https://arxiv.org/abs/1707.04585) y
[Reformer](https://arxiv.org/html/2001.04451v2). Un candidato para nuestras formas
mantiene dos estados de dimensión 576, inicializados desde el embedding, y usa
las mismas dimensiones de atención y FFN:

\[
y_1=x_1+\alpha A(x_2),\qquad y_2=x_2+\beta G(y_1).
\]

Su inversa sobre reales es
\(x_2=y_2-\beta G(y_1)\),
\(x_1=y_1-\alpha A(x_2)\). No requiere invertir atención ni FFN. Un readout
inicial a estudiar sería \(N((y_1+y_2)/2)\), manteniendo el embedding/clasificador
atado. Reutiliza dimensiones y casi todo el presupuesto de parámetros; cambia
la función y duplica el estado residual de una frontera.

Con adjuntos \(g_1,g_2\) a la salida, el backward se expresa como

\[
t_1=g_1+\beta J_G(y_1)^Tg_2,\quad
\overline x_1=t_1,\quad
\overline x_2=g_2+\alpha J_A(x_2)^Tt_1.
\]

Los gradientes de parámetros son los VJP de \(G\) con \(\beta g_2\) y de
\(A\) con \(\alpha t_1\). La reconstrucción exige la misma versión de pesos
y, si hubiera aleatoriedad, su reproducción. AdamW sigue esperando a completar
la acumulación/reducción y el clipping global.

La inversión deja de ser exacta en punto flotante: sumar puede destruir bits
que restar no recupera. Por ello el candidato usaría estados residuales FP32,
ramas BF16 y puntos de restauración cada \(k\) bloques. Su inventario ideal de
fronteras escala como \(O((L/k)BTd)\), más dos estados y el trabajo de un bloque;
los intermedios del bloque siguen necesitando recomposición o almacenamiento.
La memoria baja a cambio de reevaluar ramas. Para Brandon, que cabe en las H100,
eso puede perder velocidad; resulta más relevante al aumentar modelo/contexto.

P2P puede alojar los puntos de restauración y anticipar su lectura. La reserva
debe registrar propietario, versión, consumidor y evento del último acceso.
No se restringe el diseño por la futura 3090: allí se evaluará otra ubicación
de esos estados cuando falte el enlace entre GPU.

## 6. Comprobaciones matemáticas reproducibles y criterio experimental

[`analyze_transformer_training_math.py`](../tools/analyze_transformer_training_math.py)
usa NumPy en CPU y semilla 20261014. Sus resultados se guardan en
[architecture-math/results.json](benchmarks/h100-training-2026-10-13/architecture-math/results.json).

* El adjunto de la repetición GQA cumple la identidad de productos internos con
  error absoluto 1,78e-15; el JVP de RMSNorm coincide con diferencias centrales
  dentro de 1,41e-10 en el caso probado.
* La pérdida por bloques coincide con la densa; los gradientes difieren como
  máximo 5,56e-17 en FP64. Esto no prueba igualdad de kernels BF16.
* Al dividir \(\alpha\) por dos, el resto tras la corrección de primer orden del
  bloque paralelo cae por factores 3,98–3,99, coherente con el término cuadrático.
* En un acoplamiento de juguete de 30 bloques, escala residual 1 y ramas FP32,
  cuantizar cada contribución residual y los estados a BF16 produce hasta
  **2,324% de error L2 relativo** al
  reconstruir. Restaurar cada cinco bloques lo reduce a 0,819%, sin hacerlo
  exacto. Con estados FP32 el máximo probado es 3,12e-7 relativo. Son pruebas
  del problema numérico, no mediciones de gradientes o calidad de un Transformer.

Las ramas de ese juguete usan matrices y funciones FP32 incluso en el caso
BF16; no reproduce el redondeo interno de GEMM/atención BF16. Por tanto, el
resultado FP32 tampoco garantiza esa precisión al reconstruir un bloque real
con ramas BF16. Esa combinación requiere su propia prueba.

Antes de entrenar un candidato nuevo: implementar forward/VJP y reconstrucción
en pequeño, comprobar causalidad y gradientes, medir error numérico a profundidad
30 y comparar memoria/cómputo incluyendo toda recomposición. Después usar al menos
tres semillas, mismo corpus/tokenizer/evaluación, presupuesto de parámetros
declarado y dos comparaciones: mismos tokens y mismo tiempo total. Inicialización
nueva con distribución común es la comparación de arquitectura; adaptar el
checkpoint existente sería un experimento de conversión separado.

La prioridad concreta es retirar intermediarios del head y del despacho repetido,
mantener los controles exactos del entrenador actual y preparar la ablación
paralela antes de entrenar otro modelo. También queda aislar el pico de memoria
de RoPE sin instrumentación. La reversibilidad necesita primero una
validación de reconstrucción y gradientes en un bloque de atención real.

## 7. La pérdida por bloques en el modelo real

La comprobación FP64 de la sección 6 ya tiene una contraparte CUDA/BF16:
[target-loss/RESULTS.md](benchmarks/h100-training-2026-10-13/target-loss/RESULTS.md).
La pérdida fusionada que conserva el GEMM completo reproduce los 272 gradientes
y seis continuaciones estrictas exactamente. Retira intermediarios grandes,
con ahorro de memoria y cambios pequeños/variables de velocidad.

Dividir el vocabulario en bloques de 2048 produce el mismo escalar de pérdida
inicial, pero altera el gradiente de entrada al head un 0,02835% relativo.
Mantener `dz` fijo reproduce casi la misma discrepancia: el orden de acumulación
del GEMM por bloques es la causa dominante en ese diagnóstico. Usar un único
bloque de 8192, incluso con salida GEMM FP32 y redondeo final BF16, es exacto.
Desactivar reducciones de precisión reducida tampoco cambia ese resultado
completo; ese flag no queda identificado como causante.

La diferencia puede amplificarse a través del backward del modelo: el peor
tensor alcanza 6,33% relativo, frente al umbral diagnóstico conservado de 2%.
La continuación de 100 pasos pasa las tolerancias generales con 1,91% de
discrepancia de actualización; eso no equivale a identidad ni a validación de
convergencia larga. La variante queda experimental: sólo ahorra 0,260 GiB
adicionales y no demuestra aceleración frente a los controles estrictos.

La lección para el diseño por bloques es preservar el contrato de redondeo y
medir la vida de todos los intermediarios, incluida la recomposición. Una
fusión más profunda de proyección/pérdida necesita planificación Tensor Core;
la equivalencia algebraica de una partición no prueba equivalencia BF16.

## 8. Reutilizar almacenamiento exige conservar sus vidas útiles

La captura de forward/backward del modelo actual ya está medida:
[labels-graph/RESULTS.md](benchmarks/h100-training-2026-10-13/labels-graph/RESULTS.md).
Mejora 3,57% estricto y 4,45% habitual frente a controles nuevos; conserva
exactamente las ocho continuaciones estrictas. El número de kernels GPU sube
ligeramente, mientras los lanzamientos desde CPU bajan 85,1% en los perfiles.
Esto valida retirar despacho repetido sin cambiar la arquitectura matemática.

La primera implementación expuso una condición necesaria del plan de bloques.
Si el buffer reutilizado contiene \(g_1\) y también es el acumulador, escribir
\(g_2\) antes de sumar destruye \(g_1\): termina en \(2g_2\), no en
\(g_1+g_2\). Reservar una dirección no garantiza que pueda asumir ambos roles.
La prueba 1+3 produce 6 con alias y 4 con acumulación independiente. Los
gradientes del modelo completo y el preflight DDP verifican la corrección.

Para una reserva especulativa, cada bloque debe declarar productor, consumidores
pendientes y momento de reutilización. Capturar o fusionar la acumulación exige
mantener esa separación, el orden FP32 y la identidad de los pesos ligados.
Las sumas y divisiones DDP que quedan fuera del grafo son el siguiente candidato
medible; todavía no se ha demostrado una mejora al agruparlas. Tampoco se debe
interpretar la vida corta del tensor como memoria libre: el pool privado del
grafo permanece reservado durante replay.

## 9. Acumulación explícita como contrato del plan

Sea \(g_{r,m}=\nabla_\theta(\ell_{r,m}/M)\) el gradiente FP32 producido
por el microbatch \(m\) del rank \(r\), con \(M\) microbatches por paso.
La variante por buckets conserva dos órdenes:

\[
A_r^{(1)}\leftarrow g_{r,1},\qquad
A_r^{(m)}\leftarrow\operatorname{fl}(A_r^{(m-1)}+g_{r,m}),\qquad
G\leftarrow\operatorname{AllReduceSum}(\operatorname{fl}(A_r^{(M)}/W)).
\]

Cada suma mantiene su orden; la división por los \(W\) ranks precede a NCCL.
No se desplaza la división después de la reducción. Agrupar llamadas no exige
cambiar esa aritmética. El primer microbatch sobrescribe el destino: elimina
el cero previo y su lectura en una suma, con almacenamiento independiente del
gradiente temporal que reutilizará el siguiente replay.

Los 107.604.288 parámetros únicos ocupan 430.417.152 bytes de gradientes FP32.
Las vistas `parameter.grad` apuntan a esos bloques persistentes. La identidad
del embedding ligado se conserva enumerando parámetros únicos; las vistas no
se convierten en parámetros nuevos ni cambian AdamW. Se rechazan cambios de
propietario, de stream y reducciones antes de completar los microbatches.

El contrato actual libera todos los gradientes al final del backward capturado;
por eso se ensayan tanto 18 buckets de capacidad nominal 25 MiB como un solo
bucket de 410,5 MiB. Ninguno adelanta todavía la comunicación por capas dentro
del grafo. Es una decisión de este calendario, no una conclusión de que un
único bloque sea óptimo para modelos mayores o para un backward segmentado.

La prueba ya tiene resultados en
[gradient-buckets/RESULTS.md](benchmarks/h100-training-2026-10-13/gradient-buckets/RESULTS.md):
18 continuaciones, diez estrictas exactas y ocho trazas. La variante de un
bloque mejora 1,27% habitual y 1,14% contra DDP agrupado estricto. Retira 95,62%
de las llamadas de lanzamiento restantes y 13,60% de kernels GPU, con cuentas
de kernels de matriz idénticas. Ese contraste cuantifica el límite de seguir
retirando sólo despacho para este modelo: el siguiente cambio debe medir el
trabajo dentro del grafo o un solapamiento nuevo, no suponer una aceleración
proporcional al número de llamadas eliminadas.
