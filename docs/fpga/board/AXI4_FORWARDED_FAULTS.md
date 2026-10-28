# Escrituras AXI4 malformadas durante reenvío

## Resultado vigente

El banco R2 `20261006T024145Z-V4-ffn-supervised-axi4-section-forward-r2` pasa **ocho casos** sobre los módulos candidatos
R2, sin modificar RTL de producción. Usa el Clock Converter AMD real
512/64/4, H/D=7/5 y 7/37 ns y gestión a 10 ns. La revisión independiente
rechaza **210 alteraciones de evidencia**.

| Caso, repetido en ambos perfiles | Infracción externa |
| --- | --- |
| AW antes de W | Descriptor de cuatro beats, WLAST en el segundo |
| W antes de AW en la fuente | Mismo desacuerdo, descubierto al emparejar |
| Ráfaga de 256 beats | AWLEN=255/AWSIZE=4 válido y ausencia de WLAST |
| Cierre con W parcial | Un beat pendiente al entrar en FINISH, luego WLAST anticipado |

Todos los descriptores quedan dentro de una página de 4 KiB. Los strobes
del caso de 256 beats rotan por grupos de 16 bytes; se comparan los 512 bits
de datos, IDs, atributos y límites de cada transferencia.

Antes de introducir el fallo, el guard retiene una B sana y el último beat
R de una lectura normal de 16 beats. Las conserva durante las alarmas y
ambas terminan al liberar READY. Cada caso entrega a D dos AW, una AR y
18 o 272 beats W; solo produce la B y los 16 R de la operación normal.
No se fabrica una respuesta para la escritura corrupta.

## Inventario y certificado

Admisión registra `af=4`, y ambos monitores `hf=df=8`: únicamente el fallo
de escritura previsto. Las tres fuentes quedan retenidas en gestión
(`fm=66564`). Llegan en **21.9–29.5 ns simulados**,
incluso con su reloj de origen detenido. Una generación/tag nuevos no
limpian el fallo ni autorizan el certificado.

La admisión permanece en `FINISH`. En las ráfagas cortas termina el prefijo;
en la ráfaga sin WLAST conserva una escritura por terminar y no afirma
`source_fenced`. El guard congela AW/W/parcial en 2/2/0 o 2/1/255, incluso
después de consumir la B sana. Estos son **contadores conservadores tras
corrupción**, no dos obligaciones reales todavía pendientes. La historia
externa reconstruida deja una escritura fallida; la respuesta sana ya terminó.
El banco no exige vaciar ni resetear inventarios que dejaron de ser fiables.

## Diagnóstico de reloj conservado

`20261006T023743Z-V4-ffn-supervised-axi4-section-forward` emitió marcas HDL
de éxito para los dos primeros casos lentos, pero agotó el límite de 100 µs
en el caso de 256 beats. S y H habían aceptado 272 beats W contando la
operación normal; D solo había aceptado 271. H se detuvo un picosegundo
después de registrar su fallo y no volvió a tener flancos. D mantuvo su
contador parcial en 255 y no podía detectar todavía la ausencia del WLAST final.

No se deduce pérdida ni ubicación interna del beat a partir de ese timeout.
R2 verifica primero que la alarma de H llega a gestión sin otro flanco H;
después reanuda H, permite que llegue el beat pendiente y pausa D tras su
propia alarma. Conserva las aserciones, el estímulo y la comparación completa.
Los ocho casos revisados entregan todos los beats aceptados por la fuente.

También se revisa el orden por canal: W antes de AW en S/H puede aparecer
en otro orden en D por las latencias independientes. El comprobador exige
conservación y causalidad de cada canal, sin imponer un orden entre canales
que el protocolo no garantiza.

## Control negativo y respaldo

`20261006T025005Z-V4-ffn-supervised-axi4-section-forward-mutant` cambia una expresión en una copia privada del guard: retira
BVALID al aparecer el fallo W. El banco positivo R2 sin cambios falla su
aserción de respuesta sana retenida. La traza independiente identifica B
con ID 3/RESP 0 ya aceptada en H, nunca consumida en S, que desaparece
con BREADY=0. Solo se califica este control negativo concreto.

| Evidencia | SHA-256 del archivo |
| --- | --- |
| Diagnóstico inicial | `c0dd26558dbf06af7548fd3fc537dfae1ba286553fec8fd59f21cd3e64bc340d` |
| Ocho casos R2 | `d4f90ff7a98480e8f87ccd5adf4074acce641ee8f16dc88f5033d1f8a8ec9e73` |
| Mutación del guard | `1cad6fcae9c7d4325851973414c6095606c2f852b99e96a29ef7f347fa282060` |

Adenda revisada: `8d4706119a4180650069da32d05099cfdb676e5ea6f739d93cba1589b51257c2`. Fuentes, compilación, bibliotecas,
logs y archivos están verificados localmente y en NFS.
Ejecutor: `tools/ntrpu_board_axi4_section_forward_r2.tcl`.
Revisor: `tools/check_ntrpu_axi4_section_forward_r2.py`.

## Dependencias restantes

El [análisis de capacidad defensiva](AXI4_FORWARDED_FAULT_PLAN.md) sigue sin
prueba formal. Se conserva como tal; no se fuerza una alarma interna para
simular alcanzabilidad. La siguiente integración debe identificar el contrato
público de parada del productor XDMA y sus colas protegidas. Esta sección
no autoriza reset del endpoint ni reutilización de propiedad. El timeout
nativo de 50 ms se conserva. CDC-15, restricciones físicas, DIMM exacto y
placa completa siguen abiertos; el FIFO funcional AMD omite retardos de
sincronización de datos y estas latencias no son medidas físicas.
