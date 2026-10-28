# Integración C/E/D/A/B/R y certificado de inventario vacío

## Estado y dependencia

Diseño pendiente de implementar y verificar. Conservar las revisiones
inmutables C3/E2/D2/A1/B1/R1 y sus pruebas. La ejecución de la integración
depende de revisar la síntesis y reproducción de la netlist propia de R.
No implica aprobación física ni sustituye las pruebas nativas.

## Composición y observabilidad

Crear un contenedor con las seis instancias y conexiones tipadas de
`ntrpu_pcie_symbolic_frontend_r3`, A, B y R. C ya expone `observing`,
`early_fault_sources` y `control_pending`: conectar esos puertos para
vigilar el estado actual, sin referencias jerárquicas en RTL sintetizable.
Las salidas públicas proceden exclusivamente de R y conservan la latencia
de cinco flancos respecto al origen capturado por C.

El banco debe observar las instancias de ese contenedor. Sustituir las
instancias paralelas A/B/R y adaptar las observaciones C/E/D al mismo
camino; conservar todos los estados, tablas y comprobaciones de origen.
Las referencias jerárquicas de diagnóstico pertenecen al testbench.

## Reglas de vacío

- Guardar requester y máscara de atributos al armar C, en el mismo
  flanco y con la misma condición. No aprender identidad desde RC.
- Bloquear vacío inmediatamente ante actividad RQ/RC, configuración
  distinta, perfil/link/reset inválidos o desconocidos, ready desconocido
  o una máscara RQ ready distinta de `0000`/`1111`.
- Exigir frío desactivado conocido, publicación válida y observación
  activa; fallos actuales de C y retirados de R a cero conocido.
- Mantener pendiente el control vivo de C y las obligaciones de tráfico
  en C/E/D/A/B. Tratar un pendiente desconocido como ocupado.
- Exigir estado retirado vacío: tags, lecturas, bytes y parciales a cero.
  La consistencia del bitmap también debe comprobarse contra R7.

`R.effect_pending` etiqueta el origen que acaba de retirarse: no representa
trabajo futuro. Tampoco usar sin interpretación el `control_pending`
capturado dentro de R, que contiene validez de tráfico de ese origen.
Las publicaciones ociosas no deben mantener el inventario ocupado.
Solo frío cancela obligaciones; los demás cambios revocan el certificado.
Este observador pasivo no acredita colas internas PCIe ni recepción remota
de escrituras posted.

## Pruebas de aceptación

Construir el oráculo desde R7/S1 y cinco posiciones de historia; nunca
desde pendientes o salidas del candidato. Comprobar antes y después del
flanco. Incluir arming, tráfico detenido, último retiro, reutilización de
tag, terminal con payload retenido, fallos tempranos/tardíos, cambios de
configuración, X/Z, frío ocupado y vaciado completo.

Exigir testigos de revocación inmediata y recuperación exacta en el último
retiro, con mutantes que omitan guardas o etapas pendientes y otro que
mantenga ocupado un ciclo adicional. Mantener los bancos originales y
su contabilidad explícita de cancelaciones. Medir después la composición
real; no sumar recursos o frecuencias de módulos aislados.
