# Recuperación R5: transporte SSH

## Fallo conservado

R4 cerró la revisión de B995, congeló el replay y respaldó sus fuentes.
El runner falló al verificarlas remotamente con `WinError 206`: el comando
excedía el límite de Windows. Todavía no había ID de tanda ni lanzamiento
de Vivado. El owner, la configuración, el log y el coordinador detenido
se preservaron. R5 conserva la configuración congelada, SHA-256:
`82bdef88123e9b2a5e7b3d959ef7e33d1313a7dfa26398a2c1e5651d6db30f54`.

El adaptador envía comandos a `bash -se` por stdin en bytes UTF-8. Su
prueba pasó un mensaje de 50.000 bytes y comprobó la propagación de salida
23. La primera prueba detectó conversión CRLF de Windows; el adaptador R2
la elimina y conserva R1 como evidencia del fallo.

## Reanudación y límites

Para B solamente, se valida el owner detenido sin `run` y se ejecuta el
AST original desde `save`, tras cargar importaciones, configuración y
comprobación del hash del runner. Se omite solo la creación inicial del
owner existente. Se vuelven a comprobar slot, fuentes y netlist antes de
Vivado. Un recibo registra owner anterior, hash del AST y adaptador.
Las etapas siguientes ejecutan sus runners originales completos.

La tanda B `20261011T210603Z` inició a las 21:06 UTC. El resto conserva
orden y criterios de R4 con coordinadores R5 nuevos. Síntesis y replay E/D
se añadieron después, sujetos a los cierres revisados y respaldados.

## Evidencia

En `build/cloud-setup/20261002-board/`:

- Plan `pcie-symbolic-monitor-predecode-pipeline-recovery-r5.json`:
  `b494eb6afc6fc53aaeba2c7469b1d99c86d41965e32bd236284d666139b87403`.
- Auditoría `pcie-symbolic-monitor-predecode-pipeline-recovery-r5-audit.json`:
  `b59b4aab4719b9fa27f8c64adbd6c598f234b847572ceb0618f21d57808a233e`.
- Recibo `pcie-symbolic-monitor-predecode-ssh-stdin-r2-20261011T210548Z.json`:
  `d88919f37c62b3e594efd63ee1d1e88dca1726a0fd13d1df9af80ac456718924`.

El respaldo `symbolic-monitor-predecode-working-r17-20261011T210523Z`
verificó 1.399 miembros y 42.203.367 bytes en D: y NFS:
`2d5c5119590739e96249125327bb83f36ee03888ae31145ef28a468ae43d4fc1`.
Esto acredita recuperación y procedencia; la simulación sigue pendiente.
