# Suma de consumos en E: resultado aislado, tanda 986

## Resultado

`20261011T184437Z-V4-ffn-supervised-pcie-symbolic-monitor-predecode-effect-sum-r1`
terminó con código 0 en **157,025 s**. La revisión independiente aprobó
**283.148 filas nuevas**, con estabilidad antes del flanco y equivalencia
de cuatro estados después, sobre los **4.554 bits públicos** de E.

El banco incluye 262.144 combinaciones de controles/perfiles de words,
16.896 casos de bits individuales, 4.096 registros pseudoaleatorios y
reset frío 0/1/X/Z. Conserva el E original como referencia independiente.
Se compilaron seis fuentes defectuosas; todas divergieron en su fila,
fase y comando previstos. El revisor rechazó 16 corrupciones de evidencia.
Los siete proyectos compilaron las seis fuentes esperadas y solo emitieron
el aviso XSIM de entorno de compilador ya identificado.

## Evidencia y respaldo

El archivo de resultados contiene **776.691 bytes y 342 archivos**,
verificados en D: y NFS, SHA-256
`8e7e0513fbec10225378778d5d8c89ca01e54caad5df1c65f2aa357fac7f2c47`.
El snapshot `source-20261011T184325Z.zip` tiene SHA-256
`183c6e7c4305134ba2b9fe3809ec25317babfe05f7bf3fd4c210feae389ab79a`.

La revisión `-independent-effect-sum-review-r1.json` registra
`PASS_ACTUAL_E_EFFECT_SUM_R1`. Se encuentra en
`build/cloud-setup/20261002-board/`; el certificado de recuperación está
en `artifacts/ntrpu-board-vivado/<tanda>/VERIFIED.json`.
El coordinador cerró revisión y respaldo a las **18:48 UTC**.

## Decisión y límites

La suma de 14 bits y saturación única queda calificada como módulo aislado.
Puede prepararse su integración sobre el monitor RQ validado en 983/985,
cambiando únicamente E y conservando las demás etapas y referencias.
Primero se completa la lectura comparativa de timing por etapas y los
dos bancos ya encolados de límites B, sin sustituir sus entradas.

La integración deberá repetir las 183.058 filas del monitor, medir su
síntesis y probar su propia netlist. Este resultado no acredita mejora
de timing, ruteo, PCIe nativo ni hardware.
