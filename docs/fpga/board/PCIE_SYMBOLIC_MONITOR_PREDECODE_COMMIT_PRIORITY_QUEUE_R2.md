# Prioridad RC: recuperación de ejecución R2

La tanda 976 `20261011T162337Z` terminó con código 1 **antes de compilar
o simular**. Vivado interpretó el valor que comenzaba con `-testplusarg`
como una opción de `set_property`. No constituye un resultado del RTL.
El archivo y todos sus miembros quedaron verificados en D: y NFS:
`c005a62e5a865772015b37faf2f28ddc0f9ddad174d518abf0a3632cc4851aed`.

## Corrección limitada

El Tcl R2 cambia una sola instrucción:

```tcl
set_property -dict [list xsim.simulate.xsim.more_options $vector_arg] [get_filesets sim_1]
```

Así se pasa el argumento como valor del diccionario. Se preservaron el
Tcl ejecutado R1, sus helpers y la tanda fallida. RTL, bancos, vectores,
referencias, tres mutantes y criterios de aceptación permanecen iguales.

El nuevo flujo aislado usa archivos `*-commit-priority-r2-*` y revisor
R2. El estado PASS mantiene el sufijo R1 del **candidato RTL**, que no
cambió; el nombre y hash del revisor identifican la revisión de ejecución.

## Dependencias

La integración completa R1 no llegó a arrancar: su coordinador se
detuvo al fallar el requisito aislado. Se creó un coordinador/freezer R2
que espera el nuevo resultado aislado y conserva el candidato, banco,
runner y revisor completos R1. No se altera ningún resultado histórico.

Cadena vigente: replay 975 aprobado y respaldado → ensayo aislado R2 →
revisión y respaldo → 183.058 filas de integración fresca. Síntesis,
replay de la nueva integración y temporización física siguen pendientes.

Registro `pcie-symbolic-monitor-predecode-commit-priority-r2-recovery.json`,
SHA-256 `3ad123ee6b143871a33cb6e05f3f9dea1758f146f2f319bf0aeb50243698464d`.
Flujo R2: `ccda543b2932a254ba7c8448612fc5d31b8a7f78d82a62d6812a3195dce1226b`.
