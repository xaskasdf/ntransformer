# Vivado/Vitis en una estación remota

> Estado actualizado (2026-10-01): licencia instalada; INT8 pasa co-simulación C/RTL. Q8 y cuantizador siguen en validación; el cuantizador no cumple 5 ns. Véase [evidencia HLS](validation/hls-20261001.md). Las menciones anteriores a licencia pendiente describen el bloqueo histórico.

El propósito del cloud es sintetizar, simular e implementar NTRPU con herramientas AMD. Brandon es el modelo pequeño de validación. El contenedor PyTorch de `docker/ntrpu-reference/` sirve únicamente para comparaciones numéricas y no instala Vivado/Vitis.

## Recursos y elección

La instancia propuesta, 30 vCPU y 200–220 GiB RAM, tiene margen frente a los requisitos iniciales de HLS. AMD recomienda 64 GB para desarrollo HLS y contempla 200 GB de disco para una instalación Vitis completa; el consumo varía con los componentes. Esto no garantiza memoria suficiente para cualquier implementación completa U50/U250: medir el pico de RAM con el dispositivo y diseño seleccionados. [Requisitos Vitis 2025.2](https://docs.amd.com/r/2025.2-English/ug1742-vitis-release-notes/Installation-Requirements).

Opciones publicadas por [Lambda](https://lambda.ai/pricing), consultadas el 2026-10-01:

| Instancia | vCPU | RAM | SSD | USD/h, antes de impuestos |
| --- | ---: | ---: | ---: | ---: |
| A100 SXM 40 GB | 30 | 220 GiB | 512 GiB | 1,99 |
| A10 24 GB | 30 | 226 GiB | 1,3 TiB | 1,29 |

Para nuestro flujo HLS, priorizar CPU, RAM y disco; no depende de CUDA ni necesita acceso a la GPU. Por capacidad y precio, la A10 es la primera opción entre estas dos si hay disponibilidad. La ficha no permite asegurar igual rendimiento de CPU: comprobar modelo y frecuencia antes de comparar tiempos. Los 30 vCPU tampoco implican acelerar una compilación 30 veces.

El SSD de 512 GiB permite comenzar con una instalación seleccionada y el núcleo actual, pero hay que contar instalador, temporales, checkpoints y ondas de simulación. El disco mayor de la A10 deja más margen. Mantener repositorio e informes fuera del ciclo de vida de la instancia; conservar una copia externa antes de terminarla.

## Entorno propuesto

El informe recuperado usa **Vivado/Vitis 2025.2**. Para la nueva VM, el usuario proporcionó el instalador **2026.1**: conservar esta diferencia en los manifiestos y comprobar el flujo Tcl con las herramientas instaladas antes de comparar resultados. Seleccionar una distribución admitida y verificar la imagen realmente disponible. Lambda permite seleccionar otras imágenes base al crear la instancia: [gestión de instancias](https://docs.lambda.ai/public-cloud/on-demand/creating-managing-instances/).

Preferir inicialmente instalación nativa en la VM y ejecución por SSH/batch. Si se encapsula en Docker, montar instalación AMD y proyectos en volúmenes persistentes, comprobar dependencias y compatibilidad, y ejecutar sin `--gpus`. No se ha construido ni validado un contenedor AMD en este repositorio.

Obtener el instalador mediante la cuenta AMD, seleccionar Vivado/HLS y soporte para la pieza elegida. Verificar edición y licencias del dispositivo/IP antes de una implementación completa; disponer de HLS no demuestra que todas las etapas estén licenciadas. En 2026.1, la tabla de [dispositivos por licencia](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Device-Availability-by-Subscription-Tier) incluye XCZU7EV en Basic y enumera Alveo bajo su licencia específica. Basic requiere generar una licencia gratuita con renovación anual; la activación efectiva queda pendiente de probar en la VM.

## Primer trabajo remoto

Copiar este árbol de trabajo, incluidos los archivos nuevos de FPGA, a la VM. Cargar el `settings64.sh` de la instalación real y comprobar:

```bash
vitis-run --version
vivado -version
python3 --version
free -h
df -h .
lscpu
```

Desde la raíz del repositorio, usar primero la pieza del informe recuperado como referencia de reconstrucción; no es una selección de placa final:

```bash
python3 scripts/run_ntrpu_hls.py \
  --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage csim
python3 scripts/run_ntrpu_hls.py \
  --part xczu7ev-ffvc1156-2-e --clock-ns 5 --stage cosim
```

Ejecutar la segunda orden solo si la primera pasa. `cosim` incluye C simulation y síntesis HLS. Revisar el manifiesto, `tool.log`, utilización, latencia/II y resultado RTL en `build/ntrpu-hls/`. Después se puede usar `--stage export` para generar el IP. El [flujo HLS](HLS.md) guarda cada ejecución en un directorio nuevo.

El primer entregable remoto es un informe de síntesis y una co-simulación correctos del GEMV. La integración Vivado con memoria/PCIe y el place-and-route completo requieren después el dispositivo, constraints y diseño de tarjeta. No se necesita una FPGA conectada para esa compilación; la validación física sigue siendo un hito separado.

## Preparación remota realizada el 2026-10-01

El usuario aprovisionó una VM A10 y autorizó el acceso SSH. Se verificaron 30 vCPU Intel Xeon Platinum 8358, aproximadamente 222 GiB de RAM visibles y 1,4 TiB de disco local. El sistema informa Ubuntu 24.04.5; esa revisión concreta no figura en la matriz 2025.2 consultada, por lo que la compatibilidad AMD debe comprobarse durante la instalación y primera ejecución.

Se instalaron compilador C/C++, CMake, Ninja, soporte de entornos Python y utilidades de transferencia/archivos. Se transfirió una instantánea de 107 archivos del árbol de trabajo, incluidos cambios sin commit, y se comprobaron sus hashes en destino. El simulador compiló en Release y pasó **45/45 pruebas** en la VM.

| Ruta remota | Uso |
| --- | --- |
| `~/ntrpu/current` | Copia activa del proyecto en disco local |
| `~/vivado/ntrpu/snapshots/` | Archivo de fuentes persistente con manifiesto |
| `~/vivado/ntrpu/results/` | Pruebas e informes persistentes |
| `~/vivado/ntrpu/setup/` | Inventario y registro de preparación |
| `~/vivado/ntrpu/installers/` | Destino preparado para instaladores AMD |
| `~/ntrpu/VM_READY.md` | Notas operativas y próximo comando |

Se preparó el flujo `cosim` con `--dry-run` y se copiaron también los informes de preparación al equipo local. **Vivado/Vitis aún no está instalado; no hubo síntesis ni co-simulación AMD.**

## Instalador 2026.1 preparado

**Actualización posterior:** instalación finalizada correctamente; `vitis-run --version` y `vivado -version` responden con 2026.1, build 6511674. La primera prueba HLS falló en `set_part` por ausencia de licencia Vivado, antes de compilar el banco C++. El token de descarga funcionó y no requiere regenerarse para resolver este error. Generar e instalar Vivado Basic para la VM es el siguiente requisito; el flujo todavía no tiene resultados de síntesis ni co-simulación.

Se transfirió `FPGAs_AdaptiveSoCs_Unified_SDI_2026.1_0616_1700_Lin64.bin` (413.313.372 bytes) al directorio persistente de instaladores. El SHA-256 coincide entre origen y destino: `6583178a487800ed5cf34cacea9d0e3887c30fe688f2b5f5b215a0e910793c93`. La comprobación interna de extracción también pasó; esto verifica integridad de transferencia, no una firma del fabricante.

El cliente está extraído en `~/ntrpu/installers/2026.1/`. Se generó `~/vivado/ntrpu/setup/install_config-2026.1-ntrpu.txt` para Vitis Unified, que incluye Vivado y HLS, con Zynq UltraScale+ MPSoCs y dispositivos Alveo/edge. Destino: `~/vivado/AMDDesignTools/`. DocNav, IP Cache y accesos de escritorio están desactivados.

Se revisó `installLibs.sh` y se instalaron sus dependencias disponibles en Ubuntu 24.04 mediante `setup/install-deps-2026.1-ubuntu24.sh`. Se conserva el original con sus retornos CR y una copia normalizada. Los paquetes antiguos `libtinfo5`/`libncurses5` no están disponibles en los repositorios de esta VM; comprobar bibliotecas faltantes al ejecutar las herramientas, sin crear enlaces entre ABI incompatibles. La revisión Ubuntu 24.04.5 tampoco aparece expresamente en la matriz AMD 2026.1 consultada.

La descarga requiere un token de cuenta AMD, distinto de una licencia Vivado. Generarlo directamente en una terminal interactiva, sin guardar credenciales en scripts:

```powershell
ssh -t ubuntu@<hls-host> "cd /home/ubuntu/ntrpu/installers/2026.1 && ./xsetup -b AuthTokenGen"
```

Según [UG973](https://docs.amd.com/r/en-US/ug973-vivado-release-notes-install-license/Acquire-Authentication-Token), el token dura siete días. Después, ejecutar en la VM `bash ~/vivado/ntrpu/setup/install-amd-2026.1.sh`. El script registra la instalación y evita ejecuciones simultáneas; no genera licencias. Verificar `settings64.sh`, versiones, bibliotecas y ejecutar primero `csim`, después `cosim`.
