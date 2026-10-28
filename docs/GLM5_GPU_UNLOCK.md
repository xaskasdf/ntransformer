# GPU-Unlock × GLM-5.3-Flash — plan de iteración nocturna

> Objetivo: instalar Zagitta/gpu-unlock en el rig Linux (RTX 3090 = GA102 ✓),
> medir su impacto sobre el workload real de GLM-5.3-Flash, y documentar.
> Ejecutar DESPUÉS del probe OK y del bench base (GLM5_LINUX_BENCH.md es el baseline).

## Por qué (resumen del análisis)

| Modo | 3090 Ti antes | después | Nuestro workload |
|---|---|---|---|
| BF16 | 87.5 TFLOPS | **157.4 (+80%)** | kernels TC del futuro (ter playbook) |
| TF32 | 42.6 | **82.6 (+94%)** | GEMM batchado prefill |
| FP16 | 87.5 | 118.4 (+35%) | — |

Decode NO debería moverse (I/O bound, GPU <15% ocupado). Prefill ~10-15%.
El valor real es compounding con el 2° NVMe (0.9-1.0 tok/s → compute ~30%).

## Compatibilidad (verificada por diseño, confirmar en runtime)

Las tres capas del stack no chocan:
1. gpu-unlock → initramfs, PRE-driver, escribe SS0/SS1 + FLR al hardware
2. DKMS os-mlock.c → parche FUENTE del driver (cudaHostRegisterIoMemory)
3. VFIO/gpu-nvme-direct → userspace (BAR0 sysfs + setpci)

El driver 580.159.03 parcheado queda intacto — gpu-unlock no toca el .ko.

## Paso 0 — baseline SIN unlock (si no lo tenés del bench)

```bash
~/run_glm5.sh -p "The history of computing machines begins with mechanical calculators" -n 30 -c 512
# anotar: decode tok/s, prefill s/tok
```

## Paso 1 — instalar gpu-unlock en el live USB

```bash
# en la sesión Linux persistente:
git clone https://github.com/Zagitta/gpu-unlock
cd gpu-unlock
make install        # detecta distro (Ubuntu → install-deb); pide sudo

# cmdline: agregar al grub del USB (persistent):
#  editar /etc/default/grub → GRUB_CMDLINE_LINUX_DEFAULT="... gpu_unlock.targets=all"
sudo update-grub     # o grub-mkconfig -o /boot/grub/grub.cfg según el USB

# si el USB usa initramfs-tools (Ubuntu): el paquete ya agrega el drop-in y rebuild
```

Notas para live USB:
- La persistencia debe incluir /boot (si el overlay excluye /boot, el cmdline no
  sobrevive al reboot → verificar grub.cfg después de reiniciar)
- `journalctl -k -b | grep gpu_unlock` debe mostrar el unlock de 0000:0a:00.0
- Si nvidia ya bindeó antes del unlock: "Loading the module by hand after boot
  does nothing" — requiere reboot, no es hot

## Paso 2 — verificación del unlock

```bash
journalctl -k -b | grep gpu_unlock
# esperar: "unlocked 0000:0a:00.0" (o similar) sin errores

# micro-bench rápido de FMA rate (opcional):
python3 -c "
import torch
a = torch.randn(8192, 8192, dtype=torch.bfloat16, device='cuda')
b = torch.randn(8192, 8192, dtype=torch.bfloat16, device='cuda')
torch.cuda.synchronize()
import time
t0 = time.perf_counter()
for _ in range(50): c = a @ b
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f'BF16 GEMM: {50 * 2 * 8192**3 / dt / 1e12:.1f} TFLOPS')
# sin unlock: ~40-55 (GA102 BF16 al theoretically 165 con F32 accum al full rate)
# con unlock: ~80-110+
"
```

## Paso 3 — bench del workload real (idéntico al baseline)

```bash
# decode (NO debería cambiar — confirmación del techo I/O):
~/run_glm5.sh --token-ids 785,6722,315,9621,374 -n 30 -t 0 -c 512

# prefill chunked (aquí está la ganancia esperada):
~/run_glm5.sh -p "The history of computing machines begins with mechanical calculators and analytical engines that paved the way for modern digital computers through several decades of innovation" -n 4 -c 2048
# anotar [prefill] s/tok del chunk
```

## Paso 4 — registrar resultados

Actualizar `C:\linux_kit\GLM5_LINUX_BENCH.md` (o este archivo) con:

```
### gpu-unlock (fecha, driver, kernel)
| Métrica | sin unlock | con unlock | delta |
|---|---|---|---|
| BF16 GEMM TFLOPS | | | |
| decode tok/s | | | (esperado: sin cambio) |
| prefill s/tok | | | (esperado: -10-15%) |
| journalctl | — | unlocked OK | |
```

## Riesgos y rollback

- El unlock corre a nivel firmware del GPU (FLR + registros) — si algo saliera
  mal: sacar `gpu_unlock.targets=` del cmdline y rebootear. Sin daño persistente
  (los selectores se resetean con power cycle completo)
- NO instalar si el USB tiene problemas de persistencia de /boot (paso 1)
- Si nvidia falla al bindear tras el unlock: reportar en issues del repo
  (GA102 con driver 580 es territorio testeado, debería andar)

## Si la noche alcanza: el paso extra que aprovecha el unlock

Bench de los kernels TC que hoy no usamos pero el unlock acelera:
1. **dp4a GEMV** (ter playbook Fase C): activación int8 + Q8_0 pesos → ~2× en
   kernels de expertos. El código está en `ter/cuda/ter_cuda_forward_packed.cu`
   (`k_q8` de glm5_gpu2.cu es el punto de partida para portar dp4a)
2. **GEMM batchado BF16 para prefill**: los `glm5_q8_proj_tokens` del chunked
   prefill en BF16 TC en vez de warp-GEMV → el +80% BF16 se materializa directo

Prioridad: solo si el unlock mostró ganancia en el paso 3; si decode no movió
y prefill <5%, el unlock queda instalado "para el futuro" y la noche se dedica
al 2° NVMe (la palanca real).
