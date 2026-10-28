#!/usr/bin/env python3
# fill_second_disk.py v2 — llena el disco 1 (crudo) con los tensores de las
# capas impares. Cada tensor ocupa su slot contiguo de 288*slice_bytes; cada
# slice se escribe ALINEADO (in_off=0 en el disco 1, single-run garantizado).
# Produce disk_layout.json con los runs por tensor (LBAs del disco 1) que
# gen_lba2.py usa para el índice LBA2 multi-disco.
#
# Uso: sudo python3 fill_second_disk.py <src_dev> <dst_dev> <index.json> <out_layout.json>
# ¡CUIDADO! dst_dev se sobreescribe en crudo.
import json, os, sys

SECTOR = 512
CHUNK = 4 * 1024 * 1024

def main():
    src_dev, dst_dev, idx_path, out_layout = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    DST_SERIAL_EXPECTED = os.environ.get('DST_SERIAL', '')
    idx = json.load(open(idx_path))
    layers = sorted(idx['layers'], key=int)
    slice_bytes = idx['layers'][layers[0]]['gate']['slice_bytes']
    slot_bytes = 288 * slice_bytes

    # VERIFICACIÓN DE IDENTIDAD DEL DESTINO: por serial, nunca por nombre de device.
    # (Lección: la re-enumeración de hot-swap renombra nvmeXnY y un fill al disco
    #  equivocado destruye datos — esto pasó con el Micron/C: el 22-10-2026.)
    dst_serial = ''
    for b in os.listdir('/sys/block'):
        if not b.startswith('nvme'):
            continue
        try:
            if dst_dev.replace('/dev/', '') == b:
                dst_serial = open(f'/sys/block/{b}/device/serial').read().strip()
        except OSError:
            pass
    if DST_SERIAL_EXPECTED and dst_serial != DST_SERIAL_EXPECTED:
        print(f'RECHAZADO: {dst_dev} serial={dst_serial!r} != esperado {DST_SERIAL_EXPECTED!r}')
        print('Exporta DST_SERIAL con el serial del disco destino (cat /sys/block/<dev>/device/serial).')
        sys.exit(2)
    print(f'destino verificado: {dst_dev} serial={dst_serial}')

    dst_fd = os.open(dst_dev, os.O_WRONLY)
    dst_size = os.lseek(dst_fd, 0, os.SEEK_END)
    src_fd = os.open(src_dev, os.O_RDONLY)

    layout = {'n_disks': 2, 'slice_bytes': slice_bytes, 'tensors': {}}
    slot_idx = 0
    tensors_done = 0
    total = len(layers) * 3
    t_start = os.times()

    for layer in layers:
        il = int(layer)
        if il % 2 == 0:
            for kind in ('gate', 'up', 'down'):
                layout['tensors'][f'{il}_{kind}'] = {'disk': 0}
                tensors_done += 1
            continue
        base = slot_idx * slot_bytes
        assert base + slot_bytes <= dst_size, f'no cabe: slot {slot_idx}'
        slot_idx += 1
        lv = idx['layers'][layer]
        for kind in ('gate', 'up', 'down'):
            t = lv[kind]
            runs_out = []
            for e in range(288):
                exp = t['experts'][e]
                in_off = exp['in_off']
                runs = exp['runs']
                dst_off = base + e * slice_bytes
                os.lseek(dst_fd, dst_off, os.SEEK_SET)
                remaining = slice_bytes
                first = True
                for r in runs:
                    lba, nb = r[0], r[1]
                    skip = in_off if first else 0
                    pos = lba * SECTOR + skip
                    take = min(nb - skip, remaining)
                    left = take
                    while left > 0:
                        chunk = min(left, CHUNK)
                        os.lseek(src_fd, pos, os.SEEK_SET)
                        data = os.read(src_fd, chunk)
                        assert len(data) == chunk, f'short read @ {pos}'
                        w = os.write(dst_fd, data)
                        assert w == chunk, 'short write'
                        pos += chunk
                        left -= chunk
                    remaining -= take
                    first = False
                    if remaining <= 0:
                        break
                assert remaining <= 0
                runs_out.append([dst_off // SECTOR, slice_bytes])
            layout['tensors'][f'{il}_{kind}'] = {'disk': 1, 'in_off': 0, 'runs': runs_out}
            tensors_done += 1
            if tensors_done % 21 == 0:
                print(f'{tensors_done}/{total} tensores ({layer}_{kind} slot {slot_idx-1})', flush=True)

    os.close(src_fd); os.close(dst_fd)
    json.dump(layout, open(out_layout, 'w'))
    print(f'listo: {tensors_done} tensores, {slot_idx} slots al disco 1; layout -> {out_layout}')

if __name__ == '__main__':
    main()
