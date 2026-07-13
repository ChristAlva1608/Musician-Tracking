#!/usr/bin/env python3
"""
Remove teleport-blip identity assignments from people_id_*.json.

Signature of a false assignment (reflection in glass, person displayed ON
the iPad/TV screen, mirror): the identity JUMPS >800px to a spot, sits
there for a few entries, then JUMPS BACK to where it was — while the
positions before and after the excursion agree. A real person walking
produces continuous intermediate steps at 1-2s sampling and does not
return-teleport; near-lens walking (huge bbox) is exempted explicitly.

The blip entries lose their id (the detection stays in the timeline with
id=None-equivalent: it is dropped from 'people', since only identified
people are stored). Backup: .pre_blipclean.json

Usage:
    python3 scripts/clean_id_blips.py src/output/person_id_christina \
        [--jump 800] [--agree 300] [--max-span 45] [--w 4000]
"""

import argparse
import json
from pathlib import Path

W = 4000


def wrap_dx(a, b, w=W):
    d = abs(a - b)
    return min(d, w - d)


def cx_of(p):
    return (p['bbox'][0] + p['bbox'][2]) / 2 % W


def h_of(p):
    return p['bbox'][3] - p['bbox'][1]


def clean(doc, jump=800.0, agree=300.0, max_span=45.0, near_lens_h=1000.0):
    # per-id sequence of (entry_idx, person_idx, t, cx, h)
    tracks = {}
    for ei, e in enumerate(doc['timeline']):
        for pi, p in enumerate(e['people']):
            tracks.setdefault(p['id'], []).append(
                (ei, pi, e['t'], cx_of(p), h_of(p)))

    to_drop = set()  # (entry_idx, person_idx)
    for pid, seq in tracks.items():
        n = len(seq)
        i = 1
        while i < n - 1:
            prev = seq[i - 1]
            # candidate excursion start: big jump OR impossible speed
            d_in = wrap_dx(seq[i][3], prev[3])
            sp_in = d_in / max(1e-6, seq[i][2] - prev[2])
            if d_in <= jump and sp_in <= 400:
                i += 1
                continue
            # extend excursion while entries stay clustered together
            j = i
            while (j + 1 < n
                   and wrap_dx(seq[j + 1][3], seq[i][3]) < agree
                   and seq[j + 1][2] - prev[2] <= max_span):
                j += 1
            if j + 1 >= n:
                break
            nxt = seq[j + 1]
            span_ok = (nxt[2] - prev[2]) <= max_span
            returns = wrap_dx(nxt[3], prev[3]) < agree
            jumps_back = wrap_dx(nxt[3], seq[j][3]) > jump
            near_lens = (max(s[4] for s in seq[i:j + 1]) > near_lens_h
                         or prev[4] > near_lens_h or nxt[4] > near_lens_h)
            # entry/exit implied speeds: no one covers >400 px/s at
            # room distance — a spot unreachable from BOTH neighbors is
            # bogus even when the neighbors don't agree with each other
            v_in = wrap_dx(seq[i][3], prev[3]) / max(1e-6, seq[i][2] - prev[2])
            v_out = wrap_dx(nxt[3], seq[j][3]) / max(1e-6, nxt[2] - seq[j][2])
            impossible = v_in > 400 and v_out > 400
            if span_ok and not near_lens and ((returns and jumps_back)
                                              or impossible):
                for s in seq[i:j + 1]:
                    to_drop.add((s[0], s[1]))
                i = j + 1
            else:
                i += 1

    dropped = []
    for ei, e in enumerate(doc['timeline']):
        keep = []
        for pi, p in enumerate(e['people']):
            if (ei, pi) in to_drop:
                dropped.append((e['t'], p['id']))
            else:
                keep.append(p)
        e['people'] = keep
    return dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--jump', type=float, default=800)
    ap.add_argument('--agree', type=float, default=300)
    ap.add_argument('--max-span', type=float, default=45)
    args = ap.parse_args()

    for jpath in sorted(Path(args.data_dir).glob('people_id_360-Camera-*.json')):
        if jpath.name.endswith(('.pre_relabel.json', '.pre_blipclean.json',
                                '.pre_camswap.json')):
            continue
        doc = json.loads(jpath.read_text())
        jpath.with_suffix('.pre_blipclean.json').write_text(
            json.dumps(doc, indent=1))
        dropped = clean(doc, args.jump, args.agree, args.max_span)
        doc['stats']['blip_dropped'] = len(dropped)
        doc['stats']['blip_dropped_ts'] = sorted({int(t) for t, _ in dropped})
        jpath.write_text(json.dumps(doc, indent=1))
        print(f'{jpath.name}: dropped {len(dropped)} blip assignments '
              f'at t={sorted({int(t) for t, _ in dropped})[:20]}')


if __name__ == '__main__':
    main()
