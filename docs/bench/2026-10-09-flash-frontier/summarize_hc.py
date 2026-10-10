# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""CPU-only HC evidence audit: native BF16 weights, F32 captures, f64 oracles.

Uses independent math.fsum dots and tie/even BF16 rounding. No model execution.
A/B metrics normalize by A; GPU/reference metrics normalize by the reference.
Archive mode preserves every filename using tar hardlinks for identical bytes.
"""
import argparse
from array import array
from collections import Counter
import hashlib
import json
import lzma
import math
from pathlib import Path
import struct
import sys
import tarfile

from summarize_layer0 import audit, compare_packets, metrics


def sha(data):
    return hashlib.sha256(data).hexdigest()


def decode(raw, code):
    result = array(code)
    result.frombytes(raw)
    if sys.byteorder != 'little':
        result.byteswap()
    return result


def bf16(raw):
    return array('f', (struct.unpack('<f', struct.pack('<I', w << 16))[0]
                       for w in decode(raw, 'H')))


def rne(x):
    word, = struct.unpack('<I', struct.pack('<f', x))
    high, low = word >> 16, word & 65535
    high += low > 32768 or (low == 32768 and high & 1)
    return struct.unpack('<f', struct.pack('<I', high << 16))[0]


def errors(actual, ref):
    d2 = math.fsum((a-b)**2 for a, b in zip(actual, ref))
    norm = math.fsum(b*b for b in ref)
    return dict(rms_error=math.sqrt(d2 / len(ref)),
                max_abs=max(abs(a-b) for a, b in zip(actual, ref)),
                relative_l2=math.sqrt(d2/norm) if norm else None)


def ab(raw_a, raw_b, width, count=8):
    a, b = decode(raw_a, 'f'), decode(raw_b, 'f')
    ua, ub = decode(raw_a, 'I'), decode(raw_b, 'I')
    return dict(aggregate=metrics(a, b, ua, ub), rows=[dict(position=2048+i,
        metrics=metrics(a[i*width:(i+1)*width], b[i*width:(i+1)*width],
                        ua[i*width:(i+1)*width], ub[i*width:(i+1)*width]))
        for i in range(count)])


class Packet:
    def __init__(self, path):
        self.path = path
        self.rows = [json.loads(s) for s in path.read_bytes().splitlines() if s.strip()]
        self.files = {}
        self.problems = []
        self.compared = 0
        self.header, = self.event('header')
        self.mode = self.header['details']['hc_mode']
        self.gdn = audit(path, None)
        self.problems.extend(self.gdn['problems'])
        self.captures = {(r['schedule'], r['name']): self.section(r, 'f')
                         for r in self.event('hc_capture_tensor')}
        self.weights = {r['name']: self.section(r, 'H' if r['dtype']=='BF16' else 'f')
                        for r in self.event('hc_oracle_weight')}
        for event in ('hc_capture_complete', 'hc_oracle_complete', 'hc_weights_complete',
                      'layer0_binary_complete'):
            for record in self.event(event):
                raw = self.file(record)
                self.check(len(raw)==record['bytes'] and sha(raw)==record['sha256'],
                           f'complete binding: {record["path"]}')
        for event in ('hc_capture_tensor', 'hc_oracle_weight', 'hc_f64_oracle'):
            by_path = {}
            for r in self.event(event):
                self.section(r, 'd' if event=='hc_f64_oracle' else
                             'H' if r.get('dtype')=='BF16' else 'f')
                by_path.setdefault(Path(r['path']).name, []).append(r)
            for name, records in by_path.items():
                offset = 0
                for r in sorted(records, key=lambda r:r['byte_offset']):
                    self.check(r['byte_offset']==offset, f'overlap/gap {name}')
                    offset += r['bytes']
                self.check(offset==len(self.files[name]), f'unindexed bytes {name}')

    def check(self, condition, label):
        if not condition:
            self.problems.append(label)

    def event(self, name):
        return [r for r in self.rows if r['event']==name]

    def file(self, r):
        name = Path(r['path']).name
        if name not in self.files:
            self.files[name] = (self.path.parent/name).read_bytes()
        return self.files[name]

    def section(self, r, code):
        raw = self.file(r)[r['byte_offset']:r['byte_offset']+r['bytes']]
        self.check(len(raw)==r['bytes'] and sha(raw)==r['sha256'], f'section hash {r}')
        self.check(len(raw)==math.prod(r['shape'])*array(code).itemsize, f'shape {r}')
        vals = bf16(raw) if code=='H' else decode(raw, code)
        self.check(all(math.isfinite(x) for x in vals) and r.get('all_finite', True),
                   f'nonfinite {r.get("name", r.get("role"))}')
        return raw

    def compare(self, got, recorded, label):
        for k, value in got.items():
            self.compared += 1
            other = recorded[k]
            # Audit summation rounding only; this is not a model-quality gate.
            equal = math.isclose(value, other, rel_tol=1e-9, abs_tol=1e-10) if isinstance(value, float) else value==other
            self.check(equal, f'recorded metric mismatch {label}/{k}: {value} != {other}')

    def witness(self):
        witnesses = self.event('hc_target_witness')
        self.check(Counter(r['label'] for r in witnesses)==Counter(['A/off','A/on','B/off','B/on']), 'HC witness coverage')
        result = []
        for w in witnesses:
            expected = [(2048,3),(2051,2045)] if w['label'].startswith('A') else [(2048,2048)]
            self.check(w['expected_calls']==len(expected)*2==len(w['records']) and w['valid'], 'target count')
            self.check([(r['start'],r['tokens'],r['role']) for r in w['records']]==
                       [(s,n,role) for s,n in expected for role in ('down','up')], 'target roles/ranges')
            for r in w['records']:
                kernel = 'kernel_mat_mat_bf16_bfloat_act_f32' if self.mode=='production' and r['tokens']>3 else 'kernel_mat_mat_bf16_f32'
                self.check(r['kernel']==r['expected_kernel']==kernel, 'target kernel')
                self.check(r['policy']==('Production' if self.mode=='production' else 'F32DownUp'), 'target policy')
            census, = [r for r in self.event('warm_dispatch_witness') if r['label']==w['label']]
            kernels = census['all_kernels']['kernel_counts']
            bfloat = kernels.get('kernel_mat_mat_bf16_bfloat_act_f32',0)
            targets = sum(r['kernel']=='kernel_mat_mat_bf16_bfloat_act_f32' for r in w['records'])
            self.check(bfloat-targets==w['other_bfloat_calls']==192, 'non-target bfloat changed')
            result.append(dict(label=w['label'], calls=w['records'], target_count=len(w['records']),
                               non_target_bfloat=bfloat-targets, all_bfloat=bfloat,
                               all_bf16_f32=kernels.get('kernel_mat_mat_bf16_f32',0),
                               strict_router=census['strict_router_calls']))
        self.check(self.header['details']['bf16_activation_mode']=='production', 'broad override active')
        prefix, = [r for r in self.event('layer0_bf16_dispatch_witness') if r['label']=='prefix2048/production']
        self.check(prefix['bfloat_activation_kernel_count']==194 and prefix['f32_activation_kernel_count']==12, 'prefix changed')
        return result


def references(weight, inputs, k, m, cache, weight_hash):
    outputs = [[], []]
    for row in range(8):
        raw = inputs[row*k*4:(row+1)*k*4]
        key = (weight_hash, sha(raw))
        if key not in cache:
            x = decode(raw, 'f')
            rounded = [rne(v) for v in x]
            orig, rned = [], []
            for j in range(m):
                w = weight[j*k:(j+1)*k]
                orig.append(math.fsum(a*b for a,b in zip(w,x)))
                rned.append(math.fsum(a*b for a,b in zip(w,rounded)))
            cache[key] = (orig, rned)
        for output, values in zip(outputs, cache[key]):
            output.extend(values)
    return outputs


def analyze(packet, weights, cache):
    print(f'Auditing {packet.mode} projection references', file=sys.stderr, flush=True)
    stages = []
    for record in packet.event('hc_cross_schedule'):
        name = record['name']
        width = 320 if name in ('hc.down','hc.low') else 10240
        computed = ab(packet.captures['A',name],packet.captures['B',name],width)
        packet.compare(computed['aggregate'], record['aggregate'], name)
        for got, saved in zip(computed['rows'], record['rows']):
            packet.compare(got['metrics'], saved['metrics'], f'{name}/{got["position"]}')
        stages.append(dict(name=name,**computed))
    oracle_results = []
    for arm in ('A','B'):
        for role, k, m, inp, out in [('down',10240,320,'hc.normalized','hc.down'),
                                    ('up',320,10240,'hc.low','hc.up')]:
            name = f'blk.0.hc_attn_{role}.weight'
            refs = references(weights[name], packet.captures[arm,inp], k,m,cache,sha(packet.weights[name]))
            gpu = decode(packet.captures[arm,out], 'f')
            for activation, ref in zip(('original_f32_activation','rne_bf16_activation'), refs):
                saved, = [r for r in packet.event('hc_f64_oracle') if r['schedule']==arm and r['role']==role and r['activation']==activation]
                retained = decode(packet.section(saved,'d'),'d')
                recompute_error = errors(ref,retained)
                packet.check(all(math.isclose(a,b,rel_tol=1e-11,abs_tol=1e-9) for a,b in zip(ref,retained)), 'f64 dot recomputation disagreement')
                aggregate = errors(gpu,ref)
                packet.compare(aggregate,saved['gpu_error'],f'oracle/{arm}/{role}/{activation}')
                rows = []
                for i in range(8):
                    err = errors(gpu[i*m:(i+1)*m],ref[i*m:(i+1)*m])
                    packet.compare(err,saved['rows'][i]['gpu_error'],f'oracle/{arm}/{role}/{activation}/{i}')
                    rows.append(dict(position=2048+i,gpu_error=err))
                oracle_results.append(dict(schedule=arm,role=role,activation=activation,
                    saved_f64_recomputation_error=recompute_error,gpu_error=aggregate,rows=rows))
            saved_delta, = [r for r in packet.event('hc_rounding_reference_delta') if r['schedule']==arm and r['role']==role]
            packet.compare(errors(refs[1],refs[0]),saved_delta['metrics'],f'rounding delta/{arm}/{role}')
    endpoints = {r['label']:r for r in packet.event('endpoint')}
    a,b = [endpoints[f'{arm}/on']['state'] for arm in ('A','B')]
    state = dict(causal_metadata_equal=all(a[k]==b[k] for k in ('position','qsa_lengths','ple_prior_tokens')),
                 changed_persistent_indices=[x['index'] for x,y in zip(a['tensors'],b['tensors']) if x['sha256']!=y['sha256']],
                 total_persistent=len(a['tensors']))
    return dict(packet=packet.path.name,packet_sha256=sha(packet.path.read_bytes()),mode=packet.mode,
                source_binding=packet.header,executable_binding=packet.event('executable_binding'),
                capture_stages=stages,projection_oracles=oracle_results,
                dispatch=packet.witness(), endpoint_recorded_not_recomputed=packet.event('layer0_endpoint_cross_schedule'),
                state_hash_comparison=state,gdn=packet.gdn,
                recorded_hc_metric_fields_checked=packet.compared,problems=packet.problems)


def archive_packets(packets, destination):
    members = {}
    for packet in packets:
        members.update({name:dict(path=packet.path.parent/name, bytes=len(raw),sha256=sha(raw))
                        for name,raw in packet.files.items()})
    manifest = dict(format='deterministic USTAR + XZ/LZMA2 preset6, dictionary32MiB; identical bytes use tar hardlinks',
                    packets=[dict(path=p.path.name,sha256=sha(p.path.read_bytes())) for p in packets],
                    originals_deleted=False, original_members=[])
    known = {}
    with destination.open('xb') as output, lzma.LZMAFile(output, 'w', filters=[dict(id=lzma.FILTER_LZMA2,preset=6,dict_size=32*1024*1024)]) as compressed, tarfile.open(fileobj=compressed, mode='w|',format=tarfile.USTAR_FORMAT) as tar:
        for name,record in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.mode, info.mtime, info.uid, info.gid = 0o644,0,0,0
            digest = record['sha256']
            item = dict(path=name,bytes=record['bytes'],sha256=digest)
            if digest in known:
                info.type, info.linkname = tarfile.LNKTYPE,known[digest]
                item['hardlink_to'] = info.linkname
                tar.addfile(info)
            else:
                info.size = record['bytes']
                with record['path'].open('rb') as source:
                    tar.addfile(info,source)
                known[digest] = name
            manifest['original_members'].append(item)
    with tarfile.open(destination,'r:xz') as tar:
        assert tar.getnames()==list(sorted(members)), 'archive member list mismatch'
        for item in manifest['original_members']:
            data = tar.extractfile(item['path']).read()
            assert len(data)==item['bytes'] and sha(data)==item['sha256'], 'archive roundtrip failed'
            assert sha(members[item['path']]['path'].read_bytes())==item['sha256'], 'raw changed'
    manifest.update(archive=destination.name,bytes=destination.stat().st_size,
                    sha256=sha(destination.read_bytes()),original_bytes=sum(r['bytes'] for r in members.values()),
                    unique_payload_bytes=sum(r['bytes'] for r in manifest['original_members'] if 'hardlink_to' not in r),
                    all_original_members_roundtrip_verified=True)
    destination.with_name(destination.name.removesuffix('.tar.xz')+'.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('production',type=Path)
    parser.add_argument('f32downup',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--archive',type=Path,help='Create new lossless combined archive; refuses overwrite')
    args = parser.parse_args()
    # Exactly representable midpoint tests, including negative and odd ties.
    assert rne(1.00390625)==1.0 and rne(1.01171875)==1.015625 and rne(-1.00390625)==-1.0
    packets = [Packet(args.production),Packet(args.f32downup)]
    assert [p.mode for p in packets]==['production','f32downup']
    assert packets[0].weights==packets[1].weights, 'different weights'
    assert [r['used_token_ids'] for r in packets[0].event('prompt')]==[r['used_token_ids'] for r in packets[1].event('prompt')], 'different prompts'
    weights = {name:bf16(raw) for name,raw in packets[0].weights.items() if not name.endswith('norm.weight')}
    cache = {}
    results = [analyze(p,weights,cache) for p in packets]
    cross = [dict(schedule=arm,name=name,**ab(packets[0].captures[arm,name],packets[1].captures[arm,name],320 if name in ('hc.down','hc.low') else 10240)) for arm,name in packets[0].captures]
    result = dict(contract='CPU independent oracle audit; no speed or model quality claim; up refs use actual per-arm inputs',
                  packets=results,cross_mode_hc=cross,
                  cross_mode_gdn=compare_packets(args.production,args.f32downup),
                  unique_projection_input_rows_recomputed=len(cache))
    if args.archive:
        result['compression'] = archive_packets(packets,args.archive)
    output = json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output,end='')
    problems = [p for r in results for p in r['problems']]+result['cross_mode_gdn']['problems']
    print(f'HC audit complete: {len(cache)} unique projection input rows; {len(problems)} problems',file=sys.stderr)
    for problem in problems:
        print(problem,file=sys.stderr)
    return bool(problems)


if __name__=='__main__':
    sys.exit(main())
