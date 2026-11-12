#!/usr/bin/env python3
import os, json
ROOT = '/workspace/app'
OUT = os.path.join(ROOT, 'research', '03_experiments', 'hsam_grid')
try:
    with open(os.path.join(OUT, 'results.json')) as f:
        res = json.load(f)
    sup = res['supervised']
    cont = res['continual']
    print('===SUPERVISED (winners=3)===')
    for slots in [100,200,400,800]:
        for mode in ['full','no_tune','no_alloc']:
            k = f's{slots}_w3_{mode}'
            if k in sup:
                print(f'{k}: acc={sup[k]["acc"]["mean"]:.4f}±{sup[k]["acc"]["std"]:.4f} wall={sup[k]["wall_s"]["mean"]:.3f}s rss={sup[k]["peak_rss_mb"]["mean"]:.1f}MB')
    print('\n===CONTINUAL (winners=3)===')
    for slots in [100,200,400,800]:
        for mode in ['full','no_tune','no_alloc']:
            k = f's{slots}_w3_{mode}'
            if k in cont:
                print(f'{k}: AA={cont[k]["AA"]["mean"]:.4f}±{cont[k]["AA"]["std"]:.4f} FM={cont[k]["FM"]["mean"]:.4f}±{cont[k]["FM"]["std"]:.4f} wall={cont[k]["wall_s"]["mean"]:.3f}s')
    print('\n===GAP analysis w3 full vs no_tune vs no_alloc===')
    for slots in [100,200,400,800]:
        kf = f's{slots}_w3_full'; kt = f's{slots}_w3_no_tune'; ka = f's{slots}_w3_no_alloc'
        if all(x in sup for x in [kf,kt,ka]):
            print(f'S{slots} SUP: full={sup[kf]["acc"]["mean"]:.4f} no_tune={sup[kt]["acc"]["mean"]:.4f} (Δ={sup[kt]["acc"]["mean"]-sup[kf]["acc"]["mean"]:.4f}) no_alloc={sup[ka]["acc"]["mean"]:.4f} (Δ={sup[ka]["acc"]["mean"]-sup[kf]["acc"]["mean"]:.4f})')
        if all(x in cont for x in [kf,kt,ka]):
            print(f'S{slots} CONT: full AA={cont[kf]["AA"]["mean"]:.4f} no_tune={cont[kt]["AA"]["mean"]:.4f} (Δ={cont[kt]["AA"]["mean"]-cont[kf]["AA"]["mean"]:.4f}) no_alloc={cont[ka]["AA"]["mean"]:.4f} (Δ={cont[ka]["AA"]["mean"]-cont[kf]["AA"]["mean"]:.4f})')
except Exception as e:
    print(e)
    import traceback
    traceback.print_exc()
