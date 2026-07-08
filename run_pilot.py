import sys, os, time, importlib.util

spec = importlib.util.spec_from_file_location('qv2', './qdiff_combined_v2.py')
qv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qv2)

os.makedirs('./pilot_results', exist_ok=True)

BLOCKS_TO_CHECK = [0, 9, 11]   # n=50/c10, n=500/c10 (was broken), n=500/c40 (estimability check)

OLD_PROEL_P25 = {
    (50, 0.10): 0.9575, (500, 0.10): 0.6860, (500, 0.40): 0.8990,
}

results = {}
for block_id in BLOCKS_TO_CHECK:
    t0 = time.time()
    out = qv2.run_block(block_id, n_reps=20, seed=90000 + block_id, out_dir='./pilot_results')
    qv2.save_block(out, './pilot_results')
    results[block_id] = out
    print(f'Block {block_id} done in {time.time()-t0:.1f}s', flush=True)

print()
print('=' * 88)
print(f"{'block':>5} {'n':>5} {'cens':>5} {'p':>6} {'PROEL':>8} {'PLEL':>8} {'NA':>8} {'est.rt':>7}  {'old PROEL':>10}")
for block_id, out in results.items():
    n, c = out['n'], out['cens_rate']
    for p in [0.25, 0.50, 0.75]:
        cov = out['coverage'].get(f'a95_p{int(p*100)}')
        if cov is None: continue
        old = OLD_PROEL_P25.get((n, c)) if p == 0.25 else None
        old_str = f'{old:.4f}' if old is not None else '--'
        print(f"{block_id:>5} {n:>5} {c:>5.0%} {p:>6.2f} {cov['proel_mean']:>8.3f} "
              f"{cov['plel_mean']:>8.3f} {cov['na_mean']:>8.3f} {cov['estimability_rate']:>7.2f}  {old_str:>10}")
print('=' * 88)
