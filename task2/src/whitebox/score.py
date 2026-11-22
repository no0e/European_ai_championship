"""Boîte blanche : score chaque réponse déjà collectée sous un checkpoint local (teacher forcing).
Aucune requête au serveur de l'épreuve. Sortie : un pickle {meta, rows}, une ligne par réponse.
  python score.py --repo Qwen/Qwen2.5-0.5B-Instruct --kind ep --families QWEN_A QWEN_B --sys tmpl --tag base
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np, torch

W = Path(os.environ.get('T2W', os.path.expanduser('~/t2wb')))
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TG = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5]
TOPK = 40
WTS = [0.7, 0.8, 0.9, 1.0]
RPS = [1.0, 1.05, 1.1]
DTS = [0.6, 0.7, 0.8, 1.0]

def load_rows(kind):
    rows = []
    if kind == 'ep':
        fam = json.load(open(W / 'data/ep_families2.json'))
        for f in ('answers_ep.jsonl', 'answers_long_ep.jsonl', 'answers_hi_ep.jsonl'):
            if not (W / 'data' / f).exists(): continue
            for i, l in enumerate(open(W / 'data' / f)):
                r = json.loads(l); r['src'] = f'{f}:{i}'; r['fam'] = fam[r['endpoint']]
                r['max_tokens'] = int(r['max_tokens']); rows.append(r)
    else:
        for f in ('reference_answers.jsonl', 'snapshot_answers_ref2.jsonl', 'answers_long_ref.jsonl', 'ref_extra.jsonl'):
            p = W / 'data' / f
            if not p.exists(): continue
            for i, l in enumerate(open(p)):
                r = json.loads(l); r['src'] = f'{f}:{i}'; r['fam'] = 'REF'
                r['probe'] = r.get('probe', 'off' + str(r.get('id')))
                r['max_tokens'] = int(r.get('max_tokens', 256)); rows.append(r)
    return rows

def build_prompt(tok, repo, prompt, sysmode):
    name = repo.lower()
    if sysmode == 'tmpl':
        return tok.apply_chat_template([{'role': 'user', 'content': prompt}], add_generation_prompt=True, tokenize=False)
    if 'llama' in name:
        u = f'<|start_header_id|>user<|end_header_id|>\n\n{prompt.strip()}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        if sysmode == 'none': return '<|begin_of_text|>' + u
        if sysmode == 'date': s = 'Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n'
        else: s = sysmode
        return f'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>' + u
    u = f'<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n'
    if sysmode == 'none': return u
    return f'<|im_start|>system\n{sysmode}<|im_end|>\n' + u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True); ap.add_argument('--kind', default='ep')
    ap.add_argument('--families', nargs='*'); ap.add_argument('--endpoints', nargs='*')
    ap.add_argument('--sys', default='tmpl'); ap.add_argument('--tag', default='base')
    ap.add_argument('--quant', default='none'); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--per-endpoint', type=int, default=0); ap.add_argument('--threads', type=int, default=32)
    ap.add_argument('--lora', default=''); ap.add_argument('--lora-scale', type=float, default=1.0)
    ap.add_argument('--dec', action='store_true', help='stats de décodage (pénalité de répétition, masse au-dessus, rang)')
    ap.add_argument('--watch', type=int, default=0, help='N tokens les plus fréquents : masse attendue par réponse')
    ap.add_argument('--probes', nargs='*')
    ap.add_argument('--lite', action='store_true', help='ne garde que lp/H par T (pas de top-k)')
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    local = W / 'models' / a.repo.replace('/', '--')
    src = str(local) if local.exists() else a.repo
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32); model.eval()
    info = {}
    if a.quant != 'none':
        import quant; info = quant.apply(model, a.quant); print('quant', info, flush=True)
    if a.lora:
        import lora_util; info['lora'] = lora_util.apply(model, a.lora, a.lora_scale)
    model.to(DEV)
    name = a.repo.lower()
    eos = tok.convert_tokens_to_ids('<|eot_id|>' if 'llama' in name else '<|im_end|>')
    rows = load_rows(a.kind)
    if a.families: rows = [r for r in rows if r['fam'] in a.families]
    if a.endpoints: rows = [r for r in rows if r['endpoint'] in a.endpoints]
    if a.probes: rows = [r for r in rows if r['probe'] in a.probes]
    if a.per_endpoint:
        rng = np.random.default_rng(0); by = {}
        for r in rows: by.setdefault(r['endpoint'], []).append(r)
        rows = [r for e in sorted(by) for r in (by[e] if len(by[e]) <= a.per_endpoint else [by[e][i] for i in sorted(rng.choice(len(by[e]), a.per_endpoint, replace=False))])]
    if a.limit: rows = rows[:a.limit]
    Tg = torch.tensor(TG)
    watch = None
    if a.watch:
        import collections
        cnt = collections.Counter()
        for r in rows: cnt.update(tok(r['response'], add_special_tokens=False)['input_ids'])
        wl = [eos] + [t for t, _ in cnt.most_common(a.watch) if t != eos]
        watch = torch.tensor(wl, device=DEV)
    out = []; t0 = time.time(); ntok = 0
    lm_head = model.get_output_embeddings(); body = model.model
    for k, r in enumerate(rows):
        p_ids = tok(build_prompt(tok, a.repo, r['prompt'], a.sys), add_special_tokens=False)['input_ids']
        resp = r['response']
        y = tok(resp, add_special_tokens=False)['input_ids']
        fin = r.get('finish')
        stopped = (fin == 'stop') if fin not in (None, 'None') else (len(y) < r['max_tokens'] - 1)
        y2 = y + [eos] if stopped else y
        if not y2: continue
        ids = torch.tensor([p_ids + y2], device=DEV)
        with torch.no_grad():
            h = body(input_ids=ids).last_hidden_state[0, len(p_ids) - 1: -1]
            lg = lm_head(h).float()                                   # (n, V)
        yt = torch.tensor(y2, device=DEV)
        n = len(y2)
        lt = lg.gather(1, yt[:, None])[:, 0]
        rank = (lg > lt[:, None]).sum(1)
        row = {k2: r.get(k2) for k2 in ('endpoint', 'probe', 'repeat', 'src', 'fam', 'max_tokens')}
        row.update(ids=np.array(y2, dtype=np.int32), stopped=stopped, n_prompt=len(p_ids), rank=rank.cpu().numpy().astype(np.int32),
                   logit=lt.cpu().numpy().astype(np.float32), retok_ok=(tok.decode(y) == resp))
        lpT = np.zeros((len(TG), n), np.float32); HT = np.zeros_like(lpT); VT = np.zeros_like(lpT); eT = np.zeros_like(lpT)
        for j, T in enumerate(TG):
            ls = torch.log_softmax(lg / T, 1); p = ls.exp()
            H = -(p * ls).sum(1); V = (p * ls * ls).sum(1) - H * H
            lpT[j] = ls.gather(1, yt[:, None])[:, 0].cpu().numpy(); HT[j] = H.cpu().numpy(); VT[j] = V.cpu().numpy(); eT[j] = ls[:, eos].cpu().numpy()
        row.update(lp=lpT, H=HT, V=VT, lpeos=eT)
        if watch is not None:
            row['wE'] = np.stack([torch.softmax(lg / T, 1)[:, watch].sum(0).cpu().numpy() for T in WTS]).astype(np.float32)
        if a.dec:
            V = lg.shape[1]
            seen = torch.zeros((n, V), dtype=torch.bool, device=DEV)
            seen[:, torch.tensor(sorted(set(p_ids)), device=DEV)] = True
            for t in range(1, n): seen[t:, y2[t - 1]] = True
            dec = {}
            for rp in RPS:
                l2 = torch.where(seen, torch.where(lg > 0, lg / rp, lg * rp), lg) if rp != 1.0 else lg
                l2t = l2.gather(1, yt[:, None])
                rk2 = (l2 > l2t).sum(1)
                top20 = l2.topk(20, 1).values[:, -1:]
                for T in DTS:
                    ls = torch.log_softmax(l2 / T, 1); p = ls.exp()
                    pt = ls.gather(1, yt[:, None])[:, 0]
                    above = torch.where(l2 > l2t, p, torch.zeros_like(p)).sum(1)
                    m20 = torch.where(l2 >= top20, p, torch.zeros_like(p)).sum(1)
                    dec[(rp, T)] = dict(rank=rk2.cpu().numpy().astype(np.int32), lp=pt.cpu().numpy(), above=above.cpu().numpy(), m20=m20.cpu().numpy())
            row['dec'] = dec
        if not a.lite:
            tv, ti = lg.topk(TOPK, 1)
            row.update(top_i=ti.cpu().numpy().astype(np.int32), top_v=tv.cpu().numpy().astype(np.float16))
        out.append(row); ntok += n
        if k % 500 == 0: print(f'{k}/{len(rows)} {ntok} tok {time.time() - t0:.0f}s', flush=True)
    f = W / 'out' / f"{a.repo.split('/')[-1]}__{a.kind}__{a.tag}.pkl"
    pickle.dump({'watch': (watch.cpu().numpy() if watch is not None else None), 'WTS': WTS, 'repo': a.repo, 'sys': a.sys, 'quant': a.quant, 'info': info, 'TG': TG, 'rows': out}, open(f, 'wb'))
    print('écrit', f, len(out), 'réponses', ntok, 'tokens', f'{time.time() - t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()
