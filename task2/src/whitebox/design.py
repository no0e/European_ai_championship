"""Choix hors ligne des prompts pour les dernières requêtes : on échantillonne le modèle de base local
et on mesure l'information attendue par réponse (entropie cumulée, et KL base vs fp8 simulé). Aucune requête serveur.
  python design.py --repo Qwen/Qwen2.5-1.5B-Instruct --sys tmpl --T 0.7 --n 6"""
import argparse, copy, json, os, sys
from pathlib import Path
import torch
import score as S, quant
PROMPTS = {
 'words': "Write 150 random, unrelated English words separated by commas. Do not follow any pattern or theme.",
 'digits': "Generate a random sequence of 200 digits (0-9) separated by spaces. Make it as unpredictable as possible.",
 'stream': "Write a long surreal stream-of-consciousness paragraph (at least 250 words) that jumps unpredictably between unrelated topics.",
 'story': "Tell me a long, original story with unusual characters and unexpected events. Write at least 300 words.",
 'names': "List 100 random first names from many different countries, separated by commas, with no repetition.",
 'nounnum': "Write a random list of 100 unrelated nouns, each followed by a random number between 1 and 999, like this: lamp 482, river 77, ...",
 'letters': "Write 200 random lowercase letters separated by spaces. Make the sequence as unpredictable as possible.",
 'madeup': "Invent 80 completely made-up words that do not exist in any language, separated by commas.",
 'poem': "Write a free-verse poem of at least 40 lines. Each line must be about a completely different subject from the previous one.",
 'facts': "Write 30 short, unrelated sentences. Each sentence must be about a different random topic (science, cooking, sports, history, animals, etc.).",
 'essay': "Write a detailed 300-word essay explaining how a bicycle works.",
}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', required=True); ap.add_argument('--sys', default='tmpl')
    ap.add_argument('--T', type=float, default=0.7); ap.add_argument('--n', type=int, default=6); a = ap.parse_args()
    from transformers import AutoTokenizer, AutoModelForCausalLM
    local = S.W / 'models' / a.repo.replace('/', '--'); tok = AutoTokenizer.from_pretrained(str(local))
    m = AutoModelForCausalLM.from_pretrained(str(local), torch_dtype=torch.float32).eval().to(S.DEV)
    q = copy.deepcopy(m); print(quant.apply(q, 'fp8t'))
    eos = tok.convert_tokens_to_ids('<|eot_id|>' if 'llama' in a.repo.lower() else '<|im_end|>')
    torch.manual_seed(0)
    print(f"{'prompt':8s}  len  stop%  H/tok  H_tot  top1%  KL_fp8/tok  KL_fp8_tot(nats/réponse)")
    for name, p in PROMPTS.items():
        ids = tok(S.build_prompt(tok, a.repo, p, a.sys), add_special_tokens=False, return_tensors='pt')['input_ids'].to(S.DEV)
        L = []; Ht = []; KL = []; st = []; t1 = []
        for _ in range(a.n):
            x = ids.clone(); past = None; H = 0.0; n = 0; stopped = False; top1 = 0
            with torch.no_grad():
                out = m(input_ids=x, use_cache=True); past = out.past_key_values; lg = out.logits[0, -1]
                gen = []
                for t in range(256):
                    ls = torch.log_softmax(lg.float() / a.T, 0); pr = ls.exp(); H += float(-(pr * ls).sum())
                    y = int(torch.multinomial(pr, 1)); top1 += int(y == int(pr.argmax())); n += 1
                    if y == eos: stopped = True; break
                    gen.append(y)
                    out = m(input_ids=torch.tensor([[y]], device=S.DEV), past_key_values=past, use_cache=True); past = out.past_key_values; lg = out.logits[0, -1]
                full = torch.cat([ids, torch.tensor([gen], device=S.DEV, dtype=ids.dtype)], 1) if gen else ids
                lb = torch.log_softmax(m(input_ids=full).logits[0, ids.shape[1] - 1:].float() / a.T, 1)
                lq = torch.log_softmax(q(input_ids=full).logits[0, ids.shape[1] - 1:].float() / a.T, 1)
                KL.append(float((lb.exp() * (lb - lq)).sum()))
            L.append(n); Ht.append(H); st.append(stopped); t1.append(top1 / n)
        import numpy as np
        print(f"{name:8s} {np.mean(L):5.0f} {100*np.mean(st):5.0f} {np.sum(Ht)/np.sum(L):6.3f} {np.mean(Ht):6.1f} {100*np.mean(t1):6.1f} {np.sum(KL)/np.sum(L):10.5f} {np.mean(KL):8.3f}", flush=True)
if __name__ == '__main__': main()
