"""Simulations de quantification fp8 (e4m3) des poids, plusieurs variantes plausibles.
  fp8t   : couches linéaires des blocs, une échelle par tenseur (amax/448), style vLLM
  fp8c   : couches linéaires des blocs, une échelle par canal de sortie (optimum-quanto qfloat8, axis=0)
  fp8ch  : comme fp8c + lm_head
  fp8th  : comme fp8t + lm_head
  fp8raw : cast direct en float8_e4m3fn sans échelle (toutes les matrices 2D)
  fp8all : cast par tenseur de tous les paramètres 2D (embeddings compris)
  int8c  : entiers 8 bits symétriques par canal (quanto qint8)
"""
import torch
MAX = 448.0
def f8(w, scale):
    q = (w.float() / scale).clamp(-MAX, MAX).to(torch.float8_e4m3fn)
    return (q.float() * scale).to(w.dtype)
def apply(model, mode):
    num = den = 0.0; n = 0
    head = model.get_output_embeddings().weight
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear): continue
        is_head = mod.weight is head
        if is_head and mode not in ('fp8ch', 'fp8th', 'fp8raw', 'fp8all', 'int8ch'): continue
        w = mod.weight.data; wc = w.cpu().float()
        if mode in ('fp8t', 'fp8th', 'fp8all'): wq = f8(wc, wc.abs().max() / MAX)
        elif mode in ('fp8c', 'fp8ch'): wq = f8(wc, wc.abs().amax(1, keepdim=True).clamp(min=1e-12) / MAX)
        elif mode == 'fp8raw': wq = wc.to(torch.float8_e4m3fn).float()
        elif mode in ('int8c', 'int8ch'):
            s = wc.abs().amax(1, keepdim=True).clamp(min=1e-12) / 127.0; wq = (wc / s).round().clamp(-127, 127) * s
        else: raise ValueError(mode)
        num += float((wq - wc).pow(2).sum()); den += float(wc.pow(2).sum()); n += 1
        mod.weight.data.copy_(wq.to(w.dtype).to(w.device))
    if mode == 'fp8all':
        e = model.get_input_embeddings().weight
        if e is not head:
            wc = e.data.cpu().float(); e.data.copy_(f8(wc, wc.abs().max() / MAX).to(e.device))
    return {'mode': mode, 'n': n, 'rel_rms': (num / max(den, 1e-30)) ** 0.5}
