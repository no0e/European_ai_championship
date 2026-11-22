"""Lecture des réponses longues (answers_long_ep.jsonl) : batteries -> item -> réponse normalisée ; suites -> symboles."""
import collections, json, re
from pathlib import Path
import plan2
T = Path(__file__).resolve().parent.parent

def load(fn='answers_long_ep.jsonl'):
    D = collections.defaultdict(lambda: collections.defaultdict(list))     # probe -> endpoint -> [(rep, texte)]
    for l in open(T / fn, encoding='utf-8'):
        if l.strip():
            r = json.loads(l); D[r['probe']][r['endpoint']].append((r['repeat'], r['response']))
    for p in D:
        for e in D[p]: D[p][e] = [x for _, x in sorted(D[p][e])]
    return D

def _norm(a):
    a = a.strip().strip('`*"\' ').rstrip('.').strip()
    return re.sub(r'\s+', ' ', a).lower()

def battery(text, n_items):
    """`n. réponse` par ligne -> {n: réponse}. Les lignes sans numéro sont ignorées."""
    out = {}
    for line in text.split('\n'):
        m = re.match(r'\s*(\d{1,2})\s*[.):]\s*(.*)$', line)
        if m and 1 <= int(m.group(1)) <= n_items and int(m.group(1)) not in out:
            out[int(m.group(1))] = _norm(m.group(2))
    return out

TOOL_OPTS = {}
for line in plan2.TOOL.split('\n')[1:]:
    n, rest = line.split('.', 1)
    TOOL_OPTS[int(n)] = [t.strip() for t in rest.rsplit('Tools:', 1)[1].split(',')]
_OWNER = {}
for n, opts in TOOL_OPTS.items():
    for t in opts: _OWNER.setdefault(t, n)

def tools(text):
    """Choix d'outil par tâche = PREMIER nom d'outil de la ligne (certains tirages doublent la réponse, « x. tool_name: x »,
    ou listent plusieurs options : le premier nom reste le choix). Attribution par appartenance à la liste d'options,
    donc robuste à un décalage de numérotation."""
    out = {}
    for line in text.split('\n'):
        hits = [t for t in re.findall(r'[a-z_]+', line.lower()) if t in _OWNER]
        if hits and _OWNER[hits[0]] not in out: out[_OWNER[hits[0]]] = hits[0]
    return out

def tool_format(text):
    """Forme de la réponse : 'net' (un nom par ligne), 'double' (nom répété), 'liste' (plusieurs options), 'autre'."""
    lines = [l for l in text.split('\n') if l.strip()][:6]
    n = [len([t for t in re.findall(r'[a-z_]+', l.lower()) if t in _OWNER]) for l in lines]
    if not n or max(n) == 0: return 'autre'
    if 'tool_name' in text[:200]: return 'double'
    return 'liste' if sum(x > 1 for x in n) >= 2 else 'net'

def coin(text):
    """Suite de pile ou face, avec ou sans séparateurs. Si le texte n'est pas majoritairement fait de H et de T (phrases),
    on ne garde que les H/T isolés."""
    al = [c for c in text if c.isalpha()]
    if al and sum(c in 'HTht' for c in al) / len(al) >= 0.8: return [c.upper() for c in al if c in 'HTht']
    return re.findall(r'\b([HT])\b', text)
def digits(text): return re.findall(r'[0-9]', text)
def seq_format(text):
    """Forme de la suite : taille du premier groupe de caractères collés (1 = « 8 4 5 », 3 = « 119 803 », 0 = vide)."""
    m = re.match(r'\s*([0-9A-Za-z]+)', text)
    return min(len(m.group(1)), 9) if m else 0

def items(D, eps):
    """endpoint -> item -> [réponses] pour toutes les batteries. Un item absent d'une réponse compte comme '∅'."""
    A = {e: collections.defaultdict(list) for e in eps}
    for e in eps:
        for txt in D['bat_frag'].get(e, []):
            b = battery(txt, 22)
            for i in range(1, 23): A[e][f'frag{i:02d}'].append(b.get(i, '∅'))
        for txt in D['bat_tie'].get(e, []):
            b = battery(txt, 24)
            for i in range(1, 25): A[e][f'tie{i:02d}'].append(b.get(i, '∅'))
        for txt in (D['bat_tool_l'].get(e, []) if plan2.FAM.get(e) == 'LLAMA' else D['bat_tool'].get(e, [])):
            b = tools(txt)
            for i in range(1, 21): A[e][f'tool{i:02d}'].append(b.get(i, '∅'))
    return A
