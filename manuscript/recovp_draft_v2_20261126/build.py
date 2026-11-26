import re, json, pathlib, sys
d = pathlib.Path(__file__).parent
lang = sys.argv[1] if len(sys.argv) > 1 else 'zh'
parts = sorted(d.glob(f'{lang}_part*.md'))
text = '\n'.join(p.read_text(encoding='utf-8') for p in parts)
refs = json.loads((d / 'refs.json').read_text(encoding='utf-8'))
cite = re.compile(r'\[(\d+(?:\s*[–,]\s*\d+)*)\]')
def expand(s):
    out = []
    for tok in re.split(r'\s*,\s*', s):
        if '–' in tok:
            a, b = [int(x) for x in tok.split('–')]
            out += list(range(a, b + 1))
        else:
            out.append(int(tok))
    return out
order = []
body, _, tail = text.partition('{{REFERENCES}}')
mathre = re.compile(r'(\$\$.*?\$\$|\$[^\$\n]*?\$)', re.S)
def outside(s, f):
    segs = mathre.split(s)
    return ''.join(seg if mathre.fullmatch(seg) else f(seg) for seg in segs)
def scan(seg):
    for m in cite.finditer(seg):
        for n in expand(m.group(1)):
            if n not in order: order.append(n)
    return seg
outside(body, scan); outside(tail, scan)
missing = [n for n in order if str(n) not in refs]
unused = [k for k in refs if int(k) not in order]
new = {old: i + 1 for i, old in enumerate(order)}
def compress(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else (f'{nums[i]},{nums[j]}' if j == i + 1 else f'{nums[i]}–{nums[j]}'))
        i = j + 1
    return ','.join(out)
def sub(m):
    return '[' + compress([new[n] for n in expand(m.group(1))]) + ']'
body = outside(body, lambda g: cite.sub(sub, g)); tail = outside(tail, lambda g: cite.sub(sub, g))
reflist = '\n\n'.join(f'[{new[o]}] {refs[str(o)]}' for o in order)
out = body + reflist + tail
(d / f'ReCoVP_Hybrid_{"中文" if lang=="zh" else "English"}_draft.md').write_text(out, encoding='utf-8')
print('refs used', len(order), 'missing', missing, 'unused', unused)
print('chars', len(out))
