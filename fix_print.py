import pathlib, re

p = pathlib.Path('app.py')
txt = p.read_text(encoding='utf-8')

old = '    print("🚦  Smart Traffic Dashboard (4-Lane) \u2014 http://127.0.0.1:5000")'
new = '    print("Smart Traffic Dashboard (4-Lane) -- http://127.0.0.1:5000")'

if old in txt:
    txt = txt.replace(old, new)
    p.write_text(txt, encoding='utf-8')
    print("Fixed: emoji removed from print statement.")
else:
    # fallback: replace any line containing the startup print
    lines = txt.splitlines()
    changed = False
    for i, line in enumerate(lines):
        if 'Smart Traffic Dashboard' in line and 'print(' in line:
            lines[i] = '    print("Smart Traffic Dashboard (4-Lane) -- http://127.0.0.1:5000")'
            changed = True
            print(f"Fixed line {i+1}")
    if changed:
        p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    else:
        print("Nothing to patch - line not found.")
