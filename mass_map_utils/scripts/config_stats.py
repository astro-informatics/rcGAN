import re


def write_stats(config_path, stats):
    with open(config_path) as f:
        text = f.read()
    for key, val in stats.items():
        new = repr(float(val))
        if 'e' in new and '.' not in new:
            new = new.replace('e', '.0e')  # PyYAML (YAML 1.1) loads '5e-05' as a string
        text, n = re.subn(rf'^{re.escape(key)}:[ \t]*([^#\s]*).*$',
                          lambda m: f'# {m.group(1)}\n{key}: {new}', text, count=1, flags=re.M)
        if n != 1:
            raise KeyError(f'{key} not found in {config_path}')
    with open(config_path, 'w') as f:
        f.write(text)
