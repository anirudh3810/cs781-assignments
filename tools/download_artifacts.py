"""Download and verify archived coursework artifacts using only Python's stdlib."""
import argparse
import hashlib
import json
import pathlib
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', choices=[f'assignment-{n:02d}' for n in range(1, 7)])
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parent.parent
    manifest = json.loads((root / 'publication-manifest.json').read_text(encoding='utf-8'))
    for row in manifest['files']:
        if row['storage'] != 'release' or (args.assignment and not row['path'].startswith(args.assignment + '/')):
            continue
        target = root / row['path']
        if target.resolve().is_relative_to(root.resolve()) is False:
            raise ValueError('Manifest path escapes repository')
        if target.exists() and digest(target) == row['sha256']:
            print('Verified existing:', row['path'])
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + '.part')
        print('Downloading:', row['path'], flush=True)
        request = urllib.request.Request(row['url'], headers={'User-Agent': 'CS781-Artifact-Downloader'})
        with urllib.request.urlopen(request, timeout=120) as response, temp.open('wb') as output:
            while data := response.read(4 * 1024**2):
                output.write(data)
        if temp.stat().st_size != row['bytes'] or digest(temp) != row['sha256']:
            raise RuntimeError('Checksum or size mismatch: ' + row['path'])
        temp.replace(target)
    print('All selected release artifacts verified.')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(data)
    return h.hexdigest()


if __name__ == '__main__':
    main()
