"""Download WebShop assets from the URLs used by the pinned ETO setup script."""
import argparse
from pathlib import Path, PurePosixPath
import os
import shutil
import stat
import tempfile
import zipfile

ASSETS = {
    'data': '1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9',
    'indexes': '11zOUDkJSgGhYin9NxQtG8PVpDsika86y',
}


def ready(root, kind):
    if kind == 'data':
        files = [root/name for name in ('items_shuffle.json','items_ins_v2.json','items_human_ins.json')]
        return all(p.is_file() and p.stat().st_size > 0 for p in files)
    segments = list(root.glob('segments_*'))
    shards = list(root.glob('*.si'))
    return bool(segments and shards) and all(p.stat().st_size > 0 for p in segments + shards)


def extract_checked(archive, destination):
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            if (path.is_absolute() or '..' in path.parts or
                    '\\' in member.filename or ':' in member.filename or stat.S_ISLNK(mode)):
                raise ValueError(f'Unsafe archive path: {member.filename}')
        # Reading each entry during extraction verifies its CRC.
        bundle.extractall(destination)


def install_asset(package, kind, downloader=None):
    target = package/'data' if kind == 'data' else package/'search_index/indexes'
    pending = target/".setup-incomplete"
    if ready(target, kind) and not pending.exists():
        print(f'WebShop {kind}: existing files found; skipping download.', flush=True)
        return
    cache = package/'.asset_downloads'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache/f'{kind}.zip'
    if not archive.is_file():
        if downloader is None:
            import gdown
            downloader = gdown.download
        partial = archive.with_suffix('.zip.part')
        print(f'Downloading WebShop {kind} from ETO data source...', flush=True)
        result = downloader(id=ASSETS[kind], output=str(partial), quiet=False, resume=True)
        if not result or not zipfile.is_zipfile(partial):
            raise RuntimeError(f'WebShop {kind} download incomplete; rerun setup.sh to retry.')
        partial.replace(archive)
    with tempfile.TemporaryDirectory(prefix=f'{kind}-', dir=cache) as temporary:
        staging = Path(temporary)
        extract_checked(archive, staging)
        candidates = [staging/kind, staging]
        source = next((p for p in candidates if ready(p, kind)), None)
        if source is None:
            raise ValueError(f'Unexpected WebShop {kind} archive layout: {archive}')
        target.mkdir(parents=True, exist_ok=True)
        pending.touch()
        for item in source.rglob('*'):
            if not item.is_file():
                continue
            dest = target/item.relative_to(source)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Preserve existing complete files; replace only absent/empty files.
            if dest.is_file() and dest.stat().st_size == item.stat().st_size:
                continue
            if dest.exists() and dest.stat().st_size > 0:
                raise ValueError(f'Existing file differs in size; preserved: {dest}')
            fd, tmp = tempfile.mkstemp(prefix=dest.name+'.', suffix='.part', dir=dest.parent)
            os.close(fd)
            try:
                shutil.copyfile(item, tmp)
                os.replace(tmp, dest)
            finally:
                Path(tmp).unlink(missing_ok=True)
    if not ready(target, kind):
        raise RuntimeError(f'WebShop {kind} is incomplete after extraction')
    pending.unlink(missing_ok=True)
    archive.unlink()
    print(f'WebShop {kind}: ready at {target}', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eto-root', type=Path, required=True)
    a = p.parse_args()
    package = a.eto_root.resolve()/'envs/webshop'
    if not (package/'src/webshop').is_dir():
        raise ValueError('Download the ETO checkout before preparing WebShop assets')
    for kind in ASSETS:
        install_asset(package, kind)

if __name__ == '__main__':
    main()
