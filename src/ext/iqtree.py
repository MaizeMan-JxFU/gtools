import os
import platform
from .downloader import download_joblib
import gzip
import shutil
import sys

def main():
    scriptpath = os.path.dirname(os.path.abspath(__file__))
    platform_name = platform.system()
    assert platform_name in ['Linux', 'Darwin'], 'Only Linux and macOS are supported.'
    arc = {'Linux': 'x86_64', 'Darwin': 'universal'}
    os.makedirs(f'{scriptpath}/../../ext/bin/', exist_ok=True)
    extbin = f'{scriptpath}/../../ext/bin'
    # Denpendency check
    ## iqtree
    if not os.path.isfile(f'{extbin}/iqtree3'):
        print('Downloading and compiling IQ-TREE...')
        url = f'http://maize.jxfu.top:23000/Jingxian/JanusXext/raw/main/package/iqtree3-{platform_name}-{arc[platform_name]}.gz'
        download_joblib(url, f'{extbin}/iqtree3.gz', n_jobs=-1)
        with gzip.open(f'{extbin}/iqtree3.gz', "rb") as f_in, open(f'{extbin}/iqtree3', "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.chmod(f'{extbin}/iqtree3', 0o755)
        os.remove(f'{extbin}/iqtree3.gz')
        print('IQ-TREE downloaded and compiled successfully.')
    ## Main
    os.system(f'{extbin}/iqtree3 {" ".join(sys.argv[1:])}')

if __name__ == '__main__':
    main()