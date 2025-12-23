pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple # set mirror
pip install --upgrade pip
pip install uv
uv venv --clear
uv sync
uv pip install -e ./ext/glm_rs
uv pip install -e ./ext/lmm_rs
uv pip install -e ./ext/geno2phylip
uv pip install -e ./ext/gfreader_rs
cp ./doc/unix.sh jx
./jx -h
echo "Recommend: Add $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd) to PATH"