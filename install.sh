python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple # set mirror
python -m pip install --upgrade pip
python -m pip install uv
export UV_PYTHON_INSTALL_MIRROR=https://repo.huaweicloud.com/python/ # set mirror
python -m uv venv --clear
python -m uv sync
cp doc/unix.sh gtools
chmod +x gtools