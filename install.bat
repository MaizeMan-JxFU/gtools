python -m pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
python -m pip install uv
set UV_PYTHON_INSTALL_MIRROR=https://repo.huaweicloud.com/python/
python -m uv venv --clear
python -m uv sync
copy doc\windows.bat gtools.bat