#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=${PROJECT_ROOT:-$(cd "${script_dir}/../.." && pwd)}
accv_root=${ACCV_ROOT:-${project_root}/.runtime/deps/ACCV-Lab}
accv_repo_url=${ACCV_REPO_URL:-https://github.com/NVIDIA/ACCV-Lab.git}
accv_commit=${ACCV_COMMIT:-3791096a0ebc64ecab8b4f3401574ae7f1303586}
python_bin=${PYTHON_BIN:-/root/venvs/isaac-groot-a100/bin/python}
ffmpeg_prefix=${FFMPEG_44_PREFIX:-/root/local/ffmpeg-4.4.6}
ffmpeg_version=4.4.6

if [[ ! -d "${accv_root}/.git" ]]; then
    mkdir -p "$(dirname "${accv_root}")"
    git clone --recurse-submodules "${accv_repo_url}" "${accv_root}"
fi
if [[ "$(git -C "${accv_root}" rev-parse --verify HEAD)" != "${accv_commit}" ]]; then
    git -C "${accv_root}" fetch origin "${accv_commit}"
    git -C "${accv_root}" checkout --detach "${accv_commit}"
fi
git -C "${accv_root}" submodule update --init --recursive

expected_commit=$(git -C "${accv_root}" rev-parse --verify HEAD)
expected_short=$(git -C "${accv_root}" rev-parse --short=9 HEAD)
installed_version=$(
    "${python_bin}" -c \
        'import importlib.metadata; print(importlib.metadata.version("accvlab.on_demand_video_decoder"))' \
        2>/dev/null || true
)

if [[ "${installed_version}" == *"g${expected_short}"* ]] \
    && "${python_bin}" -c 'import accvlab.on_demand_video_decoder' >/dev/null 2>&1; then
    "${python_bin}" -c \
        'import accvlab.on_demand_video_decoder as nvc, importlib.metadata as m; print(nvc.__file__, m.version("accvlab.on_demand_video_decoder"))'
    exit 0
fi

echo "Building ACCV-Lab decoder from ${expected_commit} (installed=${installed_version:-none})"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
    build-essential cmake curl nasm ninja-build pkg-config yasm xz-utils

build_root=$(mktemp -d /tmp/accv-decoder-build.XXXXXX)
trap 'rm -rf "${build_root}"' EXIT

if [[ ! -f "${ffmpeg_prefix}/include/libavcodec/avcodec.h" ]]; then
    curl -fL \
        "https://ffmpeg.org/releases/ffmpeg-${ffmpeg_version}.tar.xz" \
        -o "${build_root}/ffmpeg.tar.xz"
    tar -C "${build_root}" -xf "${build_root}/ffmpeg.tar.xz"
    pushd "${build_root}/ffmpeg-${ffmpeg_version}"
    ./configure \
        --prefix="${ffmpeg_prefix}" \
        --enable-shared \
        --disable-static \
        --disable-programs \
        --disable-doc \
        --disable-debug
    make -j"$(nproc)"
    make install
    popd
fi

mkdir -p "${build_root}/accv-source"
tar -C "${accv_root}" \
    --exclude='packages/on_demand_video_decoder/_skbuild' \
    --exclude='packages/on_demand_video_decoder/.pytest_cache' \
    --exclude='packages/on_demand_video_decoder/*.egg-info' \
    -cf - .nav build_config packages/on_demand_video_decoder \
    | tar -C "${build_root}/accv-source" -xf -

export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ACCVLAB_BUILD_CONFIG=0.0.0+g${expected_short}
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_ACCVLAB_ON_DEMAND_VIDEO_DECODER=0.0.0+g${expected_short}

# Build wheels through setup.py so the decoder's repository-relative build
# requirement is not re-resolved by the package installer. Dependencies are
# supplied by the already-synchronized training environment and must not be
# upgraded while installing ACCV-Lab.
uv pip install --python "${python_bin}" \
    'distro==1.9.0' 'ninja==1.13.0' 'scikit-build==0.19.1' \
    'setuptools==80.9.0' 'setuptools-scm==10.2.1' 'wheel==0.47.0'
pushd "${build_root}/accv-source/build_config"
"${python_bin}" setup.py bdist_wheel --dist-dir "${build_root}/wheels"
popd
build_config_wheel=$(find "${build_root}/wheels" -maxdepth 1 -name 'accvlab_build_config-*.whl' -print -quit)
uv pip install --python "${python_bin}" --reinstall --no-deps "${build_config_wheel}"

pushd "${build_root}/accv-source/packages/on_demand_video_decoder"
FFMPEG_DIR="${ffmpeg_prefix}" CUSTOM_CUDA_ARCHS=80 \
    "${python_bin}" setup.py bdist_wheel --dist-dir "${build_root}/wheels"
popd
decoder_wheel=$(find "${build_root}/wheels" -maxdepth 1 -name '*on_demand_video_decoder-*.whl' -print -quit)
uv pip install --python "${python_bin}" --reinstall --no-deps "${decoder_wheel}"

LD_LIBRARY_PATH=${project_root}/.runtime/nvidia-driver:${LD_LIBRARY_PATH:-} \
    "${python_bin}" -c \
        'import accvlab.on_demand_video_decoder as nvc, importlib.metadata as m; print(nvc.__file__, m.version("accvlab.on_demand_video_decoder"))'
