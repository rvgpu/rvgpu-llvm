#! /bin/bash

function build()
{
    mkdir -p ${install_dir}

    cmake \
        -S /source/llvm \
        -B ${workspace}/build \
        -G Ninja \
        -DLLVM_ENABLE_RTTI=on \
        -DCMAKE_INSTALL_PREFIX=${install_dir} \
        -DCMAKE_BUILD_TYPE=release \
        -DBUILD_SHARED_LIBS=on \
        -DLLVM_BUILD_LLVM_DYLIB=off \
        -DLLVM_ENABLE_PROJECTS="clang;lld;compiler-rt" \
        -DLLVM_TARGETS_TO_BUILD="RISCV;X86"

    if [ $? -ne 0 ]; then
        echo "cmake error and exit"
        exit -1
    fi

    cmake --build ${workspace}/build
    if [ $? -ne 0 ]; then
        echo "cmake build error and exit"
        exit -1
    fi

    cmake --install ${workspace}/build
    if [ $? -ne 0 ]; then
        echo "install error"
        exit -1
    fi
}

function package_deb()
{
    mkdir -p debian
    rsync -rat /source/.ci/debian-package/* ${workspace}/debian/

    pushd ${workspace}
        dpkg -b debian/ /source/rvgpu-llvm_0.2.2_amd64.deb
    popd
}

workspace=/source/cibuild/
install_dir=${workspace}/debian/usr/local/rvgpu
mkdir -p ${install_dir}
# Build LLVM
build

# Test

# Package
package_deb

# cl
rm -rf ${workspace}
