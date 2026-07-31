#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
cat stage1f-public/kernel-xz.part.00 \
    stage1f-public/kernel-xz.part.01 \
    stage1f-public/kernel-xz.part.02 \
    stage1f-public/kernel-xz.part.03 \
    stage1f-public/kernel-xz.part.04 \
    stage1f-public/kernel-xz.part.05 \
    stage1f-public/kernel-xz.part.06 \
    stage1f-public/kernel-xz.part.07 \
    stage1f-public/kernel-xz.part.08 \
  | base64 -d > build/turbo-kernel.elf.xz
printf '%s  %s\n' '22833f788b1b79ea52d36062d8bdf4cd468266382b11cc5ff339763f8223cf50' 'build/turbo-kernel.elf.xz' | sha256sum -c -
xz -dc build/turbo-kernel.elf.xz > build/turbo-kernel.elf
printf '%s  %s\n' '3993eb7c7c31503db8e4e72fe4634a5add02d4ea7e904b4f9839e75b23e369fd' 'build/turbo-kernel.elf' | sha256sum -c -
test "$(stat -c %s build/turbo-kernel.elf)" = '119320'
