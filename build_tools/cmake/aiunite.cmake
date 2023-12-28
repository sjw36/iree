# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(AIUNITE_PROJECT_DIR "${IREE_SOURCE_DIR}/third_party/aiunite/")
set(AIU_ENABLE_LOGGING OFF)

set(AIUNITE_LLVM_PROJECT_DIR "${IREE_SOURCE_DIR}/third_party/llvm-project")
set(AIUNITE_LLVM_BINARY_DIR "${IREE_BINARY_DIR}/third_party/llvm-project")
set(AIUNITE_USE_LOCAL_LLVM OFF)
set(AIUNITE_ENABLE_RTTI OFF)
set(AIUNITE_ENABLE_EXCEPTIONS OFF)

add_subdirectory("${AIUNITE_PROJECT_DIR}" "third_party/aiunite" EXCLUDE_FROM_ALL)

include_directories(${AIUNITE_PROJECT_DIR}/include)
