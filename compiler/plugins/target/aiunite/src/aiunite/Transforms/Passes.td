// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef IREE_AIUNITE_PASSES
#define IREE_AIUNITE_PASSES

def QueryPartition : Pass<"iree-aiunite-partition",
     "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Query AIUnite to partition ops";
  let constructor = [{
    ::mlir::iree_compiler::IREE::AIUnite::createQueryPartitionPass()
  }];
}

#endif // IREE_AIUNITE_PASSES
