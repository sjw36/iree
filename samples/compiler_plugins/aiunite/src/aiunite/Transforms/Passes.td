// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef IREE_AIUNITE_PASSES
#define IREE_AIUNITE_PASSES

def QueryPartition : Pass<"iree-aiunite-partition", "mlir::func::FuncOp"> {
  let summary = "Query AIUnite to partition ops";
  let constructor = [{
    ::mlir::iree_compiler::IREE::AIUnite::createQueryPartitionPass()
  }];
  let dependentDialects = [
    "mlir::tosa::TosaDialect", "mlir::func::FuncDialect", "mlir::pdl::PDLDialect", "mlir::pdl_interp::PDLInterpDialect"
  ];
}

#endif // IREE_AIUNITE_PASSES
