// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_COMPILER_PLUGINS_AIUNITE_TRANSFORMS_PASSES_H_
#define IREE_SAMPLES_COMPILER_PLUGINS_AIUNITE_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::AIUnite {

std::unique_ptr<OperationPass<iree_compiler::IREE::HAL::ExecutableVariantOp>> createQueryPartitionPass();


}  // namespace mlir::iree_compiler::IREE::AIUnite

#endif  // IREE_SAMPLES_COMPILER_PLUGINS_AIUNITE_TRANSFORMS_PASSES_H_
