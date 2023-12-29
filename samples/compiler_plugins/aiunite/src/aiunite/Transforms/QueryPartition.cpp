// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/CAPI/IR.h"

#include "mhlo/IR/hlo_ops.h"
#include "tosa/transforms/passes.h"

#include "aiunite/client.h"

#include "aiunite/Transforms/Passes.h"

#define GEN_PASS_DEF_QUERYPARTITION
#include "aiunite/Transforms/Passes.h.inc"

namespace mlir::iree_compiler::IREE::AIUnite {
namespace {

class QueryPartitionPass
    : public ::impl::QueryPartitionBase<QueryPartitionPass> {
 public:
  void runOnOperation() override {
    AIUInitialize();
    /// TODO: clone into sub-module, convert to tosa, clone into aiunite
    auto f = getOperation();
    std::string fname = f.getName().str();
    auto *ctx = f->getContext();
    ctx->loadDialect<mlir::tosa::TosaDialect>();
    auto module = ModuleOp::create(UnknownLoc::get(ctx), "foo");
    module.push_back(f.clone());
    PassManager pm(ctx);
    pm.addNestedPass<func::FuncOp>(tosa::createStablehloPrepareForTosaPass());
    pm.addNestedPass<func::FuncOp>(tosa::createStablehloLegalizeToTosaPass());
    auto lr = pm.run(module);
    module.dump();
    if (failed(lr))
      return;

    AIUModel model;
    module.walk(
                [&](mlir::func::FuncOp f) {
                  AIUCloneModel(wrap(&*f), &model);
                });

    AIURequest request;
    AIUSendModel(model, AIU_REQUEST_GET, &request);
    
    AIUSolution solution; // GPU?
    AIUGetSolution(request, 0, &solution);

    // get binary
    AIUBinary binary;
    AIUGetBinary(solution, fname.c_str(), &binary);
    
    const char *binobj;
    AIUGetObject(binary, &binobj);

    MlirModule resMod = AIUGetModule(solution);
    unwrap(resMod).dump();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createQueryPartitionPass() {
  // call once
  return std::make_unique<QueryPartitionPass>();
}

}  // namespace mlir::iree_compiler::IREE::AIUnite
