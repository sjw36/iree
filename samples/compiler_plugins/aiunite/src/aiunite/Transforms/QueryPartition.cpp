// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/CAPI/IR.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"

#include "aiunite/client.h"

#include "aiunite/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::AIUnite {

#define GEN_PASS_DEF_QUERYPARTITION
#include "aiunite/Transforms/Passes.h.inc"

namespace {

class QueryPartitionPass
    : public impl::QueryPartitionBase<QueryPartitionPass> {
 public:
   void getDependentDialects(DialectRegistry &registry) const override {
     registry.insert<
       IREE::HAL::HALDialect, tosa::TosaDialect, func::FuncDialect,
       pdl::PDLDialect, pdl_interp::PDLInterpDialect>();
  }

  static bool init() {
    AIUInitialize();
    return true;
  }
  void runOnOperation() override {
    static bool initialize = init();
    assert(initialize);
    
    /// TODO: clone into sub-module, convert to tosa, clone into aiunite
    auto f = getOperation();
    std::string fname = f.getName().str();
    auto *ctx = f->getContext();
    auto module = ModuleOp::create(UnknownLoc::get(ctx), "foo");
    module.push_back(f.clone());
    PassManager pm(ctx);
    // stablehlo conversions
    pm.addNestedPass<func::FuncOp>(tosa::createStablehloPrepareForTosaPass());
    pm.addNestedPass<func::FuncOp>(tosa::createStablehloLegalizeToTosaPass());

#if 0
    // Torch conversions
    pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTosaPass());
    pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToArithPass());
    pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTensorPass());
#endif
    
    auto failure = [&](const char *msg, int code = 0) {
                     printf("FAILED(%d): %s\n", code, msg);
                     module.dump();
                     return;
                   };

    auto lr = pm.run(module);
    if (failed(lr))
      return failure("Pass pipeline");

#define AIU_TEST(res)                                   \
    {                                                   \
      int aiu_res = res;                                 \
      if (aiu_res != AIU_SUCCESS)                        \
        return failure(#res, aiu_res);                   \
    }
    

    AIUModel model;
    module.walk([&](mlir::func::FuncOp f) {
                  AIUCloneModel(wrap(&*f), &model);
                });

    AIURequest request;
    AIU_TEST(AIUSendModel(model, AIU_REQUEST_GET, &request));
    
    AIUSolution solution; // GPU?
    AIU_TEST(AIUGetSolution(request, 0, &solution));

#if 0
    // get binary
    AIUBinary binary;
    AIU_TEST(AIUGetBinary(solution, fname.c_str(), &binary));
    
    const char *binobj;
    AIU_TEST(AIUGetObject(binary, &binobj));
#else
    
    MlirModule resMod = AIUGetModule(solution);
    ModuleOp halMod = unwrap(resMod);
    halMod.dump();

    func::FuncOp ftop = halMod.lookupSymbol<func::FuncOp>(fname);
    if (!ftop)
      return failure("Ftop not found");

    // builder on original func parent module
    auto parMod = f->getParentOfType<ModuleOp>();
    OpBuilder b(parMod);
    
    ftop.walk([&](mhal::LaunchOp l) {
        Location loc = l->getLoc();
        CallOpInterface callIf(l);
        if (auto *callable = callIf.resolveCallable()) {
          if (auto func = dyn_cast<func::FuncOp>(callable)) {
            if (auto attr = func->getAttrOfType<ArrayAttr>("mhal.targets")) {
              for (auto targetAttr : attr.getValue()) {
                if (auto kernelPkg = targetAttr.cast<mhal::KernelPackageAttr>()) {
                  // auto arch = kernelPkg.getTarget();
                  auto targetObj = kernelPkg.getObject();
                  // auto binary = targetObj.getBinary();
                  // auto launchDims = kernelPkg.getLaunchDims();
                  // if (launchDims.size() != 2)
                  //   return b.notifyMatchFailure(l, "bad launch dims");
                  // auto gridSize = launchDims[0];
                  // auto blockSize = launchDims[1];

                  // make hal target
                  // compiler/src/iree/compiler/Modules/HAL/Loader/Transforms/MaterializeExecutables.cpp
                  // see bert.hal.mlir
                  auto ename = func.getName() + "_exe";
                  auto exeOp = b.create<IREE::HAL::ExecutableOp>(loc, ename.str());
                  OpBuilder eb(exeOp.getBody());
                  StringRef format = "embedded-elf-aiu";
                  auto data = targetObj.getBinary();
                  std::vector<uint8_t> dataVec(data.begin(), data.end());
                  // StringRef mimeType = "application/x-elf";
                  eb.create<IREE::HAL::ExecutableBinaryOp>(loc, "embedded_elf_aiu",
                                                           format, dataVec);
                }
              }
            }
          }
        }

      });

    parMod.dump();
#endif
    
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createQueryPartitionPass() {
  // call once
  return std::make_unique<QueryPartitionPass>();
}

}  // namespace mlir::iree_compiler::IREE::AIUnite
