// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
//#include "mlir/Dialect/MHAL/IR/MHAL.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/conversions/tosa/transforms/Passes.h"

#if 0
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Conversion/TorchToArith/TorchToArith.h"
#include "torch-mlir/Conversion/TorchToSCF/TorchToSCF.h"
#include "torch-mlir/Conversion/TorchToTosa/TorchToTosa.h"
#include "torch-mlir/Conversion/TorchToTensor/TorchToTensor.h"
#endif

#include "aiunite/client.h"

#include "aiunite/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::AIUnite {

#define GEN_PASS_DEF_QUERYPARTITION
#include "aiunite/Transforms/Passes.h.inc"

namespace {

////////////////////////////////////////////////////////////////////////
struct FlowLoadRewritePattern
    : public OpRewritePattern<IREE::Flow::DispatchTensorLoadOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorLoadOp ldOp,
                                PatternRewriter &rw) const final {
    // %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offs     et(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x5xf32>>
    // %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [3, 5], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<3x5xf32>> -> tensor<3x5xf32>
    auto *src = ldOp.getSource().getDefiningOp();
    auto bind = cast<IREE::HAL::InterfaceBindingSubspanOp>(src);
    assert(bind);
    int32_t idx = bind.getBindingAttr().getInt();
    auto type = bind.getResult().getType().template cast<IREE::Flow::DispatchTensorType>();
    assert(type.getAccess() == IREE::Flow::TensorAccess::ReadOnly);
    auto func = bind->getParentOfType<func::FuncOp>();
    auto inputArg = func.getArgument(idx);
    rw.replaceOp(ldOp, inputArg);
    return success();
  }
};
  
////////////////////////////////////////////////////////////////////////
struct FlowStoreRewritePattern
    : public OpRewritePattern<IREE::Flow::DispatchTensorStoreOp> {
  using OpRewritePattern<IREE::Flow::DispatchTensorStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::Flow::DispatchTensorStoreOp stOp,
                                PatternRewriter &rw) const final {
    // flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [3, 3], strides = [1, 1] : tensor<3x3xf32> -> !flow.dispatch.tensor<readwrite:tensor<3x3xf32>>
    // assert(only 1 store) // currently does not support multiple outputs
    rw.replaceOpWithNewOp<func::ReturnOp>(stOp, stOp.getValue());
    return success();
  }
};

////////////////////////////////////////////////////////////////////////
struct EmptyReturnRewritePattern
  : public OpRewritePattern<func::ReturnOp> {
  using OpRewritePattern<func::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(func::ReturnOp retOp,
                                PatternRewriter &rw) const final {
    if (retOp.getNumOperands() != 0)
      return rw.notifyMatchFailure(retOp, "has return value");
    rw.eraseOp(retOp);
    return success();
  }
};

////////////////////////////////////////////////////////////////////////
struct LinalgMatmulRewritePattern
  : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp mmOp,
                                PatternRewriter &rw) const final {
    // outs (C) must be fill of zeros
    // TODO: verify

    rw.replaceOpWithNewOp<tosa::MatMulOp>(mmOp, mmOp.getResult(0).getType(), mmOp.getInputs());
    return success();
  }
};


////////////////////////////////////////////////////////////////////////
class QueryPartitionPass
    : public impl::QueryPartitionBase<QueryPartitionPass> {
 public:

  static bool init() {
    AIUInitialize();
    return true;
  }
  void runOnOperation() override {
    static bool initialize = init();
    assert(initialize);
    
    auto failure = [&](const char *msg, int code = 0) {
                     printf("FAILED(%d): %s\n", code, msg);
                     //module.dump();
                     // remove target
                     //signalPassFailure();
                     return;
                   };

    /// TODO: clone into sub-module, convert to tosa, clone into aiunite
    auto variantOp = getOperation();
    func::FuncOp f;
    variantOp.walk([&](func::FuncOp func) {
                     f = func; // last
                });
    std::string fname = f.getName().str();
    auto *ctx = f->getContext();
    auto module = ModuleOp::create(UnknownLoc::get(ctx), "foo");

    auto aiuFunc = f.clone();
    module.push_back(aiuFunc);

    {
      f.walk([&](IREE::HAL::InterfaceBindingSubspanOp ifOp) {
               // %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offs     et(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<3x5xf32>>
               int32_t idx = ifOp.getBindingAttr().getInt();
               auto type = ifOp.getResult().getType().template cast<IREE::Flow::DispatchTensorType>();
               if (type.getAccess() == IREE::Flow::TensorAccess::ReadOnly) {
                 aiuFunc.insertArgument(idx, type.getBoundType(), {}, ifOp.getLoc());
               } else {
                 // assert WriteOnly
                 //assert(idx == results.size());
                 assert(aiuFunc.getNumResults() == 0);
                 aiuFunc.insertResult(0, type.getBoundType(), {});
               }
             });
    }
    
    {
      // convert to proper form (no Flow/HAL ops)
      RewritePatternSet patterns(ctx);
      patterns.add<FlowLoadRewritePattern, FlowStoreRewritePattern,
                   EmptyReturnRewritePattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(aiuFunc, std::move(patterns)))) {
        return failure("hal conversion");
      }
    }
    
    {
      // convert to TOSA
      // TODO: make a conversion pass
      RewritePatternSet patterns(ctx);
      patterns.add<LinalgMatmulRewritePattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(aiuFunc, std::move(patterns)))) {
        return failure("hal conversion");
      }
    }
    
    PassManager pm(ctx);

#if 0
    // stablehlo conversions
    pm.addNestedPass<func::FuncOp>(tosa::createStablehloPrepareForTosaPass());
    pm.addNestedPass<func::FuncOp>(tosa::createStablehloLegalizeToTosaPass());

    // Torch conversions
    pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTosaPass());
    pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToArithPass());
    pm.addNestedPass<func::FuncOp>(torch::createConvertTorchToTensorPass());
    
    auto lr = pm.run(module);
    if (failed(lr))
      return failure("Pass pipeline");

#endif


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
#if 0 
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
#endif
    parMod.dump();
#endif
    
  }
};

}  // namespace

std::unique_ptr<OperationPass<iree_compiler::IREE::HAL::ExecutableVariantOp>> createQueryPartitionPass() {
  // call once
  return std::make_unique<QueryPartitionPass>();
}

}  // namespace mlir::iree_compiler::IREE::AIUnite
