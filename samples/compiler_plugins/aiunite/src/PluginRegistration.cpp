// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include "aiunite/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace detail {
namespace {

#define GEN_PASS_REGISTRATION
#include "aiunite/Transforms/Passes.h.inc"

}  // namespace
}  // namespace detail

namespace {

struct MyOptions {
  bool flag = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("IREE AIUnite Plugin");
    binder.opt<bool>("iree-aiunite-flag", flag,
                     llvm::cl::desc("Dummy flag for the example plugin"),
                     llvm::cl::cat(category));
  }
};

struct MySession : public PluginSession<MySession, MyOptions> {
  static void registerPasses() { ::detail::registerPasses(); }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<mlir::tosa::TosaDialect>();
  }

  LogicalResult onActivate() override { return success(); }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    // from third_party/mlir-hlo
    pm.addNestedPass<func::FuncOp>(IREE::AIUnite::createQueryPartitionPass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool iree_register_compiler_plugin_aiunite(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<MySession>("aiunite");
  return true;
}
