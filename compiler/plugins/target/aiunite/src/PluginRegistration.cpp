// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
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

struct AIUniteOptions {
  bool flag = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("IREE AIUnite Plugin");
    binder.opt<bool>("iree-aiunite-flag", flag,
                     llvm::cl::desc("Dummy flag for the example plugin"),
                     llvm::cl::cat(category));
  }
};

class AIUniteTargetBackend : public mlir::iree_compiler::IREE::HAL::TargetBackend {
public:
  AIUniteTargetBackend(const AIUniteOptions &options)
      : options(options) {}

  // NOTE: we could vary this based on the options such as 'metal-v2'.
  std::string name() const override { return "aiunite"; }

  void getDependentDialects(DialectRegistry &registry) const override {
     registry.insert<
       mlir::iree_compiler::IREE::HAL::HALDialect, tosa::TosaDialect, func::FuncDialect,
       pdl::PDLDialect, pdl_interp::PDLInterpDialect,
       IREE::Codegen::IREECodegenDialect,
       IREE::Flow::FlowDialect>();
  }

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildConfigurationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                      OpPassManager &passManager) override {
    // For now we disable configuration if the variant has external object
    // files.
    if (variantOp.isExternal())
      return;

    //buildSPIRVCodegenConfigurationPassPipeline(passManager);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    // For now we disable translation if the variant has external object files.
    // We could instead perform linking with those objects (if they're Metal
    // archives, etc).
    if (variantOp.isExternal())
      return;

    passManager.addPass(IREE::AIUnite::createQueryPartitionPass());
    //buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false);
  }
  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    assert(0);
#if 0
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();
    if (!serOptions.dumpIntermediatesPath.empty()) {
      std::string assembly;
      llvm::raw_string_ostream os(assembly);
      spvModuleOp.print(os, OpPrintingFlags().useLocalScope());
      dumpDataToPath(serOptions.dumpIntermediatesPath, serOptions.dumpBaseName,
                     variantOp.getName(), ".mlir", assembly);
    }

    // The runtime use ordinals instead of names but Metal requires function
    // names for constructing pipeline states. Get an ordered list of the entry
    // point names.
    SmallVector<StringRef, 8> spirvEntryPointNames;
    spvModuleOp.walk([&](spirv::EntryPointOp exportOp) {
      spirvEntryPointNames.push_back(exportOp.getFn());
    });

    // 1. Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
      return variantOp.emitError() << "failed to serialize spirv.module";
    }
    if (!serOptions.dumpIntermediatesPath.empty()) {
      dumpDataToPath<uint32_t>(serOptions.dumpIntermediatesPath,
                               serOptions.dumpBaseName, variantOp.getName(),
                               ".spv", spvBinary);
    }

    // 2. Cross compile SPIR-V to MSL source code.
    SmallVector<MetalShader, 2> mslShaders;
    SmallVector<std::string, 2> mslEntryPointNames;
    mslShaders.reserve(spirvEntryPointNames.size());
    mslEntryPointNames.reserve(spirvEntryPointNames.size());
    for (const auto &entryPoint : spirvEntryPointNames) {
      // We can use ArrayRef here given spvBinary reserves 0 bytes on stack.
      ArrayRef spvData(spvBinary.data(), spvBinary.size());
      std::optional<std::pair<MetalShader, std::string>> msl =
          crossCompileSPIRVToMSL(options.targetPlatform, spvData, entryPoint);
      if (!msl) {
        return variantOp.emitError()
        << "failed to cross compile SPIR-V to Metal shader";
      }   
      mslShaders.push_back(std::move(msl->first));
      mslEntryPointNames.push_back(std::move(msl->second));
    }   

    if (!serOptions.dumpBinariesPath.empty()) {
      for (auto shader : llvm::enumerate(mslShaders)) {
        dumpDataToPath(
            serOptions.dumpBinariesPath, serOptions.dumpBaseName,
            (variantOp.getName() + std::to_string(shader.index())).str(),
            ".metal", shader.value().source);
      }   
    }   

    // 3. Compile MSL to MTLLibrary.
    SmallVector<std::unique_ptr<llvm::MemoryBuffer>> metalLibs;
    if (options.compileToMetalLib) {
      // We need to use offline Metal shader compilers.
      // TODO(#14048): The toolchain can also exist on other platforms. Probe
      // the PATH instead.
      auto hostTriple = llvm::Triple(llvm::sys::getProcessTriple());
      if (hostTriple.isMacOSX()) {
        for (auto [shader, entryPoint] :
             llvm::zip(mslShaders, mslEntryPointNames)) {
          std::unique_ptr<llvm::MemoryBuffer> lib = compileMSLToMetalLib(
              options.targetPlatform, shader.source, entryPoint);
          if (!lib) {
            return variantOp.emitError()
                   << "failed to compile to MTLLibrary from MSL:\n\n"
                   << shader.source << "\n\n";
          }
          metalLibs.push_back(std::move(lib));
        }
      }   
    }   

    // 4. Pack the MTLLibrary and metadata into a FlatBuffer.
    FlatbufferBuilder builder;
    iree_hal_metal_ExecutableDef_start_as_root(builder);

    auto entryPointNamesRef = builder.createStringVec(mslEntryPointNames);
    iree_hal_metal_ExecutableDef_entry_points_add(builder, entryPointNamesRef);

    iree_hal_metal_ThreadgroupSize_vec_start(builder);
    for (auto &shader : mslShaders) {
      iree_hal_metal_ThreadgroupSize_vec_push_create(
          builder, shader.threadgroupSize.x, shader.threadgroupSize.y,
          shader.threadgroupSize.z);
    }
    auto threadgroupSizesRef = iree_hal_metal_ThreadgroupSize_vec_end(builder);
    iree_hal_metal_ExecutableDef_threadgroup_sizes_add(builder,
                                                       threadgroupSizesRef);

    if (metalLibs.empty()) {
      auto shaderSourcesRef = builder.createStringVec(
          llvm::map_range(mslShaders, [&](const MetalShader &shader) {
            return shader.source;
          }));
      iree_hal_metal_ExecutableDef_shader_sources_add(builder,
                                                      shaderSourcesRef);
    } else {
      auto refs = llvm::to_vector<8>(llvm::map_range(
          metalLibs, [&](const std::unique_ptr<llvm::MemoryBuffer> &buffer) {
            return flatbuffers_string_create(builder, buffer->getBufferStart(),
                                             buffer->getBufferSize());
          }));
      auto libsRef =
          flatbuffers_string_vec_create(builder, refs.data(), refs.size());
      iree_hal_metal_ExecutableDef_shader_libraries_add(builder, libsRef);
    }

    iree_hal_metal_ExecutableDef_end_as_root(builder);

    // 5. Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));
#endif
    return success();
  }
private:
  StringAttr getAIUniteTargetEnv(MLIRContext *context) const {
    return StringAttr::get(context, "foobar");
  }
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(
        getExecutableTarget(context, getAIUniteTargetEnv(context)));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr
  getExecutableTarget(MLIRContext *context,
                      StringAttr targetEnv) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr("aiunite.target_env"),
                             targetEnv);

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("aiunite"), b.getStringAttr("aiunite-msl-fb"),
        configAttr);
  }

  const AIUniteOptions &options;
};

struct AIUniteSession
    : public PluginSession<AIUniteSession, AIUniteOptions,
                           PluginActivationPolicy::DefaultActivated> {

  void populateHALTargetBackends(IREE::HAL::TargetBackendList &targets) {
     // #hal.device.target<"metal", ...
     targets.add("aiunite", [=]() {
       return std::make_shared<AIUniteTargetBackend>(options);
     });
   }

};

}  // namespace

extern "C" bool iree_register_compiler_plugin_hal_target_aiunite(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<AIUniteSession>("hal_target_aiunite");
  return true;
}

IREE_DEFINE_COMPILER_OPTION_FLAGS(AIUniteOptions);
