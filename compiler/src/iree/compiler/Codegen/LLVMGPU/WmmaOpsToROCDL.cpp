//===------ WmmaOpsToAMDGPU.cpp - WMMA LD/ST/Compute to AMDGPU lowering -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of patterns to lower GPU Subgroup MMA ops to
// AMDGPU Dialect.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/CommonPasses.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;

namespace {


template <typename T>
struct AMDMMAInfo {
  T mmaOp;
  // !gpu.mma_matrix<16x8xf32, "AOp">
  gpu::MMAMatrixType mmaType;
  MemRefType regType;

  template <typename TC>
  gpu::MMAMatrixType getMatrixType() {
    if (mmaOp && !mmaType) {
      mmaType = cast<gpu::MMAMatrixType>(mmaOp.getRes().getType());
    }
    return mmaType;
  }

  template <>
  gpu::MMAMatrixType getMatrixType<gpu::SubgroupMmaStoreMatrixOp>() {
    if (mmaOp && !mmaType) {
      mmaType = cast<gpu::MMAMatrixType>(mmaOp.getSrc().getType());
    }
    return mmaType;
  }

  AMDMMAInfo(T op) : mmaOp(op) {
    getMatrixType<T>();
  }

  AMDMMAInfo(gpu::MMAMatrixType t) {
    mmaType = t;
  }

  MemRefType getRegType() {
    if (!regType) {
      assert(mmaType);
      auto shape = mmaType.getShape();
      int64_t numElems = 1;
      llvm::for_each(shape, [&numElems](int64_t v) { numElems *= v; });
      auto elemType = mmaType.getElementType();
      int64_t elemBytes = elemType.getIntOrFloatBitWidth() / 8;
      assert(elemBytes == 4); // @@
      int64_t numRegs = numElems / (64 * 4 / elemBytes);
      auto private_as = gpu::AddressSpaceAttr::get(mmaType.getContext(),
                                                   gpu::GPUDialect::getPrivateAddressSpace());
      MemRefLayoutAttrInterface layout;
      regType = MemRefType::get({numRegs}, elemType, layout, private_as);
    }
    return regType;
  }

  SmallVector<Value, 4> computeIndices(RewriterBase &rewriter, Value lane, int64_t gpr_index) {
    SmallVector<Value, 4> newCoords;
    Location loc = mmaOp->getLoc();
    auto coords = mmaOp.getIndices();

    uint64_t wave_size = 64;
    // Convert from base shared-mem addressing to lane adjusted shared mem addressing
    
    // %34 = gpu.subgroup_mma_load_matrix %alloc_2[%25, %33, %c0] {leadDimension = 20 : index}
    //     : memref<4x32x20xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">

    auto computeStridedAxis = [&](Value offset, int64_t size) {
      // addr = offset + stride * (lane / size) + (gpr * stride)
      int64_t stride = wave_size / size;
      Value size_v = rewriter.createOrFold<arith::ConstantIndexOp>(loc, size);
      Value strd_v = rewriter.createOrFold<arith::ConstantIndexOp>(loc, stride);
      Value ldiv_v = rewriter.create<arith::DivUIOp>(loc, lane, size_v); // (lane / m_size)
      Value lmul_v = rewriter.create<arith::MulIOp>(loc, ldiv_v, strd_v); // stride * ^
      Value ladd_v = rewriter.create<arith::AddIOp>(loc, offset, lmul_v); // y + ^
      if (gpr_index) {
        Value gpr_idx_v = rewriter.createOrFold<arith::ConstantIndexOp>(loc, gpr_index * stride);
        ladd_v = rewriter.create<arith::AddIOp>(loc, ValueRange{ladd_v, gpr_idx_v});
      }
      return ladd_v;
    };

    auto computeModAxis = [&](Value offset, int64_t size) {
      // asix = offset + (lane % size)
      Value size_v = rewriter.createOrFold<arith::ConstantIndexOp>(loc, size);
      Value lmod = rewriter.create<arith::RemUIOp>(loc, lane, size_v);
      return rewriter.create<arith::AddIOp>(loc, ValueRange{offset, lmod});
    };
    

    if (mmaType.getOperand() == "AOp") {
      // A = BxMxK = 4x32x20
      // -- Tile = MxK = 16x4
      // -- R0 =  0,0  1,0 ... 15,0 | 0,1  1,1 ... 15,1 | 0,2  1,2 ... 15,2 | 0,3 1,3 ... 15,3
      // -- R1 =  0,4  1,4 ... 15,4 | 0,5  1,5 ... 15,5 | 0,6  1,6 ... 15,6 | 0,7 1,7 ... 15,7

      // coords = [b, y, x]

      // batch size
      newCoords.push_back(coords[0]);

      // m = y + (lane % m_size)
      int64_t m_size = mmaType.getShape()[0];
      newCoords.push_back(computeModAxis(coords[1], m_size));
      
      // k = x + stride * (lane / m_size) + (gpr * stride)
      // start +        inter_reg         +    gpr_offset
      newCoords.push_back(computeStridedAxis(coords[2], m_size));
      
    } else if (mmaType.getOperand() == "BOp") {
      // B = BxKxN = 4x16x36
      // -- Tile = KxN = 4x16
      // -- Reg =  0,0  0,1 ... 0,15 | 1,0  1,1 ... 1,15 | 2,0  2,1 ... 2,15 | 3,0 3,1 ... 3,15

      // coords = [b, y, x]

      // batch size
      newCoords.push_back(coords[0]);

      // k = y + stride * (lane / k_size) + (gpr * stride)
      int64_t k_size = mmaType.getShape()[0];
      newCoords.push_back(computeStridedAxis(coords[1], k_size));
      
      // j = x + (lane % k_size)
      newCoords.push_back(computeModAxis(coords[2], k_size));

    } else if (mmaType.getOperand() == "COp") {

      // A i: (lane % 16)
      // A k: floor(lane / 16)
      // A block: 0
      // B j: (lane % 16)
      // B k: floor(lane / 16)
      // B block: 0
      // C or D i: 4 * floor(lane / 16) + (GPR_num % 4)
      // C or D j: (lane % 16)
      // C or D block: 0

      // C = MxN = 16x36
      // -- Tile = MxN = 16x16
      //  lane =    0    1 ..   15    16   17 ..   31    32   33 ..   47    48   49 ..   63
      // -- R0 =  0,0  0,1 .. 0,15 | 1,0  1,1 .. 1,15 | 2,0  2,1 .. 2,15 | 3,0  3,1 .. 3,15
      // -- R1 =  4,0  4,1 .. 4,15 | 5,0  5,1 .. 5,15 | 6,0  6,1 .. 6,15 | 7,0  7,1 .. 7,15
      // -- R2 =  8,0  8,1 .. 8,15 | 9,0  9,1 .. 9,15 |10,0 10,1 ..10,15 |11,0 11,1 ..11,15
      // -- R3 = 12,0 12,1 ..12,15 |13,0 13,1 ..13,15 |14,0 14,1 ..14,15 |15,0 15,1 ..15,15

      // coords = [y, x]

      // m = y + stride * (lane / m_size) + (gpr * stride)
      // start +        inter_reg         +    gpr_offset
      int64_t m_size = mmaType.getShape()[0];
      newCoords.push_back(computeStridedAxis(coords[0], m_size));
      
      // n = x + (lane % m_size)
      newCoords.push_back(computeModAxis(coords[1], m_size));
    }

    return newCoords;
  }

};

AMDMMAInfo(gpu::SubgroupMmaLoadMatrixOp) -> AMDMMAInfo<gpu::SubgroupMmaLoadMatrixOp>;
AMDMMAInfo(gpu::SubgroupMmaConstantMatrixOp) -> AMDMMAInfo<gpu::SubgroupMmaConstantMatrixOp>;
AMDMMAInfo(gpu::SubgroupMmaComputeOp) -> AMDMMAInfo<gpu::SubgroupMmaComputeOp>;

AMDMMAInfo(gpu::MMAMatrixType) -> AMDMMAInfo<gpu::SubgroupMmaLoadMatrixOp>;

/// This class implements the conversion of GPU MMA loadOp to wmma.load op
/// in the AMDGPU dialect. The conversion not only emits the AMDGPU op but also
/// emits code that is necessary to store the data in the destination memref
/// after it has been loaded.
struct WmmaLoadOpToAMDGPULowering
    : public OpConversionPattern<gpu::SubgroupMmaLoadMatrixOp> {
  using OpConversionPattern<gpu::SubgroupMmaLoadMatrixOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaLoadMatrixOp subgroupMmaLoadMatrixOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // %34 = gpu.subgroup_mma_load_matrix %alloc_2[%25, %33, %c0] {leadDimension = 20 : index} : memref<4x32x20xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">

    auto op = subgroupMmaLoadMatrixOp;
    Location loc = op->getLoc();
    
    AMDMMAInfo mmaInfo(op);
    
    auto regType = mmaInfo.getRegType();
    Value regs = rewriter.create<memref::AllocOp>(loc, regType);
      
    Value laneId = rewriter.create<gpu::LaneIdOp>(loc);

    // This is the individual element type.
    Type loadedElType = regType.getElementType();

    bool isTransposeLoad = op.getTranspose().value_or(false);

    // If we are not transposing, then we can use vectorized loads. Otherwise, we
    // must load each element individually.
    if (!isTransposeLoad) {
      // if (!isa<MemRefType>(loadedElType)) {
      //   auto private_as = gpu::AddressSpaceAttr::get(rewriter.getContext(),
      //                     gpu::GPUDialect::getPrivateAddressSpace());
      //   MemRefLayoutAttrInterface layout;
      //   loadedElType = MemRefType::get({1}, loadedElType, layout, private_as);
      // }

      for (int i = 0; i < regType.getNumElements(); i++) {
        auto newIndices = mmaInfo.computeIndices(rewriter, laneId, i);

        Value el = rewriter.create<memref::LoadOp>(loc, loadedElType,
                                                   op.getSrcMemref(), newIndices);
        Value innerVal = rewriter.createOrFold<arith::ConstantIndexOp>(loc, i);
        rewriter.create<memref::StoreOp>(loc, el, regs, ValueRange{innerVal});
      }
    } else {
      assert(0);
    }
      
    rewriter.replaceOp(op, regs);

    return success();
  }
};

/// This class implements the conversion of GPU MMA storeOp to wmma.store op
/// in the AMDGPU dialect. The conversion not only emits the AMDGPU op but also
/// emits code that is necessary to unpack the data in the source and
/// convert the data in the format that is needed by the AMDGPU op.
struct WmmaStoreOpToAMDGPULowering
    : public OpConversionPattern<gpu::SubgroupMmaStoreMatrixOp> {
  using OpConversionPattern<gpu::SubgroupMmaStoreMatrixOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaStoreMatrixOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // gpu.subgroup_mma_store_matrix %120, %alloc[%7, %46] {leadDimension = 36 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<32x36xf32, #gpu.address_space<workgroup>>
    Location loc = op->getLoc();

    AMDMMAInfo mmaInfo(op);

    auto regType = mmaInfo.getRegType();

    Value laneId = rewriter.create<gpu::LaneIdOp>(loc);

    // This is the individual element type.
    Type loadedElType = regType.getElementType();

    bool isTransposeStore = op.getTranspose().value_or(false);
    assert(!isTransposeStore);

    // If we are not transposing, then we can use vectorized loads. Otherwise, we
    // must load each element individually.
    if (!isTransposeStore) {
      // if (!isa<MemRefType>(loadedElType)) {
      //   auto private_as = gpu::AddressSpaceAttr::get(rewriter.getContext(),
      //                     gpu::GPUDialect::getPrivateAddressSpace());
      //   MemRefLayoutAttrInterface layout;
      //   loadedElType = MemRefType::get({1}, loadedElType, layout, private_as);
      // }

      for (int i = 0; i < regType.getShape()[0]; i++) {
        Value innerVal = rewriter.createOrFold<arith::ConstantIndexOp>(loc, i);
        Value el = rewriter.create<memref::LoadOp>(loc, loadedElType, adaptor.getSrc(),
                                                   ValueRange{innerVal});
        auto newIndices = mmaInfo.computeIndices(rewriter, laneId, i);
        rewriter.create<memref::StoreOp>(loc, el,
                                         op.getDstMemref(), newIndices);
      }
    } else {
      assert(0);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

/// This class implements the conversion of GPU MMA computeOp to wmma.mma op
/// in the AMDGPU dialect.
struct WmmaMmaOpToAMDGPULowering
    : public OpConversionPattern<gpu::SubgroupMmaComputeOp> {
  using OpConversionPattern<gpu::SubgroupMmaComputeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaComputeOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // %119 = gpu.subgroup_mma_compute %115, %117, %114 : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">

    AMDMMAInfo mmaInfo(op);

    // Get the shapes of the MMAMatrix type being used. The shapes will
    // choose which intrinsic this op will be lowered to.
    auto aType = op.getOpA().getType();
    ArrayRef<int64_t> aTypeShape = aType.getShape();
    auto cType = op.getOpC().getType();
    ArrayRef<int64_t> cTypeShape = cType.getShape();
    // restricted to f32 in/out element types
    assert(aType.getElementType().isF32());
    assert(cType.getElementType().isF32());

    int64_t m = cTypeShape[0];
    int64_t n = cTypeShape[1];
    int64_t k = aTypeShape[1];

    int64_t blocks = 1;
    
    assert(m <= 32 && n <= 32);

    int64_t elemSize = cType.getElementType().getIntOrFloatBitWidth() / 8;

    int64_t bytes = m * n * k * elemSize;

    // bool aTranspose = subgroupMmaComputeOp.getATranspose();
    // bool bTranspose = subgroupMmaComputeOp.getBTranspose();
    // ROCDL::MMATypes sourceType = getElementType(aType);
    // ROCDL::MMATypes destType = getElementType(cType);
    // if (ROCDL::WMMAMmaOp::getIntrinsicID(m, n, k, aLayout, bLayout, sourceType,
    //                                     destType) == 0)
    //   return rewriter.notifyMatchFailure(op, kInvalidCaseStr);

    // ROCDL::MMATypes bElementType = getElementType(
    //     cast<gpu::MMAMatrixType>(subgroupMmaComputeOp.getOpB().getType()));
    // if (bElementType != sourceType)
    //   return rewriter.notifyMatchFailure(
    //       op, "WMMA compute op input matrix element types must match.");

    int64_t iters = bytes / (1024 * 4);

    auto vectorType = cast<MemRefType>(getTypeConverter()->convertType(op->getResultTypes()[0]));
    auto elType = vectorType.getElementType();
    Location loc = op->getLoc();

    int64_t subk = k / iters;
    assert(subk > 0 && (k % iters) == 0);

    Value zero = rewriter.createOrFold<arith::ConstantIndexOp>(loc, 0);
    
    Value mmaC = adaptor.getOpC();

    for (int64_t cnt = 0; cnt < iters; ++cnt) {
      Value idx = rewriter.createOrFold<arith::ConstantIndexOp>(loc, cnt);

      Value ldA = rewriter.create<memref::LoadOp>(loc, elType,
                                                  adaptor.getOpA(), idx);
      
      Value ldB = rewriter.create<memref::LoadOp>(loc, elType,
                                                  adaptor.getOpB(), idx);
      // vector.transfer_read
      auto vecType = VectorType::get(mmaInfo.getRegType().getShape(), elType);
      Value ldC = rewriter.create<vector::TransferReadOp>(loc, vecType, mmaC, ValueRange{zero});

      Value ldD = rewriter.create<amdgpu::MFMAOp>(loc, vecType, m, n, subk, blocks,
                                                  ldA, ldB, ldC);
      // cbsz, abid, blgp, /*reducePrecision=*/false, /*negateA=*/false,
      // /*negateB=*/false, /*negateC=*/false);
      // int32_t cbsz = 0;
      // int32_t abid = 0;
      //DefaultValuedAttr<I32Attr, "0">:$cbsz;
      //DefaultValuedAttr<I32Attr, "0">:$abid,
      //DefaultValuedAttr<AMDGPU_MFMAPermBAttr, "::mlir::amdgpu::MFMAPermB::none">:$blgp,
      // ::mlir::amdgpu::MFMAPermB blgp = ::mlir::amdgpu::MFMAPermB::none;
      //UnitAttr:reducePrecision,
      //UnitAttr:negateA,
      //UnitAttr:negateB,
      //UnitAttr:negateC)>,

      // vector.transfer_write
      rewriter.create<vector::TransferWriteOp>(loc, ldD, mmaC, ValueRange{zero});
      
    }

    rewriter.replaceOp(op, mmaC);
    
    return success();
  }
};

/// Convert GPU MMA ConstantMatrixOp to a chain of InsertValueOp.
struct WmmaConstantOpToAMDGPULowering
    : public OpConversionPattern<gpu::SubgroupMmaConstantMatrixOp> {
  using OpConversionPattern<gpu::SubgroupMmaConstantMatrixOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaConstantMatrixOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // if (failed(areAllLLVMTypes(subgroupMmaConstantOp.getOperation(),
    //                            adaptor.getOperands(), rewriter)))
    //   return failure();
    
    // %0 = gpu.subgroup_mma_constant_matrix %cst : !gpu.mma_matrix<16x16xf32, "COp">
    // > Make local registers
    // > Fill with %cst
    AMDMMAInfo mmaInfo(op);
    auto regType = mmaInfo.getRegType();
    
    // Value cst = adaptor.getOperands()[0];
    Value cBuffer = rewriter.replaceOpWithNewOp<memref::AllocOp>(op, regType);
    (void)cBuffer;

    // Initialize to zero
    
    // rewriter.replaceOpWithNewOp<arith::ConstantOp>(
    //     subgroupMmaConstantOp, ntype, rewriter.getZeroAttr(ntype.getElementType()));
    //rewriter.replaceOp(subgroupMmaConstantOp, cBuffer);
    
    return success();
  }
};

#if 0
static Value createMinMaxF(OpBuilder &builder, Location loc, Value lhs,
                           Value rhs, bool isMin) {
  auto floatType = cast<FloatType>(getElementTypeOrSelf(lhs.getType()));
  Type i1Type = builder.getI1Type();
  if (auto vecType = dyn_cast<VectorType>(lhs.getType()))
    i1Type = VectorType::get(vecType.getShape(), i1Type);
  Value cmp = builder.create<LLVM::FCmpOp>(
      loc, i1Type, isMin ? LLVM::FCmpPredicate::olt : LLVM::FCmpPredicate::ogt,
      lhs, rhs);
  Value sel = builder.create<LLVM::SelectOp>(loc, cmp, lhs, rhs);
  Value isNan = builder.create<LLVM::FCmpOp>(
      loc, i1Type, LLVM::FCmpPredicate::uno, lhs, rhs);
  Value nan = builder.create<LLVM::ConstantOp>(
      loc, lhs.getType(),
      builder.getFloatAttr(floatType,
                           APFloat::getQNaN(floatType.getFloatSemantics())));
  return builder.create<LLVM::SelectOp>(loc, isNan, nan, sel);
}

static Value createScalarOp(OpBuilder &builder, Location loc,
                            gpu::MMAElementwiseOp op,
                            ArrayRef<Value> operands) {
  switch (op) {
  case gpu::MMAElementwiseOp::ADDF:
    return builder.create<LLVM::FAddOp>(loc, operands[0].getType(), operands);
  case gpu::MMAElementwiseOp::MULF:
    return builder.create<LLVM::FMulOp>(loc, operands[0].getType(), operands);
  case gpu::MMAElementwiseOp::DIVF:
    return builder.create<LLVM::FDivOp>(loc, operands[0].getType(), operands);
  case gpu::MMAElementwiseOp::MAXF:
    return createMinMaxF(builder, loc, operands[0], operands[1],
                         /*isMin=*/false);
  case gpu::MMAElementwiseOp::MINF:
    return createMinMaxF(builder, loc, operands[0], operands[1],
                         /*isMin=*/true);
  default:
    llvm_unreachable("unknown op");
  }
}

/// Convert GPU MMA elementwise ops to extract + op + insert.
struct WmmaElementwiseOpToAMDGPULowering final
  : public OpConversionPattern<gpu::SubgroupMmaElementwiseOp> {
  using OpConversionPattern<gpu::SubgroupMmaElementwiseOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(gpu::SubgroupMmaElementwiseOp subgroupMmaElementwiseOp,
                  PatternRewriter &rewriter) const override {
    if (failed(areAllLLVMTypes(subgroupMmaElementwiseOp.getOperation(),
                               adaptor.getOperands(), rewriter)))
      return failure();
    Location loc = subgroupMmaElementwiseOp.getLoc();
    size_t numOperands = adaptor.getOperands().size();
    LLVM::LLVMStructType destType = convertMMAToLLVMType(
        cast<gpu::MMAMatrixType>(subgroupMmaElementwiseOp.getType()));
    Value matrixStruct = rewriter.create<LLVM::UndefOp>(loc, destType);
    for (size_t i = 0, e = destType.getBody().size(); i < e; ++i) {
      SmallVector<Value> extractedOperands;
      for (size_t opIdx = 0; opIdx < numOperands; opIdx++) {
        extractedOperands.push_back(rewriter.create<LLVM::ExtractValueOp>(
            loc, adaptor.getOperands()[opIdx], i));
      }
      Value element =
          createScalarOp(rewriter, loc, subgroupMmaElementwiseOp.getOpType(),
                         extractedOperands);
      matrixStruct =
          rewriter.create<LLVM::InsertValueOp>(loc, matrixStruct, element, i);
    }
    rewriter.replaceOp(subgroupMmaElementwiseOp, matrixStruct);
    return success();
  }
};
#endif


//===----------------------------------------------------------------------===//
// scf::ForOp
//===----------------------------------------------------------------------===//
/// Pattern to convert a scf::ForOp within kernel functions into spirv::LoopOp.
struct ForOpConversion final : public ConversionPattern {
  ForOpConversion(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, "scf.for", 1, context), converter(converter) {}
  TypeConverter &converter;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    llvm::SmallVector<NamedAttribute, 4> newAttrs;
    newAttrs.reserve(op->getAttrs().size());
    for (auto attr : op->getAttrs()) {
      if (auto typeAttr = attr.getValue().dyn_cast<TypeAttr>()) {
        auto newAttr = getTypeConverter()->convertType(typeAttr.getValue());
        newAttrs.emplace_back(attr.getName(), TypeAttr::get(newAttr));
      } else {
        newAttrs.push_back(attr);
      }   
    }

    llvm::SmallVector<Type, 4> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());

    for (Region &region : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.cloneRegionBefore(region, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());

    return success();
  }

};

struct YieldOpConversion final
  : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    IRMapping map;
    for (auto pair : llvm::zip(op.getOperands(), adaptor.getOperands()))
      map.map(std::get<0>(pair), std::get<1>(pair));
    
    rewriter.clone(*op, map);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

/// Flattens n-D MemRef to 1-D MemRef and allows other types.
struct GPUMatrixTypeConverter final : public TypeConverter {
  GPUMatrixTypeConverter() {
    // Allow all other types.
    addConversion([](Type type) -> std::optional<Type> { return type; });

    // Convert n-D MemRef to 1-D MemRef.
    addConversion([](gpu::MMAMatrixType type) -> std::optional<Type> {
        AMDMMAInfo mmaInfo(type);
        Type t = mmaInfo.getRegType();
        assert(t);
        return t;
      });
  }
};

struct ConvertGPUToAMDGPUPass
  : public iree_compiler::ConvertGPUToAMDGPUPassBase<ConvertGPUToAMDGPUPass> {
  ConvertGPUToAMDGPUPass() {}
  //ConvertGPUToAMDGPUPass(const FlattenMemRefSubspanPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amdgpu::AMDGPUDialect, affine::AffineDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    GPUMatrixTypeConverter converter;
    MLIRContext *context = &getContext();

    // This pass currently doesn't support alignment hints so remove them first.
    RewritePatternSet patterns(context);
    patterns.add<WmmaLoadOpToAMDGPULowering, WmmaMmaOpToAMDGPULowering,
                 WmmaStoreOpToAMDGPULowering, WmmaConstantOpToAMDGPULowering,
                 ForOpConversion, YieldOpConversion>(converter, context);

    ConversionTarget target(getContext());
    target.addLegalDialect<func::FuncDialect, arith::ArithDialect, vector::VectorDialect,
                           scf::SCFDialect,
                           iree_compiler::IREE::HAL::HALDialect,
                           ::mlir::amdgpu::AMDGPUDialect, gpu::GPUDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<gpu::SubgroupMmaElementwiseOp, gpu::SubgroupMmaLoadMatrixOp,
                        gpu::SubgroupMmaStoreMatrixOp, gpu::SubgroupMmaComputeOp,
                        gpu::SubgroupMmaConstantMatrixOp>();
    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      return converter.isLegal(op.getRegion().front().getArgumentTypes());
    });
    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
        return converter.isLegal(op.getOperandTypes());
    });
    
    if (failed(applyFullConversion(getOperation(), target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> iree_compiler::createConvertGPUToAMDGPUPass() {
  return std::make_unique<ConvertGPUToAMDGPUPass>();
}

