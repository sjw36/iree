
!A_size = tensor<1x3x5xf32>
!B_size = tensor<1x5x3xf32>
!C_size = tensor<1x3x3xf32>

func.func @matmul_static(%A : !A_size, %B : !B_size) -> !C_size {
  %cst_0 = arith.constant 0.0 : f32
  %C = tensor.empty() : !C_size
  %C0 = linalg.fill ins(%cst_0 : f32) outs(%C : !C_size) -> !C_size
  %0 = linalg.batch_matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C0 : !C_size) -> !C_size
  return %0 : !C_size
}

// RUN: iree-compile %s --iree-hal-target-backends=aiunite --iree-opt-data-tiling=false --iree-codegen-use-transform-dialect-strategy=%p/matmul_codegen_default_spec.mlir

// EXEC: 3x3xf32=[52 52 52][52 52 52][52 52 52]
