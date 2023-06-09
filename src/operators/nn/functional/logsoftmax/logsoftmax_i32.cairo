use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::{impl_tensor_i32, impl_tensor_fp};

use orion::numbers::fixed_point::core::FixedType;

/// Cf: NNTrait::logsoftmax docstring
fn logsoftmax_i32(z: @Tensor<i32>, axis: usize) -> Tensor<FixedType> {
    let exp_tensor = z.exp();
    let sum = exp_tensor.reduce_sum(axis, true);
    let softmax = exp_tensor / sum;
    let logsoftmax = softmax.ln();

    return logsoftmax;
}