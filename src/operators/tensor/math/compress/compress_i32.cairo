use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::compress::helpers::_compress;

/// Cf: TensorTrait::compress docstring
fn compress(
    self: @Tensor<i32>,
    condition: Span<bool>,
    axis: Option<usize>
) -> Tensor<i32> {
    _compress(self, condition, axis)
}
