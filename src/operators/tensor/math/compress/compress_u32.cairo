use orion::operators::tensor::implementations::impl_tensor_u32::Tensor_u32;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::compress::helpers::_compress;

/// Cf: TensorTrait::compress docstring
fn compress(
    self: @Tensor<u32>,
    condition: Span<bool>,
    axis: Option<usize>
) -> Tensor<u32> {
    _compress(self, condition, axis)
}
