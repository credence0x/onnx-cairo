use orion::numbers::fixed_point::core::FixedType;
use orion::operators::tensor::implementations::impl_tensor_fp::Tensor_fp;
use orion::operators::tensor::core::Tensor;
use orion::operators::tensor::math::compress::helpers::_compress;

/// Cf: TensorTrait::compress docstring
fn compress(
    self: @Tensor<FixedType>,
    condition: Span<bool>,
    axis: Option<usize>
) -> Tensor<FixedType> {
    _compress(self, condition, axis)
}
