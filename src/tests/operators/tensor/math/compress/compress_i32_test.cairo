
// ===== 3D ===== //

#[cfg(test)]
mod tensor_3D {
    use array::{ArrayTrait, SpanTrait};
    use core::traits::Into;

    use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
    use orion::operators::tensor::core::{TensorTrait, ExtraParams};
    use orion::tests::helpers::tensor::i32::i32_tensor_2x2x2_helper;
    use core::debug::PrintTrait;



    #[test]
    #[available_gas(20000000)]
    fn axis_0() {
        let tensor = i32_tensor_2x2x2_helper();
        // [[0, 1],
        //  [2, 3]
        //
        //  [4, 5],
        //  [6, 7]]
        
        let mut condition = ArrayTrait::new();
        condition.append(true);
        condition.append(true);
        let result = tensor.compress(condition.span(), Option::Some(1));
        assert((*result.shape[0]).into() == 1, 'result[0] = 1');
        assert((*result.shape[1]).into() == 8, 'result[1] = 8');
    }


}
