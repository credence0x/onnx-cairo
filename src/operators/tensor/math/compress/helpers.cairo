use array::ArrayTrait;
use array::SpanTrait;
use orion::operators::tensor::helpers::replace_index;
use orion::operators::tensor::core::{Tensor, TensorTrait, ravel_index, unravel_index};
use orion::operators::tensor::helpers::{reduce_output_shape, len_from_shape, combine_indices};
use orion::utils::check_gas;
use debug::PrintTrait;

/// Cf: TensorTrait::compress docstring
fn _compress<T, 
    impl TTensorTrait: TensorTrait<T>,
    impl TDrop: Drop<T>,
    impl TCopy: Copy<T>
>(
    self: @Tensor<T>, 
    condition: Span<bool>,
    axis: Option<usize>
 ) -> Tensor<T> {

    let axis = match axis {
        Option::Some(axis) => axis,
        Option::None(_) => 9000
    };

    assert(axis < (*self.shape).len(), 'axis out of dimensions');
    assert(condition.len() <= *(*self.shape)[axis], 'invalid condition length');
    
    // if axis >= 0 {

    // } 

    let mut output_data = ArrayTrait::<T>::new();
    let output_shape = reduce_output_shape(*self.shape, axis, false);
    let output_data_len = len_from_shape(output_shape);
    let mut axis_index = 0;


    let mut index: usize = 0;
    loop {
        check_gas();

        let output_indices = unravel_index(index, output_shape);
        let input_indices = combine_indices(output_indices, axis_index, axis);
        let input_index = ravel_index(*self.shape, input_indices);
        input_index.print();
        output_data.append(*(*self.data)[input_index]);

        index += 1;
        if index == output_data_len {
            break ();
        };
    };

    456.print();
    
    
    TensorTrait::<T>::new(output_shape, output_data.span(), *self.extra)

}


// np.array(
//    [[[0, -6, -5],
//      [7,  4, 9],
//      [-9, 1, 2],
//      [12, 5, 1]],  
     
//     [[0, -6, -5],
//      [7,  4, 9],
//      [-9, 1, 2]],
//      [12, 5, 1]],
// )

// shape = (2, 4, 3)

// np.compress([True, False, True], z, axis=1) given shape (2, 4, 3) and axis=1
// array([
//         [[ 0, -6, -5],
//         [-9,  1,  2]],

//        [[ 0, -6, -5],
//         [-9,  1,  2]]
//    ])


// 000,  001,  002,
// 010,  011,  012, 
// 020,  021,  022, 

// 100,  101,  102, 
// 110,  111,  112, 
// 120,  121,  122