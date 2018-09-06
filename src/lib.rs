#[cfg(feature = "image")]
extern crate image;

extern crate rand;
extern crate tensor_transforms;

#[cfg(feature = "image")]
pub mod image_impls;
pub mod wfc_algorithm;

use std::cmp::PartialEq;
use std::marker::Sized;
pub use wfc_algorithm::{ProtoGraph, Wave};

use tensor_transforms::transformable_objects::{DirectionObject, TransformTensor};
use tensor_transforms::{Transform, TransformDimension, Transformable};

/// A type that can be compared to objects of another type (usually itself).
///
/// The direction of the alignment is based on a direction parameter, which
/// which is a `usize` that iterates over all directions.
/// To instead use the directions in each dimension, implement AlignPos<T>
pub trait CompatibleAlign<T>: TransformDimension {
    /// Check the second object to see if it is compatible with this object when moved by `direction`.
    /// `direction` is a usize that represents an axis and a sign
    /// The axis is equal to `direction/2`,
    /// and the sign is -1 if `direction` is even and 1 if `direction` is odd.
    fn align(&self, other: &T, direction: usize) -> bool;
}

/// The sole purpose of this trait is to make implementing CompatibleAlign
/// easier. CompatibleAlign uses a convention for numeric directions that is
/// widely used internally within this library, but can be confusing.
pub trait AlignPos<T>: TransformDimension {
    /// See CompatibleAlign<T>. In this case, pos will
    /// explicitly state the amount to be moved in each direction.
    /// This can be wasteful since it requires allocating a vec, though
    /// the compiler can often optimize it. This function should not be expected
    /// to return a meaningful value when pos contains anything other than an array
    /// of 0s with one 1 or -1.
    fn align(&self, other: &T, pos: &[isize]) -> bool;
}

impl<T, U: AlignPos<T>> CompatibleAlign<T> for U {
    fn align(&self, other: &T, direction: usize) -> bool {
        let mut offset = vec![0; self.dimensions()];
        offset[direction / 2] = ((direction % 2) * 2) as isize - 1;
        self.align(other, &offset[..])
    }
}

/// This trait can be used to compare rectangular (or hyper rectangular) slices of an n-dimensional array.
///
/// The function `compare_values` takes two multidimensional array indices and outputs an f64
/// that represents the difference between them, and the function `get_size` takes a dimension and
/// outputs the size of the array in that dimension. This second function should return 0 for the first
/// dimension that is not a dimension in the array. For example, a 2 dimensional array with size (15, 20)
/// should return 15, 20, and then 0. The second function should generally assumed to be relatively cheap.
pub trait AlignArray {
    /// Gets the size of the array in a certain dimension. This is generally expected to
    /// be a cheap operation.
    fn get_size(&self, dim: usize) -> usize;

    /// Compares two values in the array and returns some difference value. This difference value
    /// will be compared to a threshold to see if the comparison of larger chunks succeeds. There
    /// are two main ways to implment this. Either this function checks for equality and returns 0
    /// in cases where the values are equal and 1 when they are not, or it returns the square difference
    /// between two values.
    fn compare_values(&self, &[usize], &[usize]) -> f64;
}

impl<T: PartialEq + Sized> CompatibleAlign<TransformTensor<T>> for TransformTensor<T> {
    fn align(&self, other: &TransformTensor<T>, direction: usize) -> bool {
        if self.dimensions() == other.dimensions() {
            for (s1, s2) in self.get_size().iter().zip(other.get_size().iter()) {
                if s1 != s2 {
                    panic!("Compared values must be the same size (for now)")
                };
            }
        } else {
            panic!("Compared values must be the same dimension.");
        }
        let dim = direction / 2;
        let dir = (direction as isize % 2) * 2 - 1;
        let mut page_size = 1;
        let size = self.get_size();
        let vals = self.get_vals();
        let other_vals = other.get_vals();
        for i in 0..dim {
            page_size *= size[i];
        }
        let next_size = page_size * size[dim];
        let page_size = page_size as isize * dir;
        for i in 0..(&vals).len() {
            let pos = i as isize + page_size;
            // when pos is negative, casting to usize causes it to move outside the range
            if i / next_size == pos as usize / next_size {
                if vals[pos as usize] != other_vals[i] {
                    return false;
                }
            }
        }
        true
    }
}

impl<T: PartialEq + Sized> CompatibleAlign<DirectionObject<T>> for DirectionObject<T> {
    fn align(&self, other: &DirectionObject<T>, direction: usize) -> bool {
        let reverse_direction = direction ^ 1;
        self.get_vals()[direction] == other.get_vals()[reverse_direction]
    }
}
