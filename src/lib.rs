extern crate rand;
extern crate tensor_transforms;

pub mod wfc_algorithm;
use std::cmp::{max, PartialEq};
use std::marker::Sized;

use tensor_transforms::{Transform, Transformable, TransformDimension};
use tensor_transforms::transformable_objects::{DirectionObject, TransformTensor};

/// A type that can be compared to objects of another type.
/// In almost all cases, the second type will be Self
/// The direction of the alignment is based on a direction parameter, which
/// which is a usize that iterates over all directions.
/// To instead use the directions in each dimension, implement AlignPos<T>
pub trait CompatibleAlign<T>: TransformDimension {
    /// Check the second object to see if it is compatible with this object when moved by `direction`.
    /// `direction` is a usize that represents an axis and a sign
    /// The axis is equal to `direction/2`,
    /// and the sign is -1 if `direction` is even and 1 if `direction` is odd
    fn align(&self, other: &T, direction: usize) -> bool;
}

/// See CompatibleAlign<T>
pub trait AlignPos<T>: TransformDimension {
    /// See CompatibleAlign<T>. In this case, pos will 
    /// explicitly state the amount to be moved in each direction.
    fn align(&self, other: &T, pos: &[usize]) -> bool;
}

impl<T> CompatibleAlign<T> for AlignPos<T> {
    fn align(&self, other: &T, direction: usize) -> bool {
        let mut offset = vec![0; self.dimensions()];
        offset[direction / 2] = (direction % 2) * 2 - 1;
        self.align(other, &offset[..])
    }
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
