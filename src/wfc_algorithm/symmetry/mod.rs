use std::cmp::PartialEq;
use std::ops::{Add, AddAssign, Neg, Sub, SubAssign};

/// An n dimensional object that can be used to determine a symmetry group.
/// This contains two comparable values per dimension, one in the negative direction and one positive.
/// The symmetry group generated will contain all transforms that leave this object equal to the original.
#[derive(Clone)]
pub struct SymmetryObject<T: PartialEq> {
    n: usize,
    colors: Vec<T>,
}

use std::marker::Sized;
use wfc_algorithm::CompatibleAlign;

/// Allows an object to be transformed
pub trait Transformable: Sized + Clone {
    /// Applies a given transform.
    /// Implementing this method can be tricky, and if it is done wrong
    /// it can result in other methods not working.
    /// Transforms can be applied to other transforms and follow these rules:
    /// * ascociativity: T1(T2(A)) = (T1(T2))(A)
    /// * invertability: T1(-T1(A)) = -T1(T1(A)) = A
    /// Note that this method also has to deal with transforms that do not have the same dimension as this object.
    fn transform(&self, transform: &Transform) -> Self;
    
    /// Returns the number of dimensions of this object.
    fn dimensions(&self) -> usize;

    /// Applies all the transforms described by the axisgroups.
    /// The number of transforms is equal to produce of the sizes of the axisgroups.
    /// If an axis group has n axes then its size will be:
    /// * n! if the group is directional
    /// * (2^n)(n!) if it is not directional
    /// This can huge numbers quickly when working with more than a few dimensions.
    fn get_transformed_values(&self, groups: Vec<AxisGroup>) -> Vec<Self> {
        // get the dimension
        let n = self.dimensions();
        // the first step is to create a set of transforms for each axis group
        let mut tset = Vec::with_capacity(Vec::len(&groups));
        let mut total_transforms = 1;
        for group in &groups {
            // get the indices of the axis in the transform
            let num_axes = Vec::len(&group.axes);
            let mut axes = Vec::with_capacity(num_axes);
            for &a in &group.axes {
                axes.push(if group.directional { a / 2 } else { a });
            }
            // now permute that set of axes
            // this uses heap's algorithm
            let mut permutations = Vec::with_capacity(factorial(num_axes));
            let mut c = vec![0; num_axes];

            let mut i = 0;
            // we only permute the axes for this group,
            // but generate transforms that contain all axes
            let mut first = Transform::identity(n);
            // this fixes the directionality
            if group.directional {
                for &a in &group.axes {
                    first.transform[a / 2].sign *= (a as i8 % 2) * 2 - 1;
                }
            }
            permutations.push(first);
            while i < num_axes {
                if c[i] < i {
                    let mut last = permutations[Vec::len(&permutations) - 1].clone();
                    if i % 2 == 0 {
                        (&mut last.transform[..]).swap(axes[0], axes[i]);
                    } else {
                        (&mut last.transform[..]).swap(axes[c[i]], axes[i]);
                    }
                    permutations.push(last);
                    c[i] += 1;
                    i = 0;
                } else {
                    c[i] = 0;
                    i += 1;
                }
            }
            // if the group is directional, it is now neccessary to undo the earlier flips
            // this applies to the original axis
            // so if a negative axis is swapped with a negative axis,
            // both will be a positive axis in the transform
            // but if a positive axis is swapped with a negative axis,
            // both will be negative
            if group.directional {
                for a in &group.axes {
                    for t in &mut permutations {
                        t.transform[a / 2].sign *= (*a as i8 % 2) * 2 - 1;
                    }
                }
            } else {
                // now each transformation is copied for all possible sign
                // we can use the bits of a usize for this operation,
                // since the number of transforms cannot be larger than usize anyways
                let num_permutations = Vec::len(&permutations);
                permutations.reserve(num_permutations * (1 << num_axes));
                // the first value will be left the same, so skip it
                for bits in 1..1 << num_axes {
                    for i in 0..num_permutations {
                        let t = permutations[i].clone();
                        permutations.push(t);
                    }
                    let window =
                        &mut permutations[num_permutations * bits..num_permutations * (bits + 1)];
                    for t in window.iter_mut() {
                        for j in 0..num_axes {
                            if (bits & 1 << j) != 0 {
                                t.transform[group.axes[j]].sign *= -1;
                            }
                        }
                    }
                }
            }
            total_transforms *= Vec::len(&permutations);
            tset.push(permutations);
        }
        // now just take the cartesian product of all these transformations
        let mut all_transforms = Vec::with_capacity(total_transforms);
        all_transforms.push(self.clone());
        for set in tset {
            let old_size = Vec::len(&all_transforms);
            // skip the first transform, which is always the identity
            for t in &set[1..] {
                for i in 0..old_size {
                    let transformed = all_transforms[i].transform(t);
                    all_transforms.push(transformed);
                }
            }
        }
        all_transforms
    }
    
    /// This method is used to skip the step of generating the axisgroups from a symmetryobject
    fn transform_from<U: PartialEq>(&self, sym: &SymmetryObject<U>) -> Vec<Self> {
        self.get_transformed_values(sym.get_transforms())
    }
}

fn factorial(n: usize) -> usize {
    let mut f = 1;
    for i in 2..n {
        f *= i;
    }
    f
}

impl<T: PartialEq> SymmetryObject<T> {
    /// Generates a symmetry object, checking the provided vec to make sure it has enough elements
    pub fn new(n: usize, mut colors: Vec<T>) -> Result<SymmetryObject<T>, &'static str> {
        if Vec::len(&colors) < 2 * n {
            return Err("Not enough colors.");
        }
        colors.truncate(2 * n);
        Ok(SymmetryObject {
            n: n,
            colors: colors,
        })
    }
    
    /// Sets a specific value to a new provided value
    pub fn set_val(&mut self, dim: usize, sign: i8, value: T) {
        self.colors[dim * 2 + ((sign + 1) / 2) as usize] = value;
    }

    /// Gets axisgroups, which represent the possible transformations
    pub fn get_transforms(&self) -> Vec<AxisGroup> {
        let mut axis_groups = Vec::<AxisGroup>::new();
        // used to keep track of the values for each group
        let mut group_directions = Vec::<Directional<T>>::new();
        for i in 0..self.n {
            let mut found = false;
            for (group, dir) in (&mut axis_groups).into_iter().zip(&group_directions) {
                let placed = match &dir {
                    Directional::Yes(a, b) => {
                        if self.colors[i * 2] == **a && self.colors[i * 2 + 1] == **b {
                            group.axes.push(i * 2);
                            true
                        } else if self.colors[i * 2 + 1] == **a && self.colors[i * 2] == **b {
                            group.axes.push(i * 2 + 1);
                            true
                        } else {
                            false
                        }
                    }
                    Directional::No(a) => {
                        if self.colors[i * 2] == **a && self.colors[i * 2 + 1] == **a {
                            group.axes.push(i);
                            true
                        } else {
                            false
                        }
                    }
                };
                if placed {
                    found = true;
                    break;
                }
            }
            if !found {
                let (dir, val, is_dir) = if self.colors[i * 2] == self.colors[i * 2 + 1] {
                    (Directional::No(&self.colors[i * 2]), i, false)
                } else {
                    (
                        Directional::Yes(&self.colors[i * 2], &self.colors[i * 2 + 1]),
                        i * 2,
                        true,
                    )
                };
                axis_groups.push(AxisGroup {
                    directional: is_dir,
                    axes: vec![val],
                });
                group_directions.push(dir);
            }
        }
        axis_groups
    }
}

impl<T: Copy + PartialEq> SymmetryObject<T> {
    /// Generates a new SymmetryObject filled with the default value.
    pub fn new_filled(n: usize, default: T) -> SymmetryObject<T> {
        SymmetryObject {
            n: n,
            colors: vec![default; 2 * n],
        }
    }
}

pub struct AxisGroup {
    directional: bool,
    axes: Vec<usize>,
}

use std::fmt;

impl fmt::Display for AxisGroup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s = String::from("[");
        for &num in &self.axes {
            let n = if self.directional {
                let num = num as isize;
                (num / 2 + 1) * ((num % 2) * 2 - 1)
            } else {
                (num + 1) as isize
            };
            s += &format!("{}, ", n);
        }
        let len = s.len();
        s.truncate(len - 2);
        s += "]";
        write!(f, "directional: {}; axes: {}", self.directional, s)
    }
}

enum Directional<'a, T: PartialEq + 'a> {
    Yes(&'a T, &'a T),
    No(&'a T),
}

/// Represents a transformation.
/// Mathematically speaking, this describes a signed permutation matrix,
/// which can be used to represent any rotation or transpose of an n dimensional tensor.
/// This object to checked to make sure it represents a valid transform.
/// Transforms implement `Add`, `Sub`, and `Neg` to compose themselves, in addition to implementing transformable.
#[derive(Clone)]
pub struct Transform {
    // number of dimensions
    n: usize,
    // the transform
    transform: Vec<TransformVector>,
}

impl Transform {
    /// Gets the number of dimensions
    pub fn dimensions(&self) -> usize {
        self.n
    }
    
    /// Gets a slice containing the values in this transform
    pub fn values(&self) -> &[TransformVector] {
        &self.transform[..]
    }
    
    /// Generates an n dimensional identity transform
    pub fn identity(n: usize) -> Transform {
        let mut vec = Vec::with_capacity(n);
        for i in 0..n {
            vec.push({ TransformVector { dim: i, sign: 1 } })
        }
        Transform {
            n: n,
            transform: vec,
        }
    }
}

/// Represents a unit vector along a cardinal direction.
#[derive(Clone, Copy)]
pub struct TransformVector {
    // the dimension
    pub dim: usize,
    // either -1 or 1
    pub sign: i8,
}

impl Default for TransformVector {
    fn default() -> TransformVector {
        TransformVector { dim: 0, sign: 1 }
    }
}

impl TransformVector {
    pub fn from_direction(dir: usize) -> TransformVector {
        TransformVector {
            dim: dir / 2,
            sign: ((dir % 2) * 2) as i8 - 1,
        }
    }

    pub fn to_direction(self) -> usize {
        self.dim * 2 + ((self.sign + 1) / 2) as usize
    }
}

impl Transformable for Transform {
    fn dimensions(&self) -> usize {
        self.n
    }
    
    fn transform(&self, transform: Transform) -> Transform {
        self + transform
    }
}

// applies the other transformation to self
// does not require both transformations to be the same dimension
impl Add<Transform> for Transform {
    type Output = Transform;

    fn add(mut self, other: Transform) -> Transform {
        self += other;
        self
    }
}

impl AddAssign for Transform {
    fn add_assign(&mut self, other: Transform) {
        let mut i = 0;
        while i < self.n {
            let vec = self.transform[i];
            // otherwise the source transformation is not high enough dimensional,
            // to affect this direction
            if other.n > vec.dim {
                let v2 = other.transform[vec.dim];
                self.transform[i] = TransformVector {
                    dim: v2.dim,
                    sign: vec.sign * v2.sign,
                };
                // extend the vector and increase the dimension of the output
                while v2.dim + 1 >= self.n {
                    self.transform.push(TransformVector {
                        dim: self.n,
                        sign: 1,
                    });
                    self.n += 1;
                }
            }
            i += 1;
        }
    }
}

// Computes the inverse of a transform
impl Neg for Transform {
    type Output = Transform;

    fn neg(mut self) -> Transform {
        let mut inv_transforms = vec![TransformVector { dim: 0, sign: 1 }; self.n];
        // in a well formed Transform, this will fully populate the new vector
        for i in 0..self.n {
            let t = self.transform[i];
            inv_transforms[t.dim] = TransformVector {
                dim: i,
                // 1/1 = 1 and 1/-1 = -1, so inverting can be skipped
                sign: t.sign,
            };
        }
        self.transform = inv_transforms;
        self
    }
}

impl SubAssign for Transform {
    fn sub_assign(&mut self, other: Transform) {
        // use a combination of neg and add_assign to preform subtraction
        *self += -other;
    }
}

impl Sub<Transform> for Transform {
    type Output = Transform;

    fn sub(mut self, other: Transform) -> Transform {
        self -= other;
        self
    }
}
