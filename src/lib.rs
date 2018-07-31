extern crate rand;
extern crate tensor_transforms;

pub mod wfc_algorithm;
use std::cmp::{max, PartialEq};
use std::marker::Sized;

use symmetry::{Transform, Transformable};
use wfc_algorithm::CompatibleAlign;

// a type that can be used by a wave
#[derive(Clone)]
pub struct WaveValue<T: PartialEq + Sized> {
    // number of dimensions
    n: usize,
    // the size of each dimension
    size: Vec<usize>,
    // values
    vals: Vec<T>,
}

impl<T: PartialEq + Sized> WaveValue<T> {
    pub fn new(
        size: &[usize],
        vals: Vec<T>,
    ) -> Result<WaveValue<T>, &'static str> {
        let mut product = 1;
        for i in size {
            product *= i
        }
        if Vec::len(&vals) != product {
            return Err("Vector does not have enough elements.");
        }
        Ok(WaveValue {
            n: size.len(),
            size: size.to_vec(),
            vals: vals,
        })
    }
}

impl<T: PartialEq + Sized + Copy> Transformable<WaveValue<T>> for WaveValue<T> {
    fn transform(&self, transform: &Transform) -> Self {
        let n = transform.dimensions();
        let max = max(n, self.n);
        let axes = transform.values();
        let mut new_size = vec![1; max];
        for i in 0..self.n {
            let sindex = if i < n { axes[i].dim } else { i };
            new_size[sindex] = self.size[i];
        }
        let mut new_page_sizes = Vec::with_capacity(max + 1);
        let mut product = 1;
        new_page_sizes.push(product);
        for i in 0..max {
            product *= new_size[i];
            new_page_sizes.push(product);
        }
        let mut product = 1;
        let mut old_page_sizes = Vec::with_capacity(self.n + 1);
        old_page_sizes.push(product);
        for i in 0..max {
            product *= self.size[i];
            old_page_sizes.push(product);
        }

        let len = Vec::len(&self.vals);
        let mut new_vals = vec![self.vals[0]; len];
        let mut loop_indices = vec![0; self.n];
        let mut offset = vec![0; self.n];
        let mut transform_coordinate = 0;
        for i in 0..self.n {
            let (dim, dir) = if i < n {
                (axes[i].dim, axes[i].sign)
            } else {
                (i, 1)
            };
            offset[i] = new_page_sizes[dim] as isize * dir as isize;
            // start at the opposite end for negative indices
            if dir == -1 {
                transform_coordinate += (new_page_sizes[dim] * (self.size[i] - 1)) as isize;
            }
        }

        let mut i = 0;
        let mut old_index = 0;
        while i < self.n {
            if loop_indices[i] < self.size[i] {
                if i == 0 {
                    new_vals[transform_coordinate as usize] = self.vals[old_index];
                    old_index += 1;
                }
                loop_indices[i] += 1;

                transform_coordinate += offset[i];
                if loop_indices[i] < self.size[i] {
                    i = 0
                };
            } else {
                transform_coordinate -= offset[i] * self.size[i] as isize;
                loop_indices[i] = 0;
                i += 1;
            }
        }
        WaveValue {
            n: max,
            size: new_size,
            vals: new_vals,
        }
    }
}

impl<T: PartialEq + Sized> CompatibleAlign<WaveValue<T>> for WaveValue<T> {
    fn dimensions(&self) -> usize {
        self.n
    }

    fn align(&self, other: &WaveValue<T>, direction: usize) -> bool {
        if self.n == other.n {
            for (s1, s2) in self.size.iter().zip(other.size.iter()) {
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
        for i in 0..dim {
            page_size *= self.size[i];
        }
        let next_size = page_size * self.size[dim];
        let page_size = page_size as isize * dir;
        for i in 0..Vec::len(&self.vals) {
            let pos = i as isize + page_size;
            // when pos is negative, casting to usize causes it to move outside the range
            if i / next_size == pos as usize / next_size {
                if self.vals[pos as usize] != other.vals[i] {
                    return false;
                }
            }
        }
        true
    }
}

// a testable value that acts as a discrete tile instead of as an overlapping area
pub struct TileValue<T> {
    n: usize,
    vals: Vec<T>,
}

impl<U> TileValue<U> {
    pub fn new<T>(vals: Vec<T>) -> Result<TileValue<T>, &'static str> {
        if (&vals).len() % 2 != 0 {
            return Err("Length must be a multiple of 2.");
        }
        Ok(TileValue {
            n: (&vals).len() / 2,
            vals: vals,
        })
    }
}

impl<T: PartialEq + Sized> CompatibleAlign<TileValue<T>> for TileValue<T> {
    fn dimensions(&self) -> usize {
        self.n
    }

    fn align(&self, other: &TileValue<T>, direction: usize) -> bool {
        let reverse_direction = direction ^ 1;
        self.vals[direction] == other.vals[reverse_direction]
    }
}

impl<T: PartialEq + Sized + Copy> Transformable<TileValue<T>> for TileValue<T> {
    fn transform(&self, transform: &Transform) -> TileValue<T> {
        let n = transform.dimensions();
        if n > self.n {
            panic!("Cannot transform a tile into a higher dimension")
        };
        let axes = transform.values();
        let mut new_vals = vec![self.vals[0]; 2 * self.n];
        for i in 0..self.n {
            let (dim, dir) = if i < n {
                (2 * axes[i].dim, ((axes[i].sign + 1) / 2) as usize)
            } else {
                (2 * i, 0)
            };
            new_vals[dim + dir] = self.vals[dim];
            new_vals[dim + 1 - dir] = self.vals[dim + 1];
        }
        TileValue {
            n: self.n,
            vals: new_vals,
        }
    }
}

use fmt::{Display, Formatter};
use std::fmt;

impl<T: PartialEq + Sized + Display> Display for WaveValue<T> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.n == 1 {
            writeline(&self.vals[..], f)?;
        } else if self.n == 2 {
            writelines(&self.vals[..], self.size[0], f)?;
        } else {
            let page_size = self.size[0] * self.size[1];
            let line_size = self.size[0];
            let mut product = 1;
            for i in 2..self.n {
                product *= self.size[i];
            }
            for i in 0..product {
                writelines(&self.vals[page_size * i..page_size * (i + 1)], line_size, f)?;
                if i < product - 1 {
                    write!(f, "\n\n")?
                };
            }
        }
        Ok(())
    }
}

fn writelines<T: Display>(vals: &[T], line_length: usize, f: &mut Formatter) -> fmt::Result {
    let num_lines = vals.len() / line_length;
    for i in 0..num_lines {
        writeline(&vals[i * line_length..(i + 1) * line_length], f)?;
        if i < num_lines - 1 {
            write!(f, "\n")?
        };
    }
    Ok(())
}

fn writeline<T: Display>(vals: &[T], f: &mut Formatter) -> fmt::Result {
    for val in &vals[..vals.len() - 1] {
        write!(f, "{}, ", val)?;
    }
    write!(f, "{}", vals[vals.len() - 1])?;
    Ok(())
}
