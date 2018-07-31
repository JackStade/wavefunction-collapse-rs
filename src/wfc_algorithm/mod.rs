use rand::prelude::random;
use std::f64;
use {CompatibleAlign, Transformable, Transform};
use tensor_transforms::SymmetryObject;

/// The basic struct for performing wfc.
/// The wfc algorithm is essentially a method for solving a constrained graph.
/// Each node has a number of possible values, which are constrained by all adjacent nodes.
/// The algorithm works by first setting a value with minium nonzero shannon entropy to a concrete value,
/// and then propogating that information and repeating until all nodes have 0 entropy.
pub struct Wave<'a> {
    // number of dimensions
    n: usize,
    // vector containing the maximum size in each dimension
    // note that things start breaking if any size is 1
    size: Vec<usize>,
    // whether or not to wrap in each dimension
    wrap: Vec<bool>,
    // this stores the values that a given value is compatible with
    // the first index is the value,
    // the second index is the direction
    propogators: &'a Vec<Vec<Vec<usize>>>,
    // the wave values for this graph
    nodes: Vec<WaveItem<'a>>,
    // the boolean values in each waveitem
    // this is used for storage, and should not be read from or written to
    item_values: Vec<Vec<bool>>,
    // the compatibility array
    // first index is the node
    // second index is the value
    // third index is the direction
    comp: Vec<Vec<Vec<isize>>>,
}

/// The prototype for a wave.
/// Contains information about the compabitility of different values,
/// as well as the dimension and number of values for each node.
/// This is useful in cases where you only want to test the compatibility 
/// (a relatively expensive operation) once
pub struct ProtoGraph {
    // number of dimensions
    n: usize,
    // number of vals
    bits: usize,
    // set of propogators
    propogators: Vec<Vec<Vec<usize>>>,
}

impl ProtoGraph {
    /// Builds a new protograph, checking the provided information to make sure it is consistent.
    /// This method takes a value `n`, which is the number of dimensions, a value 'bits' which is the number of values,
    /// and a multidimensional `Vec` representing the compatibilities between different values.
    /// Note that this does NOT check that the propogators are symmetrical, if they are not symmetrical then 
    /// trying to collapse a wave built with assymetric propogators will very likely fail or cause strange behavior.
    /// This function should almost never be used, instead use `from_align`
    pub fn new(
        n: usize,
        bits: usize,
        propogators: Vec<Vec<Vec<usize>>>,
    ) -> Result<ProtoGraph, &'static str> {
        if Vec::len(&propogators) != bits {
            return Err("Outermost vec size must be equal to the number of values");
        }
        for vec in propogators.iter() {
            if Vec::len(vec) != 2 * n {
                return Err("Inner vecs must have size equal to the number of directions, which is twice the number of dimensions");
            }
        }
        Ok(ProtoGraph {
            n: n,
            bits: bits,
            propogators: propogators,
        })
    }

    /// Creates a protograph by comparing the values in `vals`. 
    /// This checks to make sure all vals have the same dimension.
    pub fn from_align<T: CompatibleAlign<T>>(vals: &[T]) -> Result<ProtoGraph, &'static str> {
        let mut vec = Vec::with_capacity(vals.len());
        let dim = vals[0].dimensions();
        for i in 0..vals.len() {
            if vals[i].dimensions() != dim {
                return Err("Cannot compare differently dimensioned data.");
            }
            let mut directions = Vec::with_capacity(2 * dim);
            for dir in 0..(2 * dim) {
                let mut values = Vec::new();
                for j in 0..vals.len() {
                    if vals[i].align(&vals[j], dir) {
                        values.push(j);
                    }
                }
                directions.push(values);
            }
            vec.push(directions);
        }
        Ok(ProtoGraph {
            n: dim,
            bits: vals.len(),
            propogators: vec,
        })
    }

    /// This can more efficiently generate the propogation vector when using transforms. 
    /// It is currently unimplemented, but will reduce the number of calls to `align` by about 50%
    pub fn from_symmetry_group<S: PartialEq, T: CompatibleAlign<T> + Transformable>(
        _vals: &[T],
        _sym: &SymmetryObject<S>,
    ) -> Result<(ProtoGraph, Vec<T>), &'static str> {
        Err("Efficient ProtoGraph generation from symmetry groups is not implemented yet.")
    }
}

use std::slice;

impl<'a> Wave<'a> {
    pub fn new<'b>(
        proto: &'b ProtoGraph,
        distrib: &'b [f64],
        size: &[usize],
        wrap: &[bool],
    ) -> Wave<'b> {
        let mut product = 1;
        for &s in size.iter() {
            product *= s;
        }
        let mut vals = Vec::with_capacity(product);
        let mut item_vals = Vec::with_capacity(product);
        for i in 0..product {
            item_vals.push(vec![true; proto.bits]);
            // each element in vals contains a reference to one of the arrays in item_vals
            // normally, rust would not let us do this,
            // because it could allow multiple mutable references to the same element in the array
            // here, unsafe code is used because it is known that each element will only be used once
            // since item_vals does not remember these references, this violates mutability rules
            // because item_values can still be mutated
            // methods in this module need to be careful of this when using those values
            let slice = unsafe {
                let pointer = item_vals[i].as_mut_ptr();
                slice::from_raw_parts_mut(pointer, item_vals[i].len())
            };
            vals.push(WaveItem::new(slice, distrib));
        }
        Wave {
            n: proto.n,
            size: size.to_vec(),
            wrap: wrap.to_vec(),
            propogators: &proto.propogators,
            nodes: vals,
            item_values: item_vals,
            comp: Self::gen_compat(product, &proto),
        }
    }

    fn gen_compat(size: usize, proto: &ProtoGraph) -> Vec<Vec<Vec<isize>>> {
        // the basic compatability array, this will be copied for each node
        let mut base = Vec::with_capacity(size);
        for i in 0..proto.bits {
            let mut base_element = Vec::with_capacity(proto.n * 2);
            for k in 0..proto.n * 2 {
                base_element.push((&proto.propogators[i][k]).len() as isize);
            }
            base.push(base_element);
        }
        let mut full_array = Vec::with_capacity(size);
        for _ in 0..size {
            full_array.push(base.clone());
        }
        full_array
    }
    
    fn get_pos(&'a mut self, pos: &[usize]) -> &'a mut WaveItem<'a> {
        let mut align_pos = 0;
        let mut prod = 1;
        for (coord, size) in pos.iter().zip(self.size.iter()) {
            align_pos += coord * prod;
            prod *= size;
        }
        &mut self.nodes[align_pos]
    }
    
    /// Gets the first value for each node that is possible.
    /// Returns `usize::max_value()` for nodes that are impossible.
    pub fn get_data(&self) -> Vec<usize> {
        let size = self.nodes.len();
        let mut vals = vec![usize::max_value(); size];
        for i in 0..size {
            for j in 0..self.nodes[i].vals.len() {
                if self.nodes[i].vals[j] {
                    vals[i] = j;
                    break;
                }
            }
        }
        vals
    }

    /// Uses the provided function to expand the values into a potentially more useable format
    pub fn get_expanded_data<T>(&self, operator: fn(usize) -> T) -> Vec<T> {
        let size = self.nodes.len();
        let mut vals = Vec::with_capacity(size);
        for i in 0..size {
            let mut found = false;
            for j in 0..self.nodes[i].vals.len() {
                if self.nodes[i].vals[j] {
                    vals.push(operator(j));
                    found = true;
                    break;
                }
            }
            if !found {
                vals.push(operator(usize::max_value()));
            }
        }
        vals
    }

    /// Expands the data, allowing each element in the wave to become a subsection of the result.
    /// This is useful for tilesets, where each tile represents several pixels in the output.
    pub fn get_box_expanded_data<T: Default>(
        &self,
        size: &[usize],
        operator: fn(usize, &[usize]) -> T,
    ) -> Vec<T> {
        let mut product = 1;
        for &s in size {
            product *= s;
        }
        let size = self.nodes.len() * product;
        let vals = Vec::with_capacity(size);
        let data = self.get_data();
        let mut i = 0;
        let mut pos = 0;
        let mut result_index = 0;
        let mut box_pos = vec![0; self.n];
        vals
    }

    /// Runs the wavefunction collapse algorithm on this wave. 
    pub fn collapse(&mut self) {
        let mut stack = Vec::with_capacity(self.nodes.len() * self.nodes[0].vals.len());
        self.check(&mut stack);
        let mut ind = 3;
        while {
            while let Some(pos) = stack.pop() {
                self.propogate(pos, &mut stack);
            }
            self.set(&mut stack)
        } {}
    }

    fn check(&mut self, stack: &mut Vec<(usize, usize)>) {
        for i in 0..(&self.comp).len() {
            for j in 0..(&self.comp[i]).len() {
                for k in 0..(&self.comp[i][j]).len() {
                    if self.comp[i][j][k] == 0 {
                        for n in 0..(&self.comp[i][j]).len() {
                            self.comp[i][j][n] = 0;
                        }
                        self.nodes[i].set(j);
                        stack.push((i, j));
                        break;
                    }
                }
            }
        }
    }

    fn propogate(&mut self, pos: (usize, usize), stack: &mut Vec<(usize, usize)>) {
        for item in WaveIterator::new(
            &self.size,
            &self.wrap,
            &self.propogators,
            self.n,
            pos.0,
            pos.1,
        ) {
            for &value in item.1 {
                self.comp[item.0][value][item.2] -= 1;
                if self.nodes[item.0].vals[value] && self.comp[item.0][value][item.2] == 0 {
                    self.nodes[item.0].set(value);
                    stack.push((item.0, value));
                }
            }
        }
    }

    // returns true if a settable value was found
    fn set(&mut self, stack: &mut Vec<(usize, usize)>) -> bool {
        let mut min_entropy = f64::MAX;
        let mut min_index: Option<usize> = None;
        let mut min_count: f64 = 0.0;
        for i in 0..self.nodes.len() {
            if self.nodes[i].num > 1 && self.nodes[i].entropy - min_entropy < 1e-14 {
                // if this entropy is less than than the previous min_entropy (but not equal),
                // then the count needs to get reset
                if min_entropy - self.nodes[i].entropy < 1e-14 {
                    min_count = 0.0;
                    min_entropy = self.nodes[i].entropy;
                }
                min_count += 1.0;
                // if multiple nodes have the same entropy, then choose one at random
                if random::<f64>() < 1.0 / min_count {
                    min_index = Some(i);
                }
            }
        }
        match min_index {
            None => false,
            Some(i) => {
                let mut count = 0.0;
                for j in 0..self.nodes[i].len() {
                    if self.nodes[i].vals[j] {
                        count += self.nodes[i].distrib[j];
                    }
                }
                // pick a random value in the range of the distribution
                let mut pick: f64 = random::<f64>() * count;
                count = 0.0;
                let mut val = 0;
                let mut found = false;
                for j in 0..self.nodes[i].len() {
                    if self.nodes[i].vals[j] {
                        count += self.nodes[i].distrib[j];
                        if !found && count > pick {
                            val = j;
                            found = true;
                        } else {
                            stack.push((i, j));
                        }
                    }
                }
                self.nodes[i].clear(val);
                true
            }
        }
    }
}

// this struct is used to keep track of the entropy of each wave node
struct WaveItem<'a> {
    // the raw values for this node
    // the length of this slice is used to determine the number of values
    vals: &'a mut [bool],

    // the distribution of values
    // this is often the same for several or all nodes, so this is a reference
    // while these are floating point values, they do not need to add to one
    distrib: &'a [f64],

    // these next values are used to help calculate entropy
    // methods modifying a waveitem will modify these values accordingly

    // the current shannon entropy of this node, in bits
    entropy: f64,

    // the distribution multiplier, normalizes the distribution,
    // so that the sum of active probabilities is 1
    mult: f64,

    // the total number of values that are possible
    num: usize,
}

impl<'a> WaveItem<'a> {
    // create a new node
    fn new<'b>(vals: &'b mut [bool], distrib: &'b [f64]) -> WaveItem<'b> {
        if vals.len() > distrib.len() {
            panic!("Not enough elements in distrib!");
        }
        let mut total: f64 = 0.0;
        let mut entropy: f64 = 0.0;
        let mut num: usize = 0;
        let num_vals = vals.len();
        for i in 0..num_vals {
            // allowing for zero probability values is dangerous
            if distrib[i] <= 0.0 {
                vals[i] = false;
            }
            // false values should be ignored
            if vals[i] {
                num += 1;
                total += distrib[i];
                entropy -= distrib[i] * distrib[i].log(2.0);
            }
        }
        // get the distribution multiplier
        let mult = 1.0 / total;
        // correct entropy to modify the distribution
        entropy *= mult;
        // correct for logarithm terms
        entropy -= mult.log(2.0);
        WaveItem {
            vals: vals,
            distrib: distrib,
            mult: mult,
            entropy: entropy,
            num: num,
        }
    }

    // copies and existing node, using the provided slice to store the vals
    fn copy(&self, vals: &'a mut [bool]) -> WaveItem<'a> {
        vals.copy_from_slice(self.vals);
        WaveItem {
            vals: vals,
            distrib: self.distrib,
            mult: self.mult,
            entropy: self.entropy,
            num: self.num,
        }
    }

    // sets the value at the specified position
    fn set(&mut self, pos: usize) {
        if !self.vals[pos] {
            panic!("Value was already set!");
        }
        self.vals[pos] = false;
        // the entropy is equal -sum(p*log(p))
        // when one value is turned off, all values of p increase by a fixed ratio
        let ratio = 1.0 / (1.0 - self.distrib[pos] * self.mult);
        self.num -= 1;
        // get the value of p for the removed node
        let p = self.mult * self.distrib[pos];
        // remove the term for the removed element
        self.entropy += p * p.log(2.0);
        // the removal of one event changes the probabilities of the others
        // correct the multiplier
        self.mult *= ratio;
        // scale up by ratio due to the increase in value
        self.entropy *= ratio;
        // correct for logarithm terms
        self.entropy -= ratio.log(2.0);
    }

    // clears all values and sets one value to be true
    fn clear(&mut self, pos: usize) {
        if !self.vals[pos] {
            panic!("Not a possible value!");
        }
        // set all values in the vector to false
        let len = self.vals.len();
        for i in 0..len {
            self.vals[i] = false;
        }
        // it is important that this still keep the entropy values correct
        // the last value could still be set to false by the propogation
        self.vals[pos] = true;
        self.entropy = 0.0;
        self.mult = 1.0 / self.distrib[pos];
        self.num = 1;
    }

    // gets the length of the node
    fn len(&self) -> usize {
        self.vals.len()
    }
}

struct WaveIterator<'a> {
    index: usize,
    // the position of the node generating this iterator
    pos: usize,
    // the value on that node
    val: usize,
    // the current dimension
    dimension: usize,
    // the current product
    product: usize,
    // the current page size
    page_size: usize,
    // the current sign:
    sign: isize,
    // a value used to test if the new value would wrap
    coord: usize,
    // the number of dimensions
    n: usize,
    // the information used by the iterator
    // layout: (size, wrap, propogator)
    wave_info: (&'a [usize], &'a [bool], &'a Vec<Vec<Vec<usize>>>),
}

use std::slice::Iter;

impl<'b> WaveIterator<'b> {
    fn new<'a>(
        size: &'a [usize],
        wrap: &'a [bool],
        prop: &'a Vec<Vec<Vec<usize>>>,
        dimensions: usize,
        pos: usize,
        val: usize,
    ) -> WaveIterator<'a> {
        WaveIterator {
            index: usize::max_value(),
            pos: pos,
            val: val,
            dimension: 0,
            product: 1,
            page_size: 1,
            sign: -1,
            coord: 1,
            n: dimensions,
            wave_info: (size, wrap, prop),
        }
    }
}

impl<'a> Iterator for WaveIterator<'a> {
    type Item = (usize, Iter<'a, usize>, usize);

    fn next(&mut self) -> Option<(usize, Iter<'a, usize>, usize)> {
        self.index = (self.index as isize + 1) as usize;
        if self.index == 2 * self.n {
            None
        } else {
            let d;
            if self.sign == -1 {
                self.product = self.page_size;
                self.page_size *= self.wave_info.0[self.dimension];
                self.coord = (self.pos / self.page_size) * self.page_size;
                d = 0;
            } else {
                d = 1;
            }
            let mut new_pos = self.pos as isize + self.product as isize * self.sign;
            let wrap = self.wave_info.1[self.dimension];
            self.dimension += d;
            self.sign = -self.sign;
            // check if this position would wrap
            if new_pos < self.coord as isize || new_pos as usize >= self.coord + self.page_size {
                match wrap {
                    // if the dimension wraps, then wrap the position
                    true => new_pos -= self.page_size as isize * self.sign,
                    // otherwise, skip the value
                    // if all dimensions have a size of at least 2,
                    // then this happens a maximum of twice for a single call to next()
                    false => return self.next(),
                }
            }
            
            Some((
                new_pos as usize,
                // we incremented the index earlier, so we need to subtract one here
                (&self.wave_info.2[self.val][self.index]).into_iter(),
                self.index ^ 1,
            ))
        }
    }
}
