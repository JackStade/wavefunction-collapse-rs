use rand::prelude::random;
use std::f64;
use tensor_transforms::SymmetryObject;
use {AlignArray, CompatibleAlign, Transform, Transformable};

/// The basic struct for performing wfc.
///
/// The wfc algorithm is essentially a method for solving a constrained graph.
/// Each node has a number of possible values, which are constrained by all adjacent nodes.
/// The algorithm works by first setting a value with minium nonzero shannon entropy to one of
/// its possible values, and then propogating that information and repeating until all nodes
/// have 0 entropy (each node is set to a value).
///
/// The problem of satisfying these constraints is np-hard (and in fact np-complete),
/// so this heuristic method can fail. Several things help to prevent this from happening:
/// * The constraints are symmetric: i.e. each node has the same constraints
/// * The set of constraints is relatively allowing
///
/// The first of these is enforced by this implementation, but it can be subverted
/// by manually setting initial conditions. The second is good to keep in mind when
/// setting constraints.
///
/// The algorithm is random, and so the only sets of constraints that will *never* work are
/// ones that have no solutions, but some will work only extremely rarely. Interesting,
/// there are some sets of constraints that will *always* work.
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
    // first index is the value,
    // second index is the direction
    propogators: Vec<Vec<Vec<usize>>>,
}

impl ProtoGraph {
    /// Builds a new protograph, checking the provided information to make sure it is consistent.
    ///
    /// This method takes a value `n`, which is the number of dimensions, a value 'bits' which is the number of values,
    /// and a multidimensional `Vec` representing the compatibilities between different values.
    /// Note that this does NOT check that the propogators are symmetrical, if they are not symmetrical then
    /// trying to collapse a wave built with assymetric propogators will very likely fail or cause strange behavior.
    ///
    /// This function is intended to allow external crates to add ways to generate a protograph. The values given to
    /// it should not be generated manually.
    ///
    /// # Panics
    /// This function panics if the provided values as inconsistent. Specifically, the value of `bits` must
    /// be equal to `propogators.len()` and `propogators[i].len()` must be equal to `2 * n` for all `i`.
    pub fn new(n: usize, bits: usize, propogators: Vec<Vec<Vec<usize>>>) -> ProtoGraph {
        if Vec::len(&propogators) != bits {
            panic!("Outermost vec size must be equal to the number of values.");
        }
        for vec in propogators.iter() {
            if Vec::len(vec) != 2 * n {
                panic!("Inner vecs must have size equal to the number of directions, which is twice the number of dimensions.");
            }
        }
        ProtoGraph {
            n: n,
            bits: bits,
            propogators: propogators,
        }
    }

    /// Creates a protograph by comparing the values in `vals`.
    ///
    /// # Panics
    /// Panics if the values do not all have the same dimension.
    pub fn from_align<T: CompatibleAlign<T>>(vals: &[T]) -> ProtoGraph {
        let mut vec = Vec::with_capacity(vals.len());
        let dim = vals[0].dimensions();
        for i in 0..vals.len() {
            if vals[i].dimensions() != dim {
                panic!("Cannot compare differently dimensioned data.");
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
        ProtoGraph {
            n: dim,
            bits: vals.len(),
            propogators: vec,
        }
    }

    /// Generates a ProtoGraph from an AlignArray.
    pub fn from_align_array<T: AlignArray>(
        array: T,
        threshold: f64,
        block_size: &[usize],
        overlap: &[usize],
    ) -> ProtoGraph {
        // here we generally assume that array.get_size() is not
        // more expensive than looking up a value from an array.
        // for this reason, the results are not cached because that would
        // require allocating extra space
        let mut len = array.get_size(0) - block_size[0] + 1;
        let mut dim = 1;
        loop {
            let x = array.get_size(dim) - block_size[dim] + 1;
            if (x != 0) {
                len *= x;
                dim += 1;
            } else {
                break;
            }
        }
        let mut vec = Vec::with_capacity(len);
        let mut pos = vec![0; dim];
        for i in 0..len {
            let mut d = 0;
            loop {
                pos[d] += 1;
                if pos[d] > array.get_size(d) - block_size[d] + 1 {
                    pos[d] = 0;
                    d += 1;
                } else {
                    break;
                }
            }
            let mut directions = Vec::with_capacity(2 * dim);
            for dir_axis in 0..dim {
                // the total threshold in this direction
                let mut total_threshold = threshold;
                for d in 0..dim {
                    if d == dir_axis {
                        total_threshold *= overlap[d] as f64;
                    } else {
                        total_threshold *= block_size[d] as f64;
                    }
                }
                let mut values = Vec::new();
                for dir in 0..2 {
                    let mut pos_2 = vec![0; dim];
                    for k in 0..len {
                        let mut d = 0;
                        loop {
                            pos_2[d] += 1;
                            if pos_2[d] > array.get_size(d) - block_size[d] + 1 {
                                pos_2[d] = 0;
                                d += 1;
                            } else {
                                break;
                            }
                        }
                        let mut pos_1_item;
                        let mut pos_2_item;
                        // pos_1 and pos_2 are swapped based on the sign
                        // of the direction, so from here on the sign doesn't matter
                        if dir == 0 {
                            // the loop that iterates through the values of the positions
                            // will reset these values when finished, so we don't need to clone
                            pos_1_item = &mut pos;
                            pos_2_item = &mut pos_2;
                        } else {
                            pos_1_item = &mut pos_2;
                            pos_2_item = &mut pos;
                        }
                        let align_offset = block_size[dir_axis] - overlap[dir_axis];
                        // we offset the value in pos_1 by the offset
                        pos_1_item[dir_axis] += align_offset;
                        let mut box_dim = 0;
                        let mut box_offset = vec![0; dim];
                        let mut box_total = 0.0;
                        loop {
                            let mut d = 0;
                            loop {
                                if d == dim {
                                    break;
                                };
                                box_offset[d] += 1;
                                // this needs to adjust the values in pos_1_item and
                                // pos_2_item
                                pos_1_item[d] += 1;
                                pos_2_item[d] += 1;
                                // the bounds on the checking direction are different from
                                // the values in the overlap array
                                let mut dimension_max = if d == dir_axis {
                                    overlap[d]
                                } else {
                                    block_size[d]
                                };
                                if box_offset[d] == dimension_max {
                                    box_offset[d] = 0;
                                    // we want to reset the other items to their previous value,
                                    // which is usually not 0
                                    pos_1_item[d] -= dimension_max;
                                    pos_2_item[d] -= dimension_max;
                                    d += 1;
                                } else {
                                    break;
                                }
                            }
                            // when d is maxed, the loop is over
                            if d == dim {
                                break;
                            };
                            // check the two values. If they do not line up, then
                            // the values cannot be said to be comparable
                            box_total += array.compare_values(pos_1_item, pos_2_item);
                            if box_total > total_threshold {
                                break;
                            }
                        }
                        if box_total <= total_threshold {
                            values.push(k);
                        }
                    }
                }
                directions.push(values);
            }
            vec.push(directions);
        }
        ProtoGraph {
            n: dim,
            bits: len,
            propogators: vec,
        }
    }
}

use std::slice;

impl<'a> Wave<'a> {
    /// Creates a new wave. The provided protograph is used to determine the
    /// the constraints on each node. The created graph will start with each value
    /// being possible.
    ///
    /// # Panics
    /// This function will panic if the length of `distrib`, `size`, or `wrap` is smaller
    /// than the length of the ProtoGraph. These lengths can be larger than that length, in
    /// which case extra elements will be ignored.
    pub fn new<'b>(
        proto: &'b ProtoGraph,
        distrib: &'b [f64],
        size: &[usize],
        wrap: &[bool],
    ) -> Wave<'b> {
        let len = proto.bits;
        if distrib.len() < len {
            panic!("The provided distribution does not contain enough elements.");
        }
        if size.len() < len {
            panic!("The provided set of sizes does not contain enough elements.");
        }
        if wrap.len() < len {
            panic!("The provided set of wrapping values does not contain enough elements.");
        }
        let mut product = 1;
        for &s in size.iter() {
            product *= s;
        }
        let mut vals = Vec::with_capacity(product);

        for _ in 0..product {
            vals.push(WaveItem::new(vec![true; len], distrib));
        }
        Wave {
            n: proto.n,
            size: size.to_vec(),
            wrap: wrap.to_vec(),
            propogators: &proto.propogators,
            nodes: vals,
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

    /// Gets the first value for each node that is possible.
    ///
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

    /// Uses the provided function to expand the values into a potentially more useable format.
    pub fn get_expanded_data<T, U: Fn(usize) -> T>(&self, operator: U) -> Vec<T> {
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

    /// Marks all values in the provided stack as impossible and propogates this
    /// information.
    ///
    /// The provided vec will be empty after this returns, but additional elements
    /// will be pushed to it during propogation, potentially increasing its capacity.
    pub fn disable_values(&mut self, stack: &mut Vec<(usize, usize)>) {
        // first mark the values in the stack as false
        for &(node, value) in stack.iter() {
            self.nodes[node].set(value);
        }
        while let Some(pos) = stack.pop() {
            self.propogate(pos, stack);
        }
    }

    /// Runs the wavefunction collapse algorithm on this wave.
    pub fn collapse(&mut self) {
        let mut stack = Vec::with_capacity(self.nodes.len() * self.nodes[0].vals.len());
        self.check(&mut stack);
        // note: do-while loop
        while {
            while let Some(pos) = stack.pop() {
                self.propogate(pos, &mut stack);
            }
            // set returns false when there are no non-zero elements
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
    vals: Vec<bool>,

    // the distribution of values
    // this is often the same for several or all nodes, so this is a reference
    // the distribution does not need to have a total of one, it will be normalized
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
    fn new<'b>(mut vals: Vec<bool>, distrib: &'b [f64]) -> WaveItem<'b> {
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
            // the values passed can contain data that is not marked as possible
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
