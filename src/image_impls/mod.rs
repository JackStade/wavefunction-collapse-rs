use image;
use image::{GenericImage, Rgb, Rgba};
use AlignArray;
use PartialEq;
use ProtoGraph;

// A wrapper that implements AlignArray for a generic image. The difference value returned
// is 0 if the values are equal and 1 if they are not.
pub struct ImageCompareWrapper<'a, T: image::Pixel + 'static, U: GenericImage<Pixel = T> + 'a> {
    image: &'a U,
}

impl<'a, T: PartialEq + image::Pixel + 'static, U: GenericImage<Pixel = T> + 'a> AlignArray
    for ImageCompareWrapper<'a, T, U>
{
    fn get_size(&self, dim: usize) -> usize {
        image_size(self.image, dim)
    }

    fn compare_values(&self, pos_1: &[usize], pos_2: &[usize]) -> f64 {
        if image_compare(self.image, pos_1, pos_2) {
            0.0
        } else {
            1.0
        }
    }
}

// A wrapper that implements AlignArray for a generic image. The difference value returns is
// the sum the squares of the difference between the channels of the pixel.
pub struct ImageSquareDiffWrapper<
    'a,
    T: image::Primitive + 'static,
    U: GenericImage<Pixel = Rgba<T>> + 'static,
> {
    image: &'a U,
}

impl<'a, T: Into<f64> + image::Primitive + 'static, U: GenericImage<Pixel = Rgba<T>> + 'static>
    AlignArray for ImageSquareDiffWrapper<'a, T, U>
{
    fn get_size(&self, dim: usize) -> usize {
        image_size(self.image, dim)
    }

    fn compare_values(&self, pos_1: &[usize], pos_2: &[usize]) -> f64 {
        image_square_diff(self.image, pos_1, pos_2)
    }
}

#[inline(always)]
fn image_size<T: GenericImage>(image: &T, dim: usize) -> usize {
    if dim == 0 {
        image.width() as usize
    } else if dim == 1 {
        image.height() as usize
    } else {
        0
    }
}

#[inline(always)]
fn image_compare<T: PartialEq + image::Pixel + 'static, U: GenericImage<Pixel = T>>(
    image: &U,
    pos_1: &[usize],
    pos_2: &[usize],
) -> bool {
    let p1 = image.get_pixel(pos_1[0] as u32, pos_1[1] as u32);
    let p2 = image.get_pixel(pos_2[0] as u32, pos_2[1] as u32);
    p1 == p2
}

#[inline(always)]
fn image_square_diff<
    T: Into<f64> + image::Primitive + 'static,
    U: GenericImage<Pixel = Rgba<T>>,
>(
    image: &U,
    pos_1: &[usize],
    pos_2: &[usize],
) -> f64 {
    let p1 = image.get_pixel(pos_1[0] as u32, pos_1[1] as u32);
    let p2 = image.get_pixel(pos_2[0] as u32, pos_2[1] as u32);
    let mut sum = 0.0;
    for i in 0..4 {
        let diff = p1.data[i].into() - p2.data[i].into();
        sum += diff * diff;
    }
    sum
}
