/*
 * // Copyright (c) Radzivon Bartoshyk 12/2025. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::mla::fmla;
use num_traits::{AsPrimitive, Float, MulAdd};
use std::ops::AddAssign;

mod mla;

pub(crate) trait TrapezSample:
    Float + PartialOrd + PartialEq + AddAssign + MulAdd<Self, Output = Self> + 'static
{
    const TOLERANCE: Self;
}

impl TrapezSample for f32 {
    const TOLERANCE: Self = 1e-6;
}

impl TrapezSample for f64 {
    const TOLERANCE: Self = 1e-12;
}

/// Calculates the definite integral of a dataset using the trapezoidal rule.
///
/// This implementation handles non-uniform spacing between x-values by
/// calculating the area of each trapezoidal segment directly.
///
/// # Arguments
/// * `y` - The array of function values.
/// * `x` - The array of abscissas.
///
/// # Returns
/// The approximate definite integral (area under the curve).
pub fn trapezoid_f64(y: &[f64], x: &[f64]) -> f64 {
    trapezoid(y, x)
}

/// Calculates the definite integral of a dataset using the trapezoidal rule.
///
/// This implementation handles non-uniform spacing between x-values by
/// calculating the area of each trapezoidal segment directly.
///
/// # Arguments
/// * `y` - The array of function values.
/// * `x` - The array of abscissas.
///
/// # Returns
/// The approximate definite integral (area under the curve).
pub fn trapezoid_f32(y: &[f64], x: &[f64]) -> f64 {
    trapezoid(y, x)
}

fn trapezoid<T: TrapezSample>(y: &[T], x: &[T]) -> T
where
    f64: AsPrimitive<T>,
{
    let n = y.len();
    if n < 2 || x.len() != n {
        return T::nan();
    }

    // Quick check for exact uniform spacing using first interval.
    let h0 = x[1] - x[0];
    // tolerance scaled to magnitude of h0 (and at least a tiny absolute tol)
    let tol = (h0.abs().max(1.0.as_())) * T::TOLERANCE;

    let mut uniform = true;
    let q = &x[1..];
    for x in q.windows(2) {
        let hi = x[1] - x[0];
        if (hi - h0).abs() > tol {
            uniform = false;
            break;
        }
    }

    if uniform {
        // Use the optimized uniform-spacing trapezoid rule:
        // integral = h * ( 0.5*y0 + sum(y[1..n-1]) + 0.5*yn )
        let mut interior_sum = T::zero();
        for v in &y[1..(n - 1)] {
            interior_sum += *v;
        }
        h0 * fmla(y[0] + y[n - 1], 0.5f64.as_(), interior_sum)
    } else {
        // General (non-uniform) trapezoid rule
        let mut integral = T::zero();
        for (y, x) in y.windows(2).zip(x.windows(2)) {
            let dx = x[1] - x[0];
            integral = fmla(dx * 0.5f64.as_(), y[0] + y[1], integral);
        }
        integral
    }
}

/// Trapezoidal integration for evenly spaced samples.
/// `y` = function values
/// `dx` = spacing between x-values
pub fn trapezoid_even_f32(y: &[f32], dx: f32) -> f32 {
    trapezoid_even(y, dx)
}

/// Trapezoidal integration for evenly spaced samples.
/// `y` = function values
/// `dx` = spacing between x-values
pub fn trapezoid_even_f64(y: &[f64], dx: f64) -> f64 {
    trapezoid_even(y, dx)
}

/// Trapezoidal integration for evenly spaced samples.
/// `y` = function values
/// `dx` = spacing between x-values
fn trapezoid_even<T: TrapezSample>(y: &[T], dx: T) -> T
where
    f64: AsPrimitive<T>,
{
    let n = y.len();
    if n < 2 || dx <= 0.0f64.as_() {
        return T::nan();
    }

    // sum of interior terms
    let mut sum = T::zero();
    let q = &y[1..n - 1];
    for &v in q.iter() {
        sum += v;
    }

    dx * fmla(0.5f64.as_(), y[0] + y[n - 1], sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trapezoid() {
        let result = trapezoid_f64(&[5., 6., 1., 4., 6., 2.], &[1., 2., 4., 6., 7., 9.]);
        assert_eq!(result, 30.5);
    }

    #[test]
    fn test_trapezoid_even() {
        let result = trapezoid_even(&[5., 6., 1., 4., 6., 2.], 0.003);
        assert_eq!(result, 0.0615);
    }
}
