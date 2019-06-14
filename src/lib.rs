#[macro_use]
extern crate float_cmp;
#[macro_use]
extern crate approx;
extern crate nalgebra as na;
extern crate statrs;

use nalgebra::base::DMatrix;
use nalgebra::base::DVector;
use nalgebra::linalg::Cholesky;
use statrs::function::erf::erf;
use std::f64::consts::{PI, SQRT_2};

pub fn sigma_n(x: &DVector<f64>, u0: f64, v: f64, sigma0: f64, k: f64) -> f64 {
    let n = x.len();
    let ident = DMatrix::<f64>::identity(n, n);
    let inv_v = DMatrix::<f64>::from_element(n, n, 1.0 / v);
    let cholesky = Cholesky::new(ident + inv_v).expect("cholesky");
    let inv_cholesky = cholesky.inverse();
    let xu = x - DVector::<f64>::repeat(x.len(), u0);
    let xu_transpose = xu.transpose();
    let transform = xu_transpose * inv_cholesky * xu;
    assert!(transform.len() == 1);
    (transform.get((0, 0)).unwrap() + k * sigma0) / (n as f64 + k)
}

#[rustfmt::skip]
pub fn posterior_integral(x: &DVector<f64>, u0: f64, v: f64, sigma0: f64, k: f64) -> f64 {
    let a = sigma_n(x, u0, v, sigma0, k);
    let b = x.mean();
    1.0/2.0*PI.sqrt()*a.sqrt()*(erf(1.0/2.0*SQRT_2*(b - u0)/a.sqrt()) + 1.0)/(PI*a).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::ApproxEq;

    #[test]
    fn test_sigma_n() {
        let x = DVector::<f64>::repeat(10, 0.0);
        let u0 = 0.0;
        let v = 0.01;
        let sigma0 = 5.0;
        let k = 10.0;
        assert!(approx_eq!(f64, dbg!(posterior_integral(&x, u0, v, sigma0, k)), 0.5, ulps = 2));
    }
}
