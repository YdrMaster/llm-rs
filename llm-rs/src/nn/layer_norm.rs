use super::{NeuralNetwork, macros::*, unique};
use crate::{Blob, Context, Tensor};
use digit_layout::types;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tensor::rw_rc::RwRc;

pub struct LayerNorm {
    w: RwRc<Tensor<Blob>>,
    b: RwRc<Tensor<Blob>>,
    x: Option<RwRc<Tensor<Blob>>>,
    mean: Option<Tensor<Blob>>,
    rstd: Option<Tensor<Blob>>,
}

impl NeuralNetwork for LayerNorm {
    type Init = [RwRc<Tensor<Blob>>; 2];

    fn init(init: Self::Init, _ctx: &mut Context) -> Self {
        let [scalar, bias] = init;
        Self {
            w: scalar,
            b: bias,
            x: None,
            mean: None,
            rstd: None,
        }
    }

    fn forward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([x] = inputs);
        self.x.replace(x);
        let Self { w, b, x, .. } = self;

        let x = x.as_ref().unwrap().read();
        dims!([batch_size, n_seq, d] = x);

        let mut y = ctx.tensor(x.dt(), &[batch_size, n_seq, d]);
        let mut mean = ctx.tensor(x.dt(), &[batch_size, n_seq]);
        let mut rstd = ctx.tensor(x.dt(), &[batch_size, n_seq]);

        ctx.bench(|| {
            forward::layer_norm(
                y.as_deref_mut().merge(0, 2),
                mean.as_deref_mut().merge(0, 2),
                rstd.as_deref_mut().merge(0, 2),
                x.as_deref().merge(0, 2),
                w.read().as_deref(),
                b.read().as_deref(),
            )
        });

        self.mean.replace(mean);
        self.rstd.replace(rstd);

        vec![y.share()]
    }

    fn backward(
        &mut self,
        inputs: impl IntoIterator<Item = RwRc<Tensor<Blob>>>,
        ctx: &mut Context,
    ) -> Vec<RwRc<Tensor<Blob>>> {
        destruct!([dy] = inputs);
        let Self {
            w,
            b,
            x,
            mean,
            rstd,
        } = self;

        let x = x.take().unwrap();
        let x = x.read();
        let mut dx = Tensor::contiguous_of(x).map(Blob::new_zeroed);

        let dw = ctx.write_gradient("w", w);
        let db = ctx.write_gradient("b", b);
        ctx.bench(|| {
            backward::layer_norm(
                dx.as_deref_mut().merge(0, 2),
                dw.write().as_deref_mut(),
                db.write().as_deref_mut(),
                dy.read().as_deref().merge(0, 2),
                x.as_deref().merge(0, 2),
                w.read().as_deref(),
                mean.take().unwrap().as_deref().merge(0, 2),
                rstd.take().unwrap().as_deref().merge(0, 2),
            )
        });

        w.release();
        b.release();

        vec![dx.share()]
    }
}

mod forward {
    use super::*;

    pub(super) fn layer_norm(
        mut y: Tensor<&mut [u8]>,
        mut mean: Tensor<&mut [u8]>,
        mut rstd: Tensor<&mut [u8]>,
        x: Tensor<&[u8]>,
        scalar: Tensor<&[u8]>,
        bias: Tensor<&[u8]>,
    ) {
        let dt = unique(&[y.dt(), mean.dt(), rstd.dt(), x.dt(), scalar.dt(), bias.dt()]).unwrap();
        assert_eq!(dt, types::F32);

        dims!([n, d_0] = y);
        dims!([n_1, d_1] = x);
        dims!([n_2] = mean);
        dims!([n_3] = rstd);
        dims!([d_2] = scalar);
        dims!([d_3] = bias);

        let n = unique(&[n, n_1, n_2, n_3]).unwrap();
        let d = unique(&[d_0, d_1, d_2, d_3]).unwrap();

        strides!([nsy, dsy] = y);
        strides!([nsx, dsx] = x);
        strides!([nsm] = mean);
        strides!([nsr] = rstd);
        strides!([sw] = scalar);
        strides!([sb] = bias);

        let scheme = Scheme {
            n,
            d,
            sy: [nsy, dsy],
            sx: [nsx, dsx],
            sm: [nsm],
            sr: [nsr],
            sw: [sw],
            sb: [sb],
            y: y.mut_ptr(),
            x: x.ptr(),
            mean: mean.mut_ptr(),
            rstd: rstd.mut_ptr(),
            scalar: scalar.ptr(),
            bias: bias.ptr(),
        };

        match dt {
            types::F32 => scheme.compute::<f32>(),
            _ => todo!(),
        }
    }

    struct Scheme {
        n: usize,
        d: usize,
        sy: [isize; 2],
        sx: [isize; 2],
        sm: [isize; 1],
        sr: [isize; 1],
        sw: [isize; 1],
        sb: [isize; 1],
        y: *mut u8,
        x: *const u8,
        mean: *mut u8,
        rstd: *mut u8,
        scalar: *const u8,
        bias: *const u8,
    }

    impl Scheme {
        fn compute<
            T: Copy
                + std::ops::Add<Output = T>
                + std::ops::Mul<Output = T>
                + std::ops::Sub<Output = T>
                + std::fmt::Debug,
        >(
            &self,
        ) where
            f32: From<T>,
            T: From<f32>,
        {
            let &Self {
                n,
                d,
                sy,
                sx,
                sm,
                sr,
                sw,
                sb,
                y,
                x,
                mean,
                rstd,
                scalar,
                bias,
            } = self;
            let y = y as usize;
            let x = x as usize;
            let mean = mean as usize;
            let rstd = rstd as usize;
            let scalar = scalar as usize;
            let bias = bias as usize;

            const EPSILON: f32 = 1e-5;

            // 处理每个batch序列
            (0..n).into_par_iter().for_each(|bt| {
                let bt = bt as isize;

                // 计算均值和方差
                let mut sum: f32 = 0.0;
                let mut sum2: f32 = 0.0;

                for j in 0..d {
                    let j = j as isize;
                    let [nsx, dsx] = sx;
                    let x_val = unsafe { *(x as *const T).byte_offset(bt * nsx + j * dsx) };
                    let x_f32: f32 = x_val.into();
                    sum += x_f32;
                    sum2 += x_f32 * x_f32;
                }

                let mean_val = sum / (d as f32);
                let rstd_val = (sum2 / (d as f32) - mean_val * mean_val + EPSILON).powf(-0.5);

                // 存储均值和标准差倒数
                let [nsm] = sm;
                unsafe {
                    *(mean as *mut T).byte_offset(bt * nsm) = T::from(mean_val);
                }

                let [nsr] = sr;
                unsafe {
                    *(rstd as *mut T).byte_offset(bt * nsr) = T::from(rstd_val);
                }

                // 计算输出
                for j in 0..d {
                    let j = j as isize;

                    // 获取输入值
                    let [nsx, dsx] = sx;
                    let x_val = unsafe { *(x as *const T).byte_offset(bt * nsx + j * dsx) };
                    let x_f32: f32 = x_val.into();

                    // 获取权重和偏置
                    let [sw] = sw;
                    let scalar_val = unsafe { *(scalar as *const T).byte_offset(j * sw) };

                    let [sb] = sb;
                    let bias_val = unsafe { *(bias as *const T).byte_offset(j * sb) };

                    // 计算归一化结果
                    let norm = (x_f32 - mean_val) * rstd_val;

                    let result = norm * f32::from(scalar_val) + f32::from(bias_val);

                    // 存储结果
                    let [nsy, dsy] = sy;
                    unsafe {
                        *(y as *mut T).byte_offset(bt * nsy + j * dsy) = T::from(result);
                    }
                }
            });
        }
    }
}

mod backward {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    pub(super) fn layer_norm(
        mut dx: Tensor<&mut [u8]>,
        mut dw: Tensor<&mut [u8]>,
        mut db: Tensor<&mut [u8]>,
        dy: Tensor<&[u8]>,
        x: Tensor<&[u8]>,
        w: Tensor<&[u8]>,
        mean: Tensor<&[u8]>,
        rstd: Tensor<&[u8]>,
    ) {
        let dt = unique(&[
            dx.dt(),
            dw.dt(),
            db.dt(),
            dy.dt(),
            x.dt(),
            w.dt(),
            mean.dt(),
            rstd.dt(),
        ])
        .unwrap();
        assert_eq!(dt, types::F32);

        dims!([n, d_0] = dx);
        dims!([n_1, d_1] = dy);
        dims!([n_2, d_2] = x);
        dims!([d_3] = dw);
        dims!([d_4] = db);
        dims!([d_5] = w);
        dims!([n_3] = mean);
        dims!([n_4] = rstd);

        let n = unique(&[n, n_1, n_2, n_3, n_4]).unwrap();
        let d = unique(&[d_0, d_1, d_2, d_3, d_4, d_5]).unwrap();

        strides!([nsdx, dsdx] = dx);
        strides!([nsdy, dsdy] = dy);
        strides!([nsx, dsx] = x);
        strides!([dw_s] = dw);
        strides!([db_s] = db);
        strides!([w_s] = w);
        strides!([nsm] = mean);
        strides!([nsr] = rstd);

        let scheme = Scheme {
            n,
            d,
            sdx: [nsdx, dsdx],
            sdy: [nsdy, dsdy],
            sx: [nsx, dsx],
            sdw: [dw_s],
            sdb: [db_s],
            sw: [w_s],
            sm: [nsm],
            sr: [nsr],
            dx: dx.mut_ptr(),
            dy: dy.ptr(),
            x: x.ptr(),
            dw: dw.mut_ptr(),
            db: db.mut_ptr(),
            w: w.ptr(),
            mean: mean.ptr(),
            rstd: rstd.ptr(),
        };

        match dt {
            types::F32 => scheme.compute::<f32>(),
            _ => todo!(),
        }
    }

    struct Scheme {
        n: usize,
        d: usize,
        sdx: [isize; 2],
        sdy: [isize; 2],
        sx: [isize; 2],
        sdw: [isize; 1],
        sdb: [isize; 1],
        sw: [isize; 1],
        sm: [isize; 1],
        sr: [isize; 1],
        dx: *mut u8,
        dy: *const u8,
        x: *const u8,
        dw: *mut u8,
        db: *mut u8,
        w: *const u8,
        mean: *const u8,
        rstd: *const u8,
    }

    impl Scheme {
        fn compute<
            T: Copy
                + std::ops::Add<Output = T>
                + std::ops::AddAssign
                + std::ops::Mul<Output = T>
                + std::ops::Sub<Output = T>,
        >(
            &self,
        ) where
            f32: From<T>,
            T: From<f32>,
        {
            let &Self {
                n,
                d,
                sdx,
                sdy,
                sx,
                sdw,
                sdb,
                sw,
                sm,
                sr,
                dx,
                dy,
                x,
                dw,
                db,
                w,
                mean,
                rstd,
            } = self;

            let dx = dx as usize;
            let dy = dy as usize;
            let x = x as usize;
            let dw = dw as usize;
            let db = db as usize;
            let w = w as usize;
            let mean = mean as usize;
            let rstd = rstd as usize;

            // 处理每个batch序列
            (0..n).into_par_iter().for_each(|bt| {
                let bt = bt as isize;

                // 获取当前batch的mean和rstd
                let [sm] = sm;
                let mean_val = unsafe { f32::from(*(mean as *const T).byte_offset(bt * sm)) };

                let [sr] = sr;
                let rstd_val = unsafe { f32::from(*(rstd as *const T).byte_offset(bt * sr)) };

                // 计算中间值
                let mut dnorm_mean: f32 = 0.0;
                let mut dnorm_norm_mean: f32 = 0.0;

                for j in 0..d {
                    let j = j as isize;

                    // 获取各个值
                    let [sdy, dsdy] = sdy;
                    let dy_val =
                        unsafe { f32::from(*(dy as *const T).byte_offset(bt * sdy + j * dsdy)) };

                    let [sx, dsx] = sx;
                    let x_val =
                        unsafe { f32::from(*(x as *const T).byte_offset(bt * sx + j * dsx)) };

                    let [sw] = sw;
                    let w_val = unsafe { f32::from(*(w as *const T).byte_offset(j * sw)) };

                    let norm = (x_val - mean_val) * rstd_val;
                    let dnorm = w_val * dy_val;

                    dnorm_mean += dnorm;
                    dnorm_norm_mean += norm * dnorm;
                }

                dnorm_mean /= d as f32;
                dnorm_norm_mean /= d as f32;

                // 更新梯度
                for j in 0..d {
                    let j = j as isize;

                    // 获取各个值
                    let [sdy, dsdy] = sdy;
                    let dy_val =
                        unsafe { f32::from(*(dy as *const T).byte_offset(bt * sdy + j * dsdy)) };

                    let [sx, dsx] = sx;
                    let x_val =
                        unsafe { f32::from(*(x as *const T).byte_offset(bt * sx + j * dsx)) };

                    let [sw] = sw;
                    let w_val = unsafe { f32::from(*(w as *const T).byte_offset(j * sw)) };

                    let norm = (x_val - mean_val) * rstd_val;
                    let dnorm = w_val * dy_val;

                    // 更新db
                    let [sdb] = sdb;
                    unsafe {
                        let db_ptr = (db as *mut T).byte_offset(j * sdb);
                        *db_ptr = T::from(f32::from(*db_ptr) + dy_val);
                    }

                    // 更新dw
                    let [sdw] = sdw;
                    unsafe {
                        let dw_ptr = (dw as *mut T).byte_offset(j * sdw);
                        *dw_ptr = T::from(f32::from(*dw_ptr) + norm * dy_val);
                    }

                    // 更新dx
                    let [sdx, dsdx] = sdx;
                    unsafe {
                        let dx_ptr = (dx as *mut T).byte_offset(bt * sdx + j * dsdx);
                        let dx_val = rstd_val * (dnorm - dnorm_mean - norm * dnorm_norm_mean);
                        *dx_ptr = T::from(f32::from(*dx_ptr) + dx_val);
                    }
                }
            });
        }
    }
}
