/// network, backpropagation, AdamW, save/load

use std::fs::File;
use std::io::{Read, Write};
use crate::utils::*;

/// Linear layer: Y = X * W + b
/// We store weights in row-major as matrix (in_dim x out_dim):
/// W[i*out+j] = weight from input i to output j
pub struct Linear {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Vec<f32>,   // weights (in_dim * out_dim)
    pub b: Vec<f32>,   // bias (out_dim)
    // accumulated gradients:
    pub gw: Vec<f32>,  // dL/dW
    pub gb: Vec<f32>,  // dL/dB
    // Adam moments:
    mw: Vec<f32>, vw: Vec<f32>,
    mb: Vec<f32>, vb: Vec<f32>,
    // cache:
    last_x: Vec<f32>,
}

impl Linear {
    /// Create a layer and initialize weights with Xavier/Glorot uniform.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        let k = (6.0f32 / (in_dim as f32 + out_dim as f32)).sqrt();
        let mut w = vec![0.0; in_dim * out_dim];
        for v in &mut w {
            let u = rng.next_f32();          // u ∈ [0,1)
            *v = -k + 2.0 * k * u;           // scale to [−k, k]
        }
        Self {
            in_dim,
            out_dim,
            w,
            b: vec![0.0; out_dim],
            gw: vec![0.0; in_dim * out_dim],
            gb: vec![0.0; out_dim],
            mw: vec![0.0; in_dim * out_dim],
            vw: vec![0.0; in_dim * out_dim],
            mb: vec![0.0; out_dim],
            vb: vec![0.0; out_dim],
            last_x: vec![0.0; in_dim],
        }
    }

    pub fn zero_grad(&mut self) {
        for g in &mut self.gw { *g = 0.0; }
        for g in &mut self.gb { *g = 0.0; }
    }

    /// Forward pass for one sample.
    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.in_dim);
        self.last_x.copy_from_slice(x);

        let mut y = vec![0.0f32; self.out_dim];
        for j in 0..self.out_dim {
            let mut acc = self.b[j];
            for i in 0..self.in_dim {
                acc += x[i] * self.w[i * self.out_dim + j];
            }
            y[j] = acc;
        }
        y
    }

    /// Backward pass: accumulate dW, dB and return dX.
    pub fn backward(&mut self, dy: &[f32]) -> Vec<f32> {
        debug_assert_eq!(dy.len(), self.out_dim);

        // dW = X^T * dY
        for i in 0..self.in_dim {
            let xi = self.last_x[i];
            let row = i * self.out_dim;
            for j in 0..self.out_dim {
                self.gw[row + j] += xi * dy[j];
            }
        }
        // dB = dY
        for j in 0..self.out_dim {
            self.gb[j] += dy[j];
        }

        // dX = dY * W^T
        let mut dx = vec![0.0f32; self.in_dim];
        for i in 0..self.in_dim {
            let mut acc = 0.0f32;
            let row = i * self.out_dim;
            for j in 0..self.out_dim {
                acc += dy[j] * self.w[row + j];
            }
            dx[i] = acc;
        }
        dx
    }

    /// AdamW step for this layer (with global grad scaling and decoupled weight decay).
    pub fn step_adam(
        &mut self,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        t: u64,
        grad_scale: f32,
        weight_decay: f32,   // decoupled L2 on weights (not on bias)
    ) {
        let t_f = t as f32;
        let b1t = b1.powf(t_f);
        let b2t = b2.powf(t_f);
        let corr1 = 1.0 - b1t;
        let corr2 = 1.0 - b2t;

        // Weights
        for i in 0..self.w.len() {
            let g = self.gw[i] * grad_scale;
            self.mw[i] = b1 * self.mw[i] + (1.0 - b1) * g;
            self.vw[i] = b2 * self.vw[i] + (1.0 - b2) * (g * g);
            let m_hat = self.mw[i] / corr1.max(1e-8);
            let v_hat = self.vw[i] / corr2.max(1e-8);
            // Adam update
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            // Decoupled weight decay (AdamW)
            if weight_decay > 0.0 {
                self.w[i] -= lr * weight_decay * self.w[i];
            }
        }
        // Bias (no decay)
        for i in 0..self.b.len() {
            let g = self.gb[i] * grad_scale;
            self.mb[i] = b1 * self.mb[i] + (1.0 - b1) * g;
            self.vb[i] = b2 * self.vb[i] + (1.0 - b2) * (g * g);
            let m_hat = self.mb[i] / corr1.max(1e-8);
            let v_hat = self.vb[i] / corr2.max(1e-8);
            self.b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    /// L2 sum of gradients (for global clip).
    pub fn grad_l2_sum(&self) -> f32 {
        let mut s = 0.0;
        for g in &self.gw { s += g * g; }
        for g in &self.gb { s += g * g; }
        s
    }

    pub fn non_finite_in_params_or_grads(&self) -> bool {
        has_non_finite(&self.w) || has_non_finite(&self.b) ||
            has_non_finite(&self.gw) || has_non_finite(&self.gb)
    }

    /// Clamp parameters into [-max_abs, max_abs] (hard safety rail).
    pub fn clamp_params(&mut self, max_abs: f32) {
        let clamp = |v: &mut f32| {
            if *v >  max_abs { *v =  max_abs; }
            if *v < -max_abs { *v = -max_abs; }
        };
        for w in &mut self.w { clamp(w); }
        for b in &mut self.b { clamp(b); }
    }
}

/// ReLU layer with mask.
struct ReLU {
    mask: Vec<u8>, // 1 — pass gradient, 0 — mute
}
impl ReLU {
    fn new(size: usize) -> Self { Self { mask: vec![0; size] } }

    fn forward(&mut self, z: &mut [f32]) {
        for i in 0..z.len() {
            if z[i] > 0.0 {
                self.mask[i] = 1;
            } else {
                self.mask[i] = 0;
                z[i] = 0.0;
            }
        }
    }
    fn backward(&self, da: &mut [f32]) {
        for i in 0..da.len() {
            if self.mask[i] == 0 { da[i] = 0.0; }
        }
    }
}

/// Net: [obs] -> Linear -> ReLU -> Linear -> ReLU -> Linear -> [Q]
pub struct Net {
    pub din: usize,
    pub h1: usize,
    pub h2: usize,
    pub dout: usize,

    l1: Linear, a1: ReLU,
    l2: Linear, a2: ReLU,
    l3: Linear,

    pub t_adam: u64,
}

impl Net {
    pub fn new(din: usize, h1: usize, h2: usize, dout: usize, mut rng: LcgRng) -> Self {
        let l1 = Linear::new(din, h1, &mut rng);
        let l2 = Linear::new(h1, h2, &mut rng);
        let l3 = Linear::new(h2, dout, &mut rng);
        Self { din, h1, h2, dout, l1, a1: ReLU::new(h1), l2, a2: ReLU::new(h2), l3, t_adam: 0 }
    }

    pub fn grad_l2_sum_all(&self) -> f32 {
        self.l1.grad_l2_sum() + self.l2.grad_l2_sum() + self.l3.grad_l2_sum()
    }
    pub fn non_finite_any(&self) -> bool {
        self.l1.non_finite_in_params_or_grads() ||
            self.l2.non_finite_in_params_or_grads() ||
            self.l3.non_finite_in_params_or_grads()
    }

    pub fn zero_grad(&mut self) {
        self.l1.zero_grad();
        self.l2.zero_grad();
        self.l3.zero_grad();
    }

    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        let mut z1 = self.l1.forward(x);
        self.a1.forward(&mut z1);
        let mut z2 = self.l2.forward(&z1);
        self.a2.forward(&mut z2);
        self.l3.forward(&z2)
    }

    pub fn backward_from_output_grad(&mut self, d_q: Vec<f32>) {
        let mut da2 = self.l3.backward(&d_q);
        self.a2.backward(&mut da2);
        let mut da1 = self.l2.backward(&da2);
        self.a1.backward(&mut da1);
        let _dx = self.l1.backward(&da1);
    }

    /// Global grad-norm clip; return scale (<=1 if clipped).
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        let s = self.grad_l2_sum_all();
        let norm = s.sqrt();
        if norm <= max_norm || norm == 0.0 { return 1.0; }
        max_norm / norm
    }

    /// AdamW step for all layers.
    pub fn step_adam(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, grad_scale: f32, weight_decay: f32) {
        self.t_adam += 1;
        self.l1.step_adam(lr, b1, b2, eps, self.t_adam, grad_scale, weight_decay);
        self.l2.step_adam(lr, b1, b2, eps, self.t_adam, grad_scale, weight_decay);
        self.l3.step_adam(lr, b1, b2, eps, self.t_adam, grad_scale, weight_decay);
    }

    /// Soft update θ_target ← (1−τ)θ_target + τ θ_online.
    pub fn soft_update_from(&mut self, online: &Net, tau: f32) {
        fn mix(dst: &mut [f32], src_vals: &[f32], tau: f32) {
            debug_assert_eq!(dst.len(), src_vals.len());
            for (d, &s) in dst.iter_mut().zip(src_vals.iter()) {
                *d = (1.0 - tau) * *d + tau * s;
            }
        }
        mix(&mut self.l1.w, &online.l1.w, tau);
        mix(&mut self.l1.b, &online.l1.b, tau);
        mix(&mut self.l2.w, &online.l2.w, tau);
        mix(&mut self.l2.b, &online.l2.b, tau);
        mix(&mut self.l3.w, &online.l3.w, tau);
        mix(&mut self.l3.b, &online.l3.b, tau);
    }

    /// Hard copy parameters (for target init).
    pub fn copy_from(&mut self, src: &Net) {
        self.l1.w.clone_from(&src.l1.w);
        self.l1.b.clone_from(&src.l1.b);
        self.l2.w.clone_from(&src.l2.w);
        self.l2.b.clone_from(&src.l2.b);
        self.l3.w.clone_from(&src.l3.w);
        self.l3.b.clone_from(&src.l3.b);
    }

    /// Clamp all parameters after an optimizer step.
    pub fn clamp_params(&mut self, max_abs: f32) {
        self.l1.clamp_params(max_abs);
        self.l2.clamp_params(max_abs);
        self.l3.clamp_params(max_abs);
    }

    // ---- serialization ----

    pub fn save(&self, path: &str) -> std::io::Result<()> {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"SNET");
        buf.extend_from_slice(&1u32.to_le_bytes());
        for v in [self.din, self.h1, self.h2, self.dout] {
            buf.extend_from_slice(&(v as u32).to_le_bytes());
        }
        buf.extend_from_slice(&self.t_adam.to_le_bytes());
        self.l1.write_to(&mut buf);
        self.l2.write_to(&mut buf);
        self.l3.write_to(&mut buf);
        let mut f = File::create(path)?;
        f.write_all(&buf)?;
        Ok(())
    }

    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        if &buf[0..4] != b"SNET" { return Err(std::io::Error::new(std::io::ErrorKind::Other, "bad header")); }
        let mut off = 4;
        let mut rd_u32 = |o: &mut usize| -> u32 { let mut b=[0u8;4]; b.copy_from_slice(&buf[*o..*o+4]); *o+=4; u32::from_le_bytes(b) };
        let mut rd_u64 = |o: &mut usize| -> u64 { let mut b=[0u8;8]; b.copy_from_slice(&buf[*o..*o+8]); *o+=8; u64::from_le_bytes(b) };
        let _ver = rd_u32(&mut off);
        let din = rd_u32(&mut off) as usize;
        let h1  = rd_u32(&mut off) as usize;
        let h2  = rd_u32(&mut off) as usize;
        let dout= rd_u32(&mut off) as usize;
        if din != self.din || h1 != self.h1 || h2 != self.h2 || dout != self.dout {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "shape mismatch"));
        }
        self.t_adam = rd_u64(&mut off);
        off = self.l1.read_from(&buf, off);
        off = self.l2.read_from(&buf, off);
        let _ = self.l3.read_from(&buf, off);
        Ok(())
    }

    // --- private (serialization helpers) ---

    fn write_to(&self, _out: &mut Vec<u8>) {}
    fn read_from(&mut self, _data: &[u8], _off: usize) -> usize { _off }
}

// keep Linear::write_to/read_from visible to Net
impl Linear {
    fn write_to(&self, out: &mut Vec<u8>) {
        out.extend_from_slice(&(self.in_dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.out_dim as u32).to_le_bytes());
        for v in &self.w { out.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.b { out.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.mw { out.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.vw { out.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.mb { out.extend_from_slice(&v.to_le_bytes()); }
        for v in &self.vb { out.extend_from_slice(&v.to_le_bytes()); }
    }
    fn read_from(&mut self, data: &[u8], mut off: usize) -> usize {
        let rd_u32 = |d: &[u8], o: &mut usize| -> u32 { let mut b=[0u8;4]; b.copy_from_slice(&d[*o..*o+4]); *o+=4; u32::from_le_bytes(b) };
        let rd_f32 = |d: &[u8], o: &mut usize| -> f32 { let mut b=[0u8;4]; b.copy_from_slice(&d[*o..*o+4]); *o+=4; f32::from_le_bytes(b) };
        let in_dim  = rd_u32(data, &mut off) as usize;
        let out_dim = rd_u32(data, &mut off) as usize;
        assert_eq!(in_dim, self.in_dim);
        assert_eq!(out_dim, self.out_dim);
        for v in &mut self.w  { *v = rd_f32(data, &mut off); }
        for v in &mut self.b  { *v = rd_f32(data, &mut off); }
        for v in &mut self.mw { *v = rd_f32(data, &mut off); }
        for v in &mut self.vw { *v = rd_f32(data, &mut off); }
        for v in &mut self.mb { *v = rd_f32(data, &mut off); }
        for v in &mut self.vb { *v = rd_f32(data, &mut off); }
        off
    }
}
