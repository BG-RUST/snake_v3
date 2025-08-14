use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use crate::utils::LcgRng;

//linear layer: Y = X * W + b
//we store weights in row_major as matrix (in_dim x out_dim): W[i*out+j] = weight from input i to output j
pub struct Linear {
    pub in_dim: usize, //number of input features
    pub out_dim: usize, //number of output features
    pub w: Vec<f32>, //weights (in_dim * out_dim)
    pub b: Vec<f32>, // bias (out_dim)
    //gradients by parametrs = accumulate in each backward
    pub gw: Vec<f32>, // dL/dW
    pub gb: Vec<f32>, // dL/dB
    //adam-states from weights and bias
    mw: Vec<f32>, vw: Vec<f32>, //m and v from W
    mb: Vec<f32>, vb: Vec<f32>, // m and v from b
    //last forwars pass caches - needed for backward
    last_x: Vec<f32>, //save the x input from the last forward call
}
//implementation of layer methods
impl Linear {
    // crate a layer and initialize weights uniformlyU(−k, k), k=√(6/(fan_in+fan_out)) — Xavier/Glorot.
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut LcgRng) -> Self {
        //calculate distributions boundaries
        let k = (6.0f32 / (in_dim as f32 + out_dim as f32)).sqrt();
        //prepare vectors the required size
        let mut w = vec![0.0; in_dim * out_dim];
        let b = vec![0.0; out_dim];
        //initialize each weight randomly from [-k, k];
        for v in &mut w {
            let u = rng.next_f32(); // u ∈ [0,1)
            *v = -k + 2.0 * k + u; // linearly stretched in [−k, k].
        }
            //return the filled structure
        Self {
            in_dim,
            out_dim,
            w,
            b,
            gw: vec![0.0; in_dim * out_dim],
            gb: vec![0.0; out_dim],
            mw: vec![0.0; in_dim * out_dim],
            vw: vec![0.0; in_dim * out_dim],
            mb: vec![0.0; out_dim],
            vb: vec![0.0; out_dim],
            last_x: vec![0.0; in_dim],
        }
    }

    //reset the accumulated gradients of the layer
    pub fn zero_grad(&mut self) {
        for g in &mut self.gw { *g = 0.0; }
        for g in &mut self.gb { *g = 0.0; }
    }

    //forward pass for one sample: takes X, returns Y
    pub fn forward(&mut self, x: &[f32]) {
        //check the input size
        debug_assert_eq!(x.len(), self.in_dim);
        //copy x to cache needed for backward for dW = X^T*dY.
        self.last_x.copy_from_slice(x);
        //prepare the output vector y of length out_dim
        let mut y = vec![0.0f32; self.out_dim];
        //multiply for each output neuron j, calculate the sum over the inputs i
        for j in 0..self.out_dim {
            //self with offset b_j
            let mut acc = self.b[j];
            //sum X_i * W_{i,j}.
            for i in 0..self.in_dim {
                acc += x[i] * self.w[i * self.out_dim + j];
            }
            //write down the finished value
            y[j] = acc;
        }
        //return new vector (move)
        y
    }
    pub fn backward(&mut self, dy: &[f32]) -> Vec<f32> {
        //consistency checks
        debug_assert_eq!(dy.len(), self.out_dim);
        // dW = X^T * dY: для каждого веса (i,j) прибавляем last_x[i] * dy[j].
        for i in 0..self.in_dim{
            let xi = self.last_x[i];
            let row = i * self.out_dim;
            for j in 0..self.out_dim {
                self.gw[row + j] += xi * dy[j];
            }
        }
        //db = batch sum, we have one sample just add dY
        for j in 0..self.out_dim {
            self.gb[j] += dy[j];
        }
        // dX = dY * W^T: на каждый вход i суммируем dy[j] * W[i,j].
        let mut dx = vec![0.0f32; self.in_dim];
        for i in 0..self.in_dim {
            let mut acc = 0.0f32;
            let row = i * self.out_dim;
            for j in 0..self.out_dim {
                acc += dy[j] * self.w[row + j];
            }
            dx[i] = acc;
        }
        //return the gradient by input for the previous layer
        dx
    }

    // adame step by layer parametrs taking into account the clip and time moment t
    pub fn step_adam(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, t: u64, grad_scale: f32) {
        //corrections for initial moment shifts (bias corrections)
        let t_f = t as f32;
        let b1t = b1.powf(t_f);
        let b2t = b2.powf(t_f);
        let corr1 = 1.0 - b1t;
        let corr2 = 1.0 - b2t;

        //update the weights W
        for i in 0..self.w.len() {
            //scale gradient(global clip)
            let g = self.gw[i] * grad_scale;
            //exponential means m an v
            self.mw[i] = b1 * self.mw[i] + (1.0 -b1) * g;
            self.vw[i] = b2 * self.vw[i] + (1.0 - b2) * (g * g);
            //biased estimates
            let m_hat = self.mw[i] / corr1.max(1e-8);
            let v_hat = self.vw[i] / corr2.max(1e-8);
            //step by parametrs
            self.w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
        //update bias b
        for i in 0..self.b.len() {
            let g = self.gb[i] * grad_scale;
            self.mb[i] = b1 * self.mb[i] + (1.0 -b1) * g;
            self.vb[i] = b2 * self.vb[i] + (1.0 - b2) * (g * g);
            let m_hat = self.mb[i] / corr1.max(1e-8);
            let v_hat = self.vb[i] / corr2.max(1e-8);
            self.b[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }

    //l2 norm of all layer gradients (for a clip)
    pub fn grad_l2_sum(&self) -> f32 {
        let mut s = 0.0;
        for g in &self.gw { s += g * g; }
        for g in &self.gb { s += g * g; }
        s
    }
    //scale gradients(если хотим «жёсткий» клип по компонентам — делаем тут; сейчас масштаб общий).
    pub fn scale_grads(&mut self, _scale: f32) { /* не нужен — общий scale передаём в step_adam */ }

    //saving parameters and adam states to a binary file ( appends to the shared buffer)
    fn write_to(&self, out: &mut Vec<u8>) {
        //we write the dimensions so that we can check them when loading
        out.extend_from_slice(&(self.in_dim as u32).to_le_bytes());
        out.extend_from_slice(&(self.out_dim as u32).to_le_bytes());
        //write weights/biases and moments as f32 arrays
        for v in &self.w {out.extend_from_slice(&v.to_le_bytes());}
        for v in &self.b {out.extend_from_slice(&v.to_le_bytes());}
        for v in &self.mw {out.extend_from_slice(&v.to_le_bytes());}
        for v in &self.vw {out.extend_from_slice(&v.to_le_bytes());}
        for v in &self.mb {out.extend_from_slice(&v.to_le_bytes());}
        for v in &self.vb {out.extend_from_slice(&v.to_le_bytes());}
    }




}