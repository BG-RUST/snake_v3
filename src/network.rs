///network, backpropagation, Adam, save/load

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use crate::utils::*;

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
            *v = -k + 2.0 * k * u; // linearly stretched in [−k, k].
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
    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
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
    //load from a byte slice (returns the offset where reading ended
    fn read_from(&mut self, data: &[u8], mut off: usize) -> usize {
        let rd_u32 = |d: &[u8], o: &mut usize| -> u32 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&d[*o..*o+4]); *o += 4; u32::from_le_bytes(buf)
        };
        let rd_f32 = |d: &[u8], o: &mut usize| -> f32 {
            let mut buf = [0u8; 4];
            buf.copy_from_slice(&d[*o..*o+4]); *o += 4; f32::from_le_bytes(buf)
        };

        let in_dim = rd_u32(data, &mut off) as usize;
        let out_dim = rd_u32(data, &mut off) as usize;
        assert_eq!(in_dim, self.in_dim);
        assert_eq!(out_dim, self.out_dim);
        //read arrays
        for v in &mut self.w {
            *v = rd_f32(data, &mut off);
        }
        for v in &mut self.b { *v = rd_f32(data, &mut off); }
        for v in &mut self.mw { *v = rd_f32(data, &mut off); }
        for v in &mut self.vw { *v = rd_f32(data, &mut off); }
        for v in &mut self.mb { *v = rd_f32(data, &mut off); }
        for v in &mut self.vb { *v = rd_f32(data, &mut off); }
        //return new offset
        off
    }

    pub fn non_finite_in_params_or_grads(&self) -> bool {
        // Проверяем веса/смещения и их градиенты на finiteness.
        has_non_finite(&self.w) || has_non_finite(&self.b) ||
            has_non_finite(&self.gw) || has_non_finite(&self.gb)
    }
}

//relu layer withour parametrs; apply maz (0, z) elementwise(поєлементно). Keep a mask of active neurons
struct ReLU {
    mask: Vec<u8>, // 1 - pass gradient, 0 - mute
}

impl ReLU {
    //create with a mask of the required length
    fn new(size: usize) -> Self { Self { mask: vec![0; size] } }

    //forward pass: input Z, output A=max(0,Z); set the mask to Z>0
    fn forward(&mut self, z: &mut [f32]) {
        //we go through all components
        for i in 0..z.len(){
            if z[i] > 0.0 {
                self.mask[i] = 1;
            } else {
                self.mask[i] = 0;
                z[i] = 0.0;
            }
        }
    }

    //backward pass: dZ = dA
    fn backward(&self, da: &mut [f32]) {
        for i in 0..da.len() {
            if self.mask[i] == 0 { da[i] = 0.0; }
        }
    }
}

//full network [obs] -> Linear -> ReLU -> Linear -> ReLU -> Linear -> [Q(action)].
pub struct Net {
    pub din: usize, // input dimension
    pub h1: usize,  // first hidden layer
    pub h2: usize,  // second hidden layer
    pub dout: usize,// number of actions ( Q - outputs )

    l1: Linear, //First linear layer
    a1: ReLU,  // Relu after first layer
    l2: Linear,// second linear layer
    a2: ReLU,  // relu after second layer
    l3: Linear,//input linear layer

    pub t_adam: u64, //count adam step(from bias corrections)
}

impl Net {
    //network constructor: create layers and relu initialize weights
    pub fn new(din: usize, h1: usize, h2: usize, dout: usize, mut rng: LcgRng) -> Self{
        //build linear layers with the specified sizes
        let l1 = Linear::new(din, h1, &mut rng);
        let l2 = Linear::new(h1, h2, &mut rng);
        let l3 = Linear::new(h2, dout, &mut rng);
        //return the fully prepared network
        Self {
            din, h1, h2, dout,
            l1,
            a1: ReLU::new(h1),
            l2,
            a2: ReLU::new(h2),
            l3,
            t_adam: 0,
        }
    }

    pub fn grad_l2_sum_all(&self) -> f32 {
        // Складываем вклад каждого слоя (w и b).
        self.l1.grad_l2_sum() + self.l2.grad_l2_sum() + self.l3.grad_l2_sum()
    }
    /// Есть ли не-числа в параметрах/градиентах сети.
    pub fn non_finite_any(&self) -> bool {
        self.l1.non_finite_in_params_or_grads() ||
            self.l2.non_finite_in_params_or_grads() ||
            self.l3.non_finite_in_params_or_grads()
    }

    //reset gradients of all layers
    pub fn zero_grad(&mut self) {
        self.l1.zero_grad();
        self.l2.zero_grad();
        self.l3.zero_grad()
    }

    //forward pass on a single observation: return a vector of q-values of length dout
    pub fn forward(&mut self, x: &[f32]) -> Vec<f32> {
        // First linear: Z1 = X*W1 + b1.
        let mut z1 = self.l1.forward(x);

        // ReLU: A1 = max(0, Z1) with mask set.
        self.a1.forward(&mut z1);
        //secont linear: Z2 = A1*W2 + b2
        let mut z2 = self.l2.forward(&z1);

        // Second ReLU: A2 = max(0, Z2).
        self.a2.forward(&mut z2);

        //output layer without activation : Q = A2 * W3 + b3
        let q = self.l3.forward(&z2);
        q

    }

    //backward pass: to the input dQ by the output, we throw it into the depth and accumulate gradients in layyers
    pub fn backward_from_output_grad(&mut self, mut d_q: Vec<f32>) {
        //dA2 = dQ * W3^T (inside l3 backward will calculate dA2 and accunulate dW3 / db3
        let mut da2 = self.l3.backward(&d_q);
        //dZ2 = dA2 ⊙ 1{Z2>0} - implements relu backward
        self.a2.backward(&mut da2);
        // dA1 = dZ2 * W2^T.
        let mut da1 = self.l2.backward(&da2);
        // dZ1 = dA1 ⊙ 1{Z1>0}.
        self.a1.backward(&mut da1);
        // dX = dZ1 * W1^T — no returns is required, but backward will execute and accumulate gradients
        let _dx = self.l1.backward(&da1);
        //gradients ate already in the gw/gb fields of the layers
    }

    //Clip by L2 norm for all network gradients, return multiplier (if <1 -was clip (clip gradient))
    pub fn clip_grad_norm(&mut self, max_norm: f32) -> f32 {
        //sum the squares of the gradients of all parameters
        let mut s = 0.0f32;
        s += self.l1.grad_l2_sum();
        s += self.l2.grad_l2_sum();
        s += self.l3.grad_l2_sum();

        //"Taking the square root gives the L2 norm of the gradient vector.
        let norm = s.sqrt();
        //if the norm is less than or equal to the threshold, set the scale to 1
        if norm <= max_norm || norm == 0.0 { return 1.0; }
        //otherwise, scale so that norm becomes equal to max_norm
        max_norm / norm
    }

    //Adam optimizer step for all layers, taking into account global gradient scaling
    pub fn step_adam(&mut self, lr: f32, b1: f32, b2: f32, eps: f32, grad_scale: f32) {
        //increment the optomizers "time"
        self.t_adam += 1;
        //update each layer
        self.l1.step_adam(lr, b1, b2, eps, self.t_adam, grad_scale);
        self.l2.step_adam(lr, b1, b2, eps, self.t_adam, grad_scale);
        self.l3.step_adam(lr, b1, b2, eps, self.t_adam, grad_scale);
    }

    // «Мягкое» копирование параметров из online-сети в эту (target): θ ← (1−τ)θ + τ θ_online.
    pub fn soft_update_from(&mut self, online: &Net, tau: f32) {
        // helper for component-wise mixing
        fn mix(dst: &mut [f32], src_vals: &[f32], tau: f32) {
            // Проверяем одинаковую длину (в отладке упадём, в релизе — вырежется).
            debug_assert_eq!(dst.len(), src_vals.len());
            // Идём парой по dst и src_vals.
            for (d, &s) in dst.iter_mut().zip(src_vals.iter()) {
                // Пересчитываем значение dst: экспоненциальное сглаживание к src.
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

    //hard copy parametres (for initialization target = online)
    pub fn copy_from(&mut self, src: &Net) {
        self.l1.w.clone_from(&src.l1.w);
        self.l1.b.clone_from(&src.l1.b);
        self.l2.w.clone_from(&src.l2.w);
        self.l2.b.clone_from(&src.l2.b);
        self.l3.w.clone_from(&src.l3.w);
        self.l3.b.clone_from(&src.l3.b);
        //adam time dont touch
    }

    //serialize the network into a binary file (including adam states and t_adam)
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        //we collect everything into a buffer in memory
        let mut buf: Vec<u8> = Vec::new();
        //magic signature and version control compatibility
        buf.extend_from_slice(b"SNET");
        buf.extend_from_slice(&1u32.to_le_bytes());
        //Dimensions
        for v in [self.din, self.h1, self.h2, self.dout] {
            buf.extend_from_slice(&(v as u32).to_le_bytes());
        }
        //Adam time
        buf.extend_from_slice(&self.t_adam.to_le_bytes());
        //parametrs all layers
        self.l1.write_to(&mut buf);
        self.l2.write_to(&mut buf);
        self.l3.write_to(&mut buf);
        //we write to disk in one piece
        let mut f = File::create(path)?;
        f.write_all(&buf)?;
        Ok(())
    }

    //Load the network from a file (if the file is missing, we return Err, the calling code will decide what to do
    pub fn load(&mut self, path: &str) -> std::io::Result<()> {
        //read th enterie file into memory
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        //check the title and version
        if &buf[0..4] != b"SNET" { return Err(std::io::Error::new(std::io::ErrorKind::Other, "bad header")); }
        let mut off = 4;
        let mut rd_u32 = |o: &mut usize| -> u32 { let mut b=[0u8;4]; b.copy_from_slice(&buf[*o..*o+4]); *o+=4; u32::from_le_bytes(b) };
        let mut rd_u64 = |o: &mut usize| -> u64 { let mut b=[0u8;8]; b.copy_from_slice(&buf[*o..*o+8]); *o+=8; u64::from_le_bytes(b) };
        let _ver = rd_u32(&mut off);
        let din = rd_u32(&mut off) as usize;
        let h1 = rd_u32(&mut off) as usize;
        let h2 = rd_u32(&mut off) as usize;
        let dout = rd_u32(&mut off) as usize;
        if din != self.din || h1 !=  self.h1 || h2 != self.h2 || dout != self.dout {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, "shape mismatch"));
        }
        self.t_adam = rd_u64(&mut off);
        //read layer
        off = self.l1.read_from(&buf, off);
        off = self.l2.read_from(&buf, off);
        let _ = self.l3.read_from(&buf, off);
        Ok(())


    }
}