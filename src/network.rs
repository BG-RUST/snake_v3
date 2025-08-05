use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::autodiff::{Var, relu, powi};

pub const INPUT_SIZE: usize = 12;
pub const HIDDEN1_SIZE: usize = 64;
pub const HIDDEN2_SIZE: usize = 32;
pub const OUTPUT_SIZE: usize = 3;

#[derive(Serialize, Deserialize, Clone)]
pub struct Network {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub w3: Vec<Vec<f32>>,
    pub b3: Vec<f32>,

    #[serde(skip)]
    pub grads: Option<Gradients>,
}

#[derive(Clone)]
pub struct Gradients {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub w3: Vec<Vec<f32>>,
    pub b3: Vec<f32>,
}

impl Network {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();

        let w1 = (0..HIDDEN1_SIZE).map(|_| {
            (0..INPUT_SIZE).map(|_| (rng.gen::<f32>() - 0.5) * 0.1).collect()
        }).collect();
        let b1 = vec![0.0; HIDDEN1_SIZE];

        let w2 = (0..HIDDEN2_SIZE).map(|_| {
            (0..HIDDEN1_SIZE).map(|_| (rng.gen::<f32>() - 0.5) * 0.1).collect()
        }).collect();
        let b2 = vec![0.0; HIDDEN2_SIZE];

        let w3 = (0..OUTPUT_SIZE).map(|_| {
            (0..HIDDEN2_SIZE).map(|_| (rng.gen::<f32>() - 0.5) * 0.1).collect()
        }).collect();
        let b3 = vec![0.0; OUTPUT_SIZE];

        Self {
            w1, b1, w2, b2, w3, b3,
            grads: None,
        }
    }

    pub fn forward_autodiff(&mut self, input: &[Var]) -> Vec<Var> {
        let mut w1_vars = vec![];
        let mut b1_vars = vec![];
        let mut w2_vars = vec![];
        let mut b2_vars = vec![];
        let mut w3_vars = vec![];
        let mut b3_vars = vec![];

        for row in &self.w1 {
            w1_vars.push(row.iter().map(|&v| Var::new(v)).collect::<Vec<_>>());
        }
        for &v in &self.b1 {
            b1_vars.push(Var::new(v));
        }
        for row in &self.w2 {
            w2_vars.push(row.iter().map(|&v| Var::new(v)).collect::<Vec<_>>());
        }
        for &v in &self.b2 {
            b2_vars.push(Var::new(v));
        }
        for row in &self.w3 {
            w3_vars.push(row.iter().map(|&v| Var::new(v)).collect::<Vec<_>>());
        }
        for &v in &self.b3 {
            b3_vars.push(Var::new(v));
        }

        let h1: Vec<Var> = w1_vars.iter().zip(&b1_vars).map(|(w, b)| {
            relu(dot_var(w, input) + b.clone())
        }).collect();

        let h2: Vec<Var> = w2_vars.iter().zip(&b2_vars).map(|(w, b)| {
            relu(dot_var(w, &h1) + b.clone())
        }).collect();

        let out: Vec<Var> = w3_vars.iter().zip(&b3_vars).map(|(w, b)| {
            dot_var(w, &h2) + b.clone()
        }).collect();

        let grads = Gradients {
            w1: w1_vars.iter().map(|row| row.iter().map(|v| v.grad()).collect()).collect(),
            b1: b1_vars.iter().map(|v| v.grad()).collect(),
            w2: w2_vars.iter().map(|row| row.iter().map(|v| v.grad()).collect()).collect(),
            b2: b2_vars.iter().map(|v| v.grad()).collect(),
            w3: w3_vars.iter().map(|row| row.iter().map(|v| v.grad()).collect()).collect(),
            b3: b3_vars.iter().map(|v| v.grad()).collect(),
        };

        self.grads = Some(grads);
        out
    }

    pub fn apply_grads(&mut self, lr: f32) {
        if let Some(grads) = &self.grads {
            for i in 0..HIDDEN1_SIZE {
                for j in 0..INPUT_SIZE {
                    self.w1[i][j] -= lr * grads.w1[i][j];
                }
                self.b1[i] -= lr * grads.b1[i];
            }

            for i in 0..HIDDEN2_SIZE {
                for j in 0..HIDDEN1_SIZE {
                    self.w2[i][j] -= lr * grads.w2[i][j];
                }
                self.b2[i] -= lr * grads.b2[i];
            }

            for i in 0..OUTPUT_SIZE {
                for j in 0..HIDDEN2_SIZE {
                    self.w3[i][j] -= lr * grads.w3[i][j];
                }
                self.b3[i] -= lr * grads.b3[i];
            }
        }
    }

    pub fn predict(&self, input: &[f32; INPUT_SIZE]) -> [f32; OUTPUT_SIZE] {
        self.forward_raw(input).2
    }

    pub fn forward_raw(&self, input: &[f32; INPUT_SIZE]) -> ([f32; HIDDEN1_SIZE], [f32; HIDDEN2_SIZE], [f32; OUTPUT_SIZE]) {
        let mut h1 = [0.0; HIDDEN1_SIZE];
        let mut h2 = [0.0; HIDDEN2_SIZE];
        let mut out = [0.0; OUTPUT_SIZE];

        for i in 0..HIDDEN1_SIZE {
            h1[i] = relu_scalar(dot(&self.w1[i], input) + self.b1[i]);
        }

        for i in 0..HIDDEN2_SIZE {
            h2[i] = relu_scalar(dot(&self.w2[i], &h1) + self.b2[i]);
        }

        for i in 0..OUTPUT_SIZE {
            out[i] = dot(&self.w3[i], &h2) + self.b3[i];
        }

        (h1, h2, out)
    }

    pub fn soft_update(&mut self, source: &Network, tau: f32) {
        for (w_t, w_s) in self.w1.iter_mut().zip(&source.w1) {
            for (v_t, v_s) in w_t.iter_mut().zip(w_s) {
                *v_t = *v_t * (1.0 - tau) + *v_s * tau;
            }
        }
        for (w_t, w_s) in self.w2.iter_mut().zip(&source.w2) {
            for (v_t, v_s) in w_t.iter_mut().zip(w_s) {
                *v_t = *v_t * (1.0 - tau) + *v_s * tau;
            }
        }
        for (w_t, w_s) in self.w3.iter_mut().zip(&source.w3) {
            for (v_t, v_s) in w_t.iter_mut().zip(w_s) {
                *v_t = *v_t * (1.0 - tau) + *v_s * tau;
            }
        }
        for (b_t, b_s) in self.b1.iter_mut().zip(&source.b1) {
            *b_t = *b_t * (1.0 - tau) + *b_s * tau;
        }
        for (b_t, b_s) in self.b2.iter_mut().zip(&source.b2) {
            *b_t = *b_t * (1.0 - tau) + *b_s * tau;
        }
        for (b_t, b_s) in self.b3.iter_mut().zip(&source.b3) {
            *b_t = *b_t * (1.0 - tau) + *b_s * tau;
        }
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}

fn dot_var(weights: &[Var], vars: &[Var]) -> Var {
    weights.iter().zip(vars.iter()).fold(Var::new(0.0), |acc, (w, v)| acc + w.clone() * v.clone())
}
