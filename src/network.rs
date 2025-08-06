use rand::Rng;
use crate::autodiff::{Var, relu, dot_var};
use crate::db::*;
use serde::{Deserialize, Serialize};


pub const INPUT_SIZE: usize = 12;
pub const HIDDEN1_SIZE: usize = 64;
pub const HIDDEN2_SIZE: usize = 32;
pub const OUTPUT_SIZE: usize = 3;

#[derive(Serialize, Deserialize, Clone)]
pub struct SerializableNetwork {
    pub w1: Vec<Vec<f32>>,
    pub b1: Vec<f32>,
    pub w2: Vec<Vec<f32>>,
    pub b2: Vec<f32>,
    pub w3: Vec<Vec<f32>>,
    pub b3: Vec<f32>,
}
#[derive(Clone)]
pub struct Network {
    pub w1: Vec<Vec<Var>>,
    pub b1: Vec<Var>,
    pub w2: Vec<Vec<Var>>,
    pub b2: Vec<Var>,
    pub w3: Vec<Vec<Var>>,
    pub b3: Vec<Var>,
}

impl Network {
    pub fn new_random() -> Self {
        let mut rng = rand::thread_rng();

        let w1 = (0..HIDDEN1_SIZE).map(|_| {
            (0..INPUT_SIZE).map(|_| Var::new((rng.gen::<f32>() - 0.5) * 0.1)).collect()
        }).collect();
        let b1 = (0..HIDDEN1_SIZE).map(|_| Var::new(0.0)).collect();

        let w2 = (0..HIDDEN2_SIZE).map(|_| {
            (0..HIDDEN1_SIZE).map(|_| Var::new((rng.gen::<f32>() - 0.5) * 0.1)).collect()
        }).collect();
        let b2 = (0..HIDDEN2_SIZE).map(|_| Var::new(0.0)).collect();

        let w3 = (0..OUTPUT_SIZE).map(|_| {
            (0..HIDDEN2_SIZE).map(|_| Var::new((rng.gen::<f32>() - 0.5) * 0.1)).collect()
        }).collect();
        let b3 = (0..OUTPUT_SIZE).map(|_| Var::new(0.0)).collect();

        Self { w1, b1, w2, b2, w3, b3 }
    }

    pub fn to_serializable(&self) -> SerializableNetwork {
        SerializableNetwork {
            w1: self.w1.iter().map(|row| row.iter().map(|v| v.value).collect()).collect(),
            b1: self.b1.iter().map(|v| v.value).collect(),
            w2: self.w2.iter().map(|row| row.iter().map(|v| v.value).collect()).collect(),
            b2: self.b2.iter().map(|v| v.value).collect(),
            w3: self.w3.iter().map(|row| row.iter().map(|v| v.value).collect()).collect(),
            b3: self.b3.iter().map(|v| v.value).collect(),
        }
    }

    pub fn from_serializable(s: SerializableNetwork) -> Self {
        Self {
            w1: s.w1.into_iter().map(|row| row.into_iter().map(Var::new).collect()).collect(),
            b1: s.b1.into_iter().map(Var::new).collect(),
            w2: s.w2.into_iter().map(|row| row.into_iter().map(Var::new).collect()).collect(),
            b2: s.b2.into_iter().map(Var::new).collect(),
            w3: s.w3.into_iter().map(|row| row.into_iter().map(Var::new).collect()).collect(),
            b3: s.b3.into_iter().map(Var::new).collect(),
        }
    }

    pub fn forward_autodiff(&self, input: &[Var]) -> Vec<Var> {
        let h1: Vec<Var> = self.w1.iter().zip(&self.b1).map(|(w, b)| relu(dot_var(w, input) + b.clone())).collect();
        let h2: Vec<Var> = self.w2.iter().zip(&self.b2).map(|(w, b)| relu(dot_var(w, &h1) + b.clone())).collect();
        let out: Vec<Var> = self.w3.iter().zip(&self.b3).map(|(w, b)| dot_var(w, &h2) + b.clone()).collect();
        out
    }

    pub fn apply_grads(&mut self, lr: f32) {
        for i in 0..HIDDEN1_SIZE {
            for j in 0..INPUT_SIZE {
                let grad = self.w1[i][j].grad();
                self.w1[i][j].value -= lr * grad;
            }
            let grad = self.b1[i].grad();
            self.b1[i].value -= lr * grad;
        }

        for i in 0..HIDDEN2_SIZE {
            for j in 0..HIDDEN1_SIZE {
                let grad = self.w2[i][j].grad();
                self.w2[i][j].value -= lr * grad;
            }
            let grad = self.b2[i].grad();
            self.b2[i].value -= lr * grad;
        }

        for i in 0..OUTPUT_SIZE {
            for j in 0..HIDDEN2_SIZE {
                let grad = self.w3[i][j].grad();
                self.w3[i][j].value -= lr * grad;
            }
            let grad = self.b3[i].grad();
            self.b3[i].value -= lr * grad;
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
            h1[i] = relu_scalar(dot(&self.w1[i], input) + self.b1[i].value);
        }

        for i in 0..HIDDEN2_SIZE {
            h2[i] = relu_scalar(dot(&self.w2[i], &h1) + self.b2[i].value);
        }

        for i in 0..OUTPUT_SIZE {
            out[i] = dot(&self.w3[i], &h2) + self.b3[i].value;
        }

        (h1, h2, out)
    }

    pub fn soft_update(&mut self, source: &Network, tau: f32) {
        for (w_t, w_s) in self.w1.iter_mut().zip(&source.w1) {
            for (v_t, v_s) in w_t.iter_mut().zip(w_s) {
                v_t.value = v_t.value * (1.0 - tau) + v_s.value * tau;
            }
        }
        for (w_t, w_s) in self.w2.iter_mut().zip(&source.w2) {
            for (v_t, v_s) in w_t.iter_mut().zip(w_s) {
                v_t.value = v_t.value * (1.0 - tau) + v_s.value * tau;
            }
        }
        for (w_t, w_s) in self.w3.iter_mut().zip(&source.w3) {
            for (v_t, v_s) in w_t.iter_mut().zip(w_s) {
                v_t.value = v_t.value * (1.0 - tau) + v_s.value * tau;
            }
        }
        for (b_t, b_s) in self.b1.iter_mut().zip(&source.b1) {
            b_t.value = b_t.value * (1.0 - tau) + b_s.value * tau;
        }
        for (b_t, b_s) in self.b2.iter_mut().zip(&source.b2) {
            b_t.value = b_t.value * (1.0 - tau) + b_s.value * tau;
        }
        for (b_t, b_s) in self.b3.iter_mut().zip(&source.b3) {
            b_t.value = b_t.value * (1.0 - tau) + b_s.value * tau;
        }
    }
}

fn dot(a: &[Var], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x.value * y).sum()
}

fn relu_scalar(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.0 }
}
