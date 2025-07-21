use crate::utils::relu;
use serde::{Serialize, Deserialize};
use rand::Rng;
use std::cmp::Ordering;

pub const INPUTS: usize = 38;
const H1: usize = 64;
const H2: usize = 32;
pub const OUTPUTS: usize = 3;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DenseLayer {
    pub weights: Vec<Vec<f32>>, // [нейрон][вход]
    pub biases: Vec<f32>,       // [нейрон]
}

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / (input_size as f32).sqrt()) * 0.5; // ограничим диапазон

        let weights = (0..output_size)
            .map(|_| {
                (0..input_size)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size]; // инициализируем bias нулями

        Self { weights, biases }
    }

    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(w_row, b)| {
                let sum: f32 = w_row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                relu(sum + b)
            })
            .collect()
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Brain {
    pub layer1: DenseLayer,
    pub layer2: DenseLayer,
    pub output: DenseLayer,
}

impl Brain {
    pub fn new_random() -> Self {
        Self {
            layer1: DenseLayer::new(INPUTS, H1),
            layer2: DenseLayer::new(H1, H2),
            output: DenseLayer::new(H2, OUTPUTS),
        }
    }

    pub fn forward_values(&self, input: &[f32; INPUTS]) -> [f32; OUTPUTS] {
        let h1 = self.layer1.forward(input);
        let h2 = self.layer2.forward(&h1);
        let mut out = [0.0; OUTPUTS];

        for (i, (w_row, b)) in self.output.weights.iter().zip(self.output.biases.iter()).enumerate() {
            out[i] = w_row.iter().zip(h2.iter()).map(|(w, x)| w * x).sum::<f32>() + b;
        }

        out
    }

    pub fn predict(&self, input: &[f32; INPUTS]) -> usize {
        let q_values = self.forward_values(input);

        if q_values.iter().any(|v| v.is_nan()) {
            eprintln!("⚠️ NaN в Q-значениях: {:?}", q_values);
        }

        q_values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    pub fn adjust_weights(&mut self, input: &[f32; INPUTS], target: &[f32; OUTPUTS], _action: usize, lr: f32) {
        // === FORWARD ===
        let h1: Vec<f32> = self.layer1.forward(input);   // [64]
        let h2: Vec<f32> = self.layer2.forward(&h1);     // [32]
        let q_values = self.forward_values(input);       // [3]

        // === dL/dOut ===
        let mut delta_output = [0.0; OUTPUTS];
        for i in 0..OUTPUTS {
            delta_output[i] = q_values[i] - target[i]; // MSE loss
        }

        // === dL/dH2 ===
        let mut delta_h2 = vec![0.0; H2];
        for j in 0..H2 {
            let mut sum = 0.0;
            for i in 0..OUTPUTS {
                sum += delta_output[i] * self.output.weights[i][j];
            }
            delta_h2[j] = if h2[j] > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // === dL/dH1 ===
        let mut delta_h1 = vec![0.0; H1];
        for k in 0..H1 {
            let mut sum = 0.0;
            for j in 0..H2 {
                sum += delta_h2[j] * self.layer2.weights[j][k];
            }
            delta_h1[k] = if h1[k] > 0.0 { sum } else { 0.0 }; // ReLU
        }

        // === UPDATE output ===
        for i in 0..OUTPUTS {
            for j in 0..H2 {
                //self.output.weights[i][j] -= lr * delta_output[i] * h2[j];
                let clipped = delta_output[i].clamp(-10.0, 10.0);
                self.output.weights[i][j] -= lr * clipped * h2[j];
            }
            self.output.biases[i] -= lr * delta_output[i];
        }

        // === UPDATE layer2 ===
        for j in 0..H2 {
            for k in 0..H1 {
                self.layer2.weights[j][k] -= lr * delta_h2[j] * h1[k];
            }
            self.layer2.biases[j] -= lr * delta_h2[j];
        }

        // === UPDATE layer1 ===
        for k in 0..H1 {
            for l in 0..INPUTS {
                self.layer1.weights[k][l] -= lr * delta_h1[k] * input[l];
            }
            self.layer1.biases[k] -= lr * delta_h1[k];
        }
    }


    pub fn to_model(&self) -> crate::db::DqnModel {
        crate::db::DqnModel {
            weights: vec![
                self.layer1.weights.clone(),
                self.layer2.weights.clone(),
                self.output.weights.clone(),
            ],
            biases: vec![
                self.layer1.biases.clone(),
                self.layer2.biases.clone(),
                self.output.biases.clone(),
            ],
        }
    }

    pub fn from_model(model: &crate::db::DqnModel) -> Self {
        Self {
            layer1: DenseLayer {
                weights: model.weights[0].clone(),
                biases: model.biases[0].clone(),
            },
            layer2: DenseLayer {
                weights: model.weights[1].clone(),
                biases: model.biases[1].clone(),
            },
            output: DenseLayer {
                weights: model.weights[2].clone(),
                biases: model.biases[2].clone(),
            },
        }
    }
}
