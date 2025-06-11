
use crate::neural::NeuralNet;
use serde::{Serialize, Deserialize};
use std::fs;

#[derive(Serialize, Deserialize, Clone)]
pub struct Genome {
    pub weights: Vec<f32>,
    pub fitness: f32,
}

impl Genome {
    pub fn new_random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = (0..54).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Genome {weights, fitness: 0.0}
    }

    pub fn to_network(&self) -> NeuralNet {
        NeuralNet::from_flat_weights(self.weights.clone())
    }

    pub fn save_to_file(&self, path: &str) {
        let json = serde_json::to_string_pretty(self).unwrap();
        fs::write(path, json).expect("Unable to save neural net");
    }
    pub fn load_from_file(path: &str) -> Option<Self> {
        let data = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&data).ok()
    }
}