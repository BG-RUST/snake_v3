use rand::Rng;
use crate::brain::*;

/// Description of one individual (individual = set of weights = neural network)
#[derive(Clone)]
pub struct Genome {
    pub weights: Vec<f32>,
}

impl Genome {
    //create random individual
    pub fn random() -> Self {
        let mut rng = rand::thread_rng();
        let total = Brain::total_weights();
        let weights = (0..total)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Self { weights }
    }

    ///crossover under two parrent = new child
    pub fn crossover(parent_a: &Genome, parent_b: &Genome) -> Self {
        let mut rng = rand::thread_rng();
        let weights = parent_a.weights.iter().zip(&parent_b.weights)
            .map(|(a, b)| if rng.gen_bool(0.5) { *a } else { *b })
            .collect();

        Self { weights }
    }

    ///mutation
    pub fn mutate(&mut self, rate: f32, magnitude: f32) {
        let mut rng = rand::thread_rng();
        for w in &mut self.weights {
            if rng.r#gen::<f32>() < rate {
                *w += rng.gen_range(-magnitude..magnitude);
                *w = w.clamp(-1.5, 1.5); // ограничиваем диапазон весов
            }
        }
    }

    ///calculates the neural network output for a given input (via Brain)
    pub fn decide(&self, input: &[f32; INPUTS]) -> usize {
        let brain = Brain::new(self.weights.clone());
        brain.forward(input)
    }
}