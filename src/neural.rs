
pub struct NeuralNet {
    pub weights: Vec<f32>,
}

impl NeuralNet {
    pub fn from_flat_weights(weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), 54);
        Self { weights }
    }

    pub fn forward(&self, inputs: &[f32; 6]) -> usize {
        let input_to_hidden = &self.weights[0..36];
        let hidden_to_output = &self.weights[36..];

        let mut hidden = [0.0f32; 6];
        for h in 0..6 {
            for i in 0..6 {
                hidden[h] += inputs[i] * input_to_hidden[h * 6 + i];
            }
            hidden[h] = hidden[h].max(0.0); //Relu
        }

        let mut outputs = [0.0f32; 3];
        for o in 0..3 {
            for h in 0..6 {
                outputs[o] += hidden[h] * hidden_to_output[o * 6 + h];
            }
        }

        outputs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(1) // по умолчанию впереж
    }
}