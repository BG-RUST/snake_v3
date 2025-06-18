use rand::Rng;

#[derive(Clone)]
pub struct Net {
    pub weights: Vec<Vec<f32>>,
}

impl Net {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        let hidden_weights = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let output_weights = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        Self {
            hidden_weights,
            output_weights,
        }
    }

    pub fn predict(&self, input: &[f32]) -> usize {
        let hidden: Vec<f32> = self.hidden_weights.iter()
            .map(|w| w.iter().zip(input).map(|(w, i)| w * i).sum::<f32>().tanh())
            .collect();

        self.output_weights.iter()
            .map(|w| w.iter().zip(&hidden).map(|(w,h)| w * h).sum::<f32>())
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}