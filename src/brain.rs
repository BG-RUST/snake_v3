use crate::game_input::GameInput;
use crate::utils::*;

pub struct Brain {
    pub weights: Vec<f32>,
}

pub const INPUTS: usize = 38;
const H1: usize = 64;
const H2: usize = 32;
const OUTPUTS: usize = 3;

impl Brain {
    pub fn new(weights: Vec<f32>) -> Self {
        assert_eq!(weights.len(), Brain::total_weights(), "Весов должно быть ровно столько, сколько требует сеть");
        Self { weights }
    }
    pub fn forward(&self, input: &[f32; INPUTS]) -> usize {
        //offset(indicies) within the overal weight vector
        let mut offset = 0;
        //first hidden layer
        let mut layer1 = [0.0f32; H1];
        for i in 0..H1 {
            let mut sum = 0.0;
            for j in 0..INPUTS {
                sum += input[j] * self.weights[offset + i * INPUTS + j];
            }
            offset += 1;
            layer1[i] = relu(sum);
        }

        //second hidden layer
        let mut layer2 = [0.0f32; H2];
        for i in 0..H2 {
            let mut sum = 0.0;
            for j in 0..H1 {
                sum += layer1[j] * self.weights[offset + i * H1 + j];
            }
            offset += H1;
            sum += self.weights[offset];
            offset += 1;
            layer2[i] = relu(sum);
        }

        //output layer
        let mut output = [0.0f32; OUTPUTS];
        for i in 0..OUTPUTS {
            let mut sum = 0.0;
            for j in 0..H2 {
                sum += layer2[j] * self.weights[offset + i * H2 + j];
            }
            offset += H2;
            sum += self.weights[offset];
            offset += 1;
            output[i] = sum;
        }

        //return idex neural with max meaning
        let mut best = 0;
        for i in 1..OUTPUTS {
            if output[i] > output[best] {
                best = i;
            }
        }
        best
    }
    ///return the total number pf weights required by the network
    pub fn total_weights() -> usize {
        INPUTS * H1 + H1 + H1 * H2 + H2 + H2 * OUTPUTS + OUTPUTS
    }
}