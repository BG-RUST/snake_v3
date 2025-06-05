use crate::network::Network;
use rand::prelude::*;
use std::fs::OpenOptions;
use std::io::Write;
use crate::snake::*;
use crate::food::*;

const DISCOUNT: f32 = 0.99;
const EPSILON_DECAY: f32 = 0.99;
const MIN_EPSILON: f32 = 0.1;

pub struct DQNAgent {
    pub network: Network,
    pub epsilon: f32,
    pub memory: Vec<(Vec<f32>, usize, f32, Vec<f32>, bool)>,
    pub rng: ThreadRng,
}

impl DQNAgent {
    pub fn new(input: usize, hidden: usize, output: usize, learning_rate: f32) -> Self {
        DQNAgent {
            network: Network::new(input, hidden, output, learning_rate),
            epsilon: 1.0,
            memory: Vec::new(),
            rng: rand::thread_rng(),
        }
    }

    pub fn new_with_network(network: Network) -> Self {
        Self {
            epsilon: 1.0,
            memory: Vec::new(),
            rng: rand::thread_rng(),
            network,
        }
    }

    pub fn select_action(&mut self, state: &[f32]) -> usize {
        if self.rng.r#gen::<f32>() < self.epsilon {
            self.rng.gen_range(0..self.network.output_size)
        } else {
            let q_values = self.network.forward(state).0;
            if q_values.is_empty() {
                return self.rng.gen_range(0..self.network.output_size);
            }

            q_values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    pub fn store_experience(&mut self, state: Vec<f32>, action: usize, reward: f32, next_state: Vec<f32>, done: bool) {
        self.memory.push((state, action, reward, next_state, done));
        const MAX_MEMORY: usize = 10000;
        if self.memory.len() >= MAX_MEMORY {
            self.memory.remove(0);
        }
    }

    pub fn train(&mut self) {
        use rand::seq::SliceRandom;
        let batch_size = 32;
        if self.memory.len() < batch_size {
            return;
        }

        let clip = |v: f32| v.clamp(-1.0, 1.0); // gradient clipping
        let batch = self.memory.choose_multiple(&mut self.rng, batch_size);

        for (state, action, reward, next_state, done) in batch {
            let (q_values, hidden) = self.network.forward(state);
            let next_q_values = if *done {
                vec![0.0; q_values.len()]
            } else {
                self.network.forward(next_state).0
            };

            let max_next_q = next_q_values.iter().cloned().fold(f32::MIN, f32::max);
            let target = *reward + DISCOUNT * max_next_q;
            let current_q = q_values[*action];
            let error = target - current_q;

            // 🔧 DEBUG (можно закомментировать потом)
            println!("Q[{}]={:.3}, Target={:.3}, Error={:.3}", action, current_q, target, error);

            // output -> hidden
            for j in 0..self.network.hidden_size {
                let grad = clip(error * hidden[j]);
                self.network.weights_ho[*action][j] += self.network.learning_rate * grad;
            }

            // hidden -> input
            for i in 0..self.network.hidden_size {
                let hidden_out = hidden[i];
                if hidden_out <= 0.0 {
                    continue;
                }

                let hidden_error = error * self.network.weights_ho[*action][i];
                for j in 0..self.network.input_size {
                    let grad = clip(hidden_error * state[j]);
                    self.network.weights_ih[i][j] += self.network.learning_rate * grad;
                }
            }
        }
    }


    pub fn log_episode(&self, episode: usize, score: u32) {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open("training_log.csv")
            .unwrap();
        writeln!(file, "{episode},{score},{:.3}", self.epsilon).unwrap();
    }
}

pub fn get_state(snake: &Snake, food: &Food) -> Vec<f32> {
    let head = snake.body[0];

    let point_left  = (head.0.wrapping_sub(1), head.1);
    let point_right = (head.0 + 1, head.1);
    let point_up    = (head.0, head.1.wrapping_sub(1));
    let point_down  = (head.0, head.1 + 1);

    let is_danger = |p: (u32, u32)| -> bool {
        p.0 >= crate::snake::BOARD_WIDTH || p.1 >= crate::snake::BOARD_HEIGHT || snake.body.contains(&p)
    };

    let danger_straight = match snake.direction {
        Direction::Up    => is_danger(point_up),
        Direction::Down  => is_danger(point_down),
        Direction::Left  => is_danger(point_left),
        Direction::Right => is_danger(point_right),
    };

    let danger_right = match snake.direction {
        Direction::Up    => is_danger(point_right),
        Direction::Down  => is_danger(point_left),
        Direction::Left  => is_danger(point_up),
        Direction::Right => is_danger(point_down),
    };

    let danger_left = match snake.direction {
        Direction::Up    => is_danger(point_left),
        Direction::Down  => is_danger(point_right),
        Direction::Left  => is_danger(point_down),
        Direction::Right => is_danger(point_up),
    };

    let dir_up    = snake.direction == Direction::Up;
    let dir_down  = snake.direction == Direction::Down;
    let dir_left  = snake.direction == Direction::Left;
    let dir_right = snake.direction == Direction::Right;

    let food_left  = food.position.0 < head.0;
    let food_right = food.position.0 > head.0;
    let food_up    = food.position.1 < head.1;
    let food_down  = food.position.1 > head.1;

    vec![
        danger_straight as u8 as f32,
        danger_right as u8 as f32,
        danger_left as u8 as f32,
        dir_left as u8 as f32,
        dir_right as u8 as f32,
        dir_up as u8 as f32,
        dir_down as u8 as f32,
        food_left as u8 as f32,
        food_right as u8 as f32,
        food_up as u8 as f32,
        food_down as u8 as f32,
    ]
}
