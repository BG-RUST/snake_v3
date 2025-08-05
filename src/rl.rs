// 💡 Переработанный train с рабочим автодиффом и сбором градиентов
use crate::autodiff::{Var, powi};
use crate::network::*;
use rand::Rng;
use crate::db::{save, load, SaveData};
use crate::log::{info, episode, reward_log};
use crate::game::Game;

#[derive(Clone)]
pub struct Transition {
    pub state: [f32; INPUT_SIZE],
    pub action: usize,
    pub reward: f32,
    pub next_state: [f32; INPUT_SIZE],
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: Vec<Transition>,
    capacity: usize,
    index: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            index: 0,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() < self.capacity {
            self.buffer.push(transition);
        } else {
            self.buffer[self.index] = transition;
        }
        self.index = (self.index + 1) % self.capacity;
    }

    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = rand::thread_rng();
        (0..batch_size)
            .map(|_| {
                let i = rng.gen_range(0..self.buffer.len());
                self.buffer[i].clone()
            })
            .collect()
    }

    pub fn is_ready(&self, batch_size: usize) -> bool {
        self.buffer.len() >= batch_size
    }
}

fn huber_loss(diff: Var, delta: f32) -> Var {
    //let abs = diff.clone().abs();
    let abs = Var::new(diff.value.abs());
    if abs.value < delta {
        //0.5 * powi(diff, 2)
        Var::new(0.5) * powi(diff, 2)
    } else {
        //delta * (abs - 0.5 * delta)
        Var::new(delta) * (abs - Var::new(0.5 * delta))
    }
}

pub fn train() {
    let mut net = Network::new_random();
    let mut target_net = net.clone();
    let mut epsilon = 1.0;
    let mut buffer = ReplayBuffer::new(100000);
    let lr = 0.001;
    let gamma = 0.99;
    let batch_size = 64;
    let mut best_score = 0;

    let mut model_loaded = false;

    if let Ok(save_data) = load("snake_model.json") {
        net = save_data.network;
        epsilon = save_data.epsilon;
        model_loaded = true;
        info("✅ Загружена модель из snake_model.json");
    } else {
        println!("⚠️ Модель не загружена. Используется новая случайная.");
    }

    for ep in 1..=20_000 {
        let mut game = Game::new();
        let mut state = game.get_state();
        let mut total_reward = 0.0;

        for _ in 0..1000 {
            let action = if rand::random::<f32>() < epsilon {
                rand::random::<usize>() % OUTPUT_SIZE
            } else {
                let (_, _, out) = net.forward_raw(&state);
                out.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            };

            let dir = match action {
                0 => game.snake.direction.left(),
                1 => game.snake.direction,
                2 => game.snake.direction.right(),
                _ => game.snake.direction,
            };

            let prev = game.snake.head().manhattan(game.food.position);
            let (base, done) = game.update(dir);
            let new = game.snake.head().manhattan(game.food.position);

            let reward = if done {
                -100.0
            } else if base > 0.0 {
                10.0
            } else if new < prev {
                0.2
            } else {
                -0.1
            };

            let next_state = game.get_state();
            buffer.push(Transition { state, action, reward, next_state, done });

            state = next_state;
            total_reward += reward;

            if done { break; }
        }

        if buffer.is_ready(batch_size) {
            let batch = buffer.sample(batch_size);
            let mut total_loss = Var::new(0.0);

            for trans in &batch {
                let pred = net.forward_raw(&trans.state).2;
                //let mut target_q = pred;
                let mut target_q = pred.clone();


                let next = target_net.forward_raw(&trans.next_state).2;
                let max_next_q = *next.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
                let target_val = if trans.done {
                    trans.reward
                } else {
                    trans.reward + gamma * max_next_q
                };
                target_q[trans.action] = target_val;

                let input_vars: Vec<Var> = trans.state.iter().map(|&x| Var::new(x)).collect();
                let output_vars = net.forward_autodiff(&input_vars);

                let mut loss = Var::new(0.0);
                for i in 0..OUTPUT_SIZE {
                    let diff = output_vars[i].clone() - Var::new(target_q[i]);
                    loss = loss + huber_loss(diff, 1.0);
                }

                total_loss = total_loss + loss / Var::new(OUTPUT_SIZE as f32);
            }

            total_loss = total_loss / Var::new(batch.len() as f32);

            if total_loss.value.is_finite() {
                total_loss.backward();
                net.apply_grads(lr);
                println!("📉 Loss: {:.6}", total_loss.value);
            }
        }

        target_net.soft_update(&net, 0.01);
        epsilon = (epsilon * 0.999).max(0.05);
        episode(ep, game.score, epsilon);

        if game.score > best_score {
            best_score = game.score;
            let _ = save(&SaveData { network: net.clone(), epsilon }, "snake_model.json");
            println!("✅ Новый лучший результат: {} (эпизод {})", best_score, ep);
        }

        //reward_log(ep, total_reward);
    }
}