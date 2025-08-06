/*use crate::autodiff::*;
use crate::network::*;
use crate::db::{load, save, SaveData};
use crate::log::{episode, info};
use crate::game::Game;

use rand::Rng;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const NUM_THREADS: usize = 8;

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
        if transition.done {
            println!("📥 Transition с done=true добавлен в буфер");
        }

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
    let abs = abs(diff.clone());
    if abs.value < delta {
        Var::new(0.5) * powi(diff, 2)
    } else {
        Var::new(delta) * (abs - Var::new(0.5 * delta))
    }
}

pub fn train() {
    let mut net = Network::new_random();
    let mut target_net = net.clone();
    let mut epsilon = 1.0;
    let mut best_score = 0;
    let mut last_improve_ep = 0;
    let gamma = 0.99;
    let batch_size = 64;
    let lr = 0.01;
    let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100_000)));

    if let Ok(save_data) = load("snake_model.json") {
        net = Network::from_serializable(save_data.network);
        epsilon = save_data.epsilon;
        info("✅ Модель успешно загружена");
    }

    // Фоновая генерация опыта
    for _ in 0..NUM_THREADS {
        let buffer = Arc::clone(&buffer);
        thread::spawn(move || loop {
            let mut game = Game::new();
            let mut state = game.get_state();
            let mut steps_since_food = 0;

            for _ in 0..1000 {
                let action = if rand::random::<f32>() < epsilon {
                    rand::random::<usize>() % OUTPUT_SIZE
                } else {
                    let input_vars: Vec<Var> = state.iter().map(|&x| Var::new(x)).collect();
                    let output_vars = net.forward_autodiff(&input_vars);
                    let q_values: Vec<f32> = output_vars.iter().map(|v| v.value).collect();

                    let max_q = q_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let exp_q: Vec<f32> = q_values.iter().map(|&q| (q - max_q).exp()).collect();
                    let sum_exp_q: f32 = exp_q.iter().sum();
                    let probs: Vec<f32> = exp_q.iter().map(|&e| e / sum_exp_q).collect();

                    let mut r = rand::random::<f32>();
                    let mut chosen = 0;
                    for (i, &p) in probs.iter().enumerate() {
                        r -= p;
                        if r <= 0.0 {
                            chosen = i;
                            break;
                        }
                    }
                    chosen
                };

                let dir = match action {
                    0 => game.snake.direction.left(),
                    1 => game.snake.direction,
                    2 => game.snake.direction.right(),
                    _ => game.snake.direction,
                };

                let (base_reward, done) = game.update(dir);

                let mut reward = if done {
                    -10.0
                } else if base_reward > 0.0 {
                    steps_since_food = 0;
                    10.0
                } else {
                    steps_since_food += 1;
                    if steps_since_food >= 10 {
                        steps_since_food = 0;
                        -1.0
                    } else {
                        -0.01
                    }
                };

                let next_state = game.get_state();
                let transition = Transition {
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                };

                if let Ok(mut buf) = buffer.lock() {
                    buf.push(transition);
                }

                state = next_state;
                if done {
                    break;
                }
            }
        });
    }

    // Основной цикл обучения
    for ep in 1..=20_000 {
        thread::sleep(Duration::from_millis(10));

        let mut loss_val = 0.0;

        if let Ok(buf) = buffer.lock() {
            if buf.is_ready(batch_size) {
                let batch = buf.sample(batch_size);
                let mut total_loss = Var::new(0.0);

                for trans in &batch {
                    let input_vars: Vec<Var> = trans.state.iter().map(|&x| Var::new(x)).collect();
                    let output_vars = net.forward_autodiff(&input_vars);

                    let next_input_vars: Vec<Var> =
                        trans.next_state.iter().map(|&x| Var::new(x)).collect();
                    let next_q_main = net.forward_autodiff(&next_input_vars);
                    let best_next_action = next_q_main
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.value.partial_cmp(&b.1.value).unwrap())
                        .unwrap()
                        .0;

                    let next_q_target = target_net.forward_autodiff(&next_input_vars);
                    let target_q = next_q_target[best_next_action].value;

                    let target_val = if trans.done {
                        trans.reward
                    } else {
                        trans.reward + gamma * target_q
                    };

                    let mut loss = Var::new(0.0);
                    for i in 0..OUTPUT_SIZE {
                        let target = if i == trans.action {
                            Var::new(target_val)
                        } else {
                            Var::new(output_vars[i].value)
                        };
                        let diff = output_vars[i].clone() - target;
                        loss = loss + huber_loss(diff, 1.0);
                    }

                    total_loss = total_loss + loss / Var::new(OUTPUT_SIZE as f32);
                }

                total_loss = total_loss / Var::new(batch.len() as f32);
                loss_val = total_loss.value;

                total_loss.backward();

                for layer in [&mut net.w1, &mut net.w2, &mut net.w3] {
                    for row in layer.iter_mut() {
                        for var in row {
                            let g = var.grad();
                            var.set_grad(g / (1.0 + g.abs()));
                        }
                    }
                }

                net.apply_grads(lr);
            }
        }

        target_net.soft_update(&net, 0.01);
        epsilon = (epsilon * 0.9995).max(0.1);
        episode(ep, 0, epsilon);
        println!("📉 Loss: {:.6}", loss_val);

        if ep - last_improve_ep > 2000 {
            println!("🛑 Early stopping: нет улучшений за 2000 эпизодов.");
            break;
        }

        // Сохраняем модель каждый 1000 эпизод
        if ep % 1000 == 0 {
            let _ = save(
                &SaveData {
                    network: net.to_serializable(),
                    epsilon,
                },
                "snake_model.json",
            );
        }
    }
}
*/
 */

use crate::autodiff::*;
use crate::network::*;
use crate::db::{save, load, SaveData};
use crate::log::{info, episode};
use crate::game::Game;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

const NUM_THREADS: usize = 8;
const BATCH_SIZE: usize = 64;
const MAX_EPISODES: usize = 20000;
const LEARNING_RATE: f32 = 0.01;
const GAMMA: f32 = 0.99;

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
    let abs = abs(diff.clone());
    if abs.value < delta {
        Var::new(0.5) * powi(diff, 2)
    } else {
        Var::new(delta) * (abs - Var::new(0.5 * delta))
    }
}

pub fn train() {
    let net = Arc::new(Mutex::new(Network::new_random()));
    let target_net = Arc::new(Mutex::new(net.lock().unwrap().clone()));
    let buffer = Arc::new(Mutex::new(ReplayBuffer::new(100_000)));

    if let Ok(save_data) = load("snake_model.json") {
        *net.lock().unwrap() = Network::from_serializable(save_data.network);
        info("✅ Модель успешно загружена");
    } else {
        println!("⚠️ Модель не загружена. Используется новая случайная.");
    }

    for thread_id in 0..NUM_THREADS {
        let net = Arc::clone(&net);
        let target_net = Arc::clone(&target_net);
        let buffer = Arc::clone(&buffer);

        thread::spawn(move || {
            for ep in 0..MAX_EPISODES / NUM_THREADS {
                let mut game = Game::new();
                let mut state = game.get_state();

                for _ in 0..1000 {
                    let action = {
                        let net = net.lock().unwrap();
                        let input_vars: Vec<Var> = state.iter().map(|&x| Var::new(x)).collect();
                        let output_vars = net.forward_autodiff(&input_vars);
                        output_vars
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.value.partial_cmp(&b.1.value).unwrap())
                            .unwrap()
                            .0
                    };

                    let dir = match action {
                        0 => game.snake.direction.left(),
                        1 => game.snake.direction,
                        2 => game.snake.direction.right(),
                        _ => game.snake.direction,
                    };

                    let (base_reward, done) = game.update(dir);

                    let reward = if done {
                        -10.0
                    } else if base_reward > 0.0 {
                        10.0
                    } else {
                        -0.01
                    };

                    let next_state = game.get_state();
                    let transition = Transition {
                        state,
                        action,
                        reward,
                        next_state,
                        done,
                    };

                    buffer.lock().unwrap().push(transition);
                    state = next_state;

                    if done {
                        break;
                    }
                }
            }
        });
    }

    // Обучение в главном потоке
    for ep in 1..=MAX_EPISODES {
        thread::sleep(Duration::from_millis(10));

        let ready = buffer.lock().unwrap().is_ready(BATCH_SIZE);
        if !ready {
            continue;
        }

        let batch = buffer.lock().unwrap().sample(BATCH_SIZE);
        let mut total_loss = Var::new(0.0);

        for trans in &batch {
            let input_vars: Vec<Var> = trans.state.iter().map(|&x| Var::new(x)).collect();
            let output_vars = net.lock().unwrap().forward_autodiff(&input_vars);

            let next_input_vars: Vec<Var> = trans.next_state.iter().map(|&x| Var::new(x)).collect();
            let next_q_main = net.lock().unwrap().forward_autodiff(&next_input_vars);
            let best_next_action = next_q_main
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.value.partial_cmp(&b.1.value).unwrap())
                .unwrap()
                .0;

            let next_q_target = target_net.lock().unwrap().forward_autodiff(&next_input_vars);
            let target_q = next_q_target[best_next_action].value;

            let target_val = if trans.done {
                trans.reward
            } else {
                trans.reward + GAMMA * target_q
            };

            let mut loss = Var::new(0.0);
            for i in 0..OUTPUT_SIZE {
                let target = if i == trans.action {
                    Var::new(target_val)
                } else {
                    Var::new(output_vars[i].value)
                };
                let diff = output_vars[i].clone() - target;
                loss = loss + huber_loss(diff, 1.0);
            }
            total_loss = total_loss + loss / Var::new(OUTPUT_SIZE as f32);
        }

        total_loss = total_loss / Var::new(batch.len() as f32);

        if total_loss.value.is_finite() {
            total_loss.backward();
            net.lock().unwrap().apply_grads(LEARNING_RATE);
            println!("📉 Loss: {:.6}", total_loss.value);
        }

        // Обновление целевой сети
        target_net.lock().unwrap().soft_update(&net.lock().unwrap(), 0.01);
    }

    let _ = save(&SaveData {
        network: net.lock().unwrap().to_serializable(),
        epsilon: 0.1,
    }, "snake_model.json");
    println!("✅ Обучение завершено и модель сохранена");
}
