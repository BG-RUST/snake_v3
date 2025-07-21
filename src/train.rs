use crate::brain::{Brain, INPUTS, OUTPUTS};
use crate::game::Game;
use crate::game_input::GameInput;
use crate::replay_buffer::{ReplayBuffer, Transition};
use crate::db::*;

use rand::seq::SliceRandom;
use rand::Rng;
use crate::event_loop::*;

// 📐 Гиперпараметры
//const GRID_WIDTH: usize = 10;
//const GRID_HEIGHT: usize = 10;
const BATCH_SIZE: usize = 64;
const MEMORY_SIZE: usize = 10000;
const EPISODES: usize = 100000;
const MAX_STEPS: usize = 1000;
const GAMMA: f32 = 0.90;
const MIN_EPSILON: f32 = 0.05;
const EPSILON_DECAY: f32 = 0.9995;
const LR: f32 = 0.001;

pub fn train(mut brain: Brain, mut epsilon: f32, start_episode: usize) {
    let mut buffer = ReplayBuffer::new(MEMORY_SIZE);
    let mut rng = rand::thread_rng();

    let mut best_reward = f32::NEG_INFINITY;
    if let Some(best_ckpt) = load_checkpoint("best_model.json") {
        best_reward = best_ckpt.meta.average_reward;
        brain = Brain::from_model(&best_ckpt.model);
        epsilon = best_ckpt.meta.epsilon;
        println!("📦 Загружена лучшая модель: reward = {:.2}", best_reward);
    }

    for episode in start_episode..EPISODES {
        let mut game = Game::new(GRID_WIDTH, GRID_HEIGHT);
        let mut total_reward = 0.0;
        let mut steps_since_eat = 0;
        let start_pos = game.snake().head();

        for _ in 0..MAX_STEPS {
            let state = GameInput::from_game(&game, 0, start_pos, steps_since_eat);

            let action = if rng.r#gen::<f32>() < epsilon {
                rng.gen_range(0..OUTPUTS)
            } else {
                brain.predict(&state)
            };

            game.set_action_index(action);
            game.update();

            let reward = game.calc_reward();
            let done = game.snake().is_dead(GRID_WIDTH, GRID_HEIGHT);
            total_reward += reward;
            steps_since_eat += 1;

            if game.snake().head() == game.food() {
                steps_since_eat = 0;
            }

            let next_state = GameInput::from_game(&game, action, start_pos, steps_since_eat);

            buffer.push(Transition {
                state,
                action,
                reward,
                next_state,
                done,
            });

            if done {
                break;
            }

            if buffer.len() >= BATCH_SIZE {
                let batch = buffer.sample(BATCH_SIZE);
                for t in batch {
                    let q_values = brain.forward_values(&t.state);
                    let next_q = brain.forward_values(&t.next_state);
                    if q_values.iter().any(|v| !v.is_finite()) || next_q.iter().any(|v| !v.is_finite()) {
                        eprintln!("⚠️ Пропущен шаг: найдены NaN/inf в Q-значениях");
                        continue;
                    }
                    let max_next_q = next_q.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

                    let target = if t.done {
                        t.reward
                    } else {
                        t.reward + GAMMA * max_next_q
                    };

                    let mut target_vec = q_values;
                    target_vec[t.action] = target;

                    brain.adjust_weights(&t.state, &target_vec, t.action, LR);
                }
            }
        }

        // 🔁 Обновление лучшей модели
        if total_reward > best_reward {
            best_reward = total_reward;
            let best_checkpoint = DqnCheckpoint {
                meta: DqnMetadata {
                    episode,
                    epsilon,
                    average_reward: total_reward,
                },
                model: brain.to_model(),
            };
            save_checkpoint("best_model.json", &best_checkpoint);
            println!("💾 ✅ Новая лучшая модель: reward = {:.2}", total_reward);
        }

        epsilon = (epsilon * EPSILON_DECAY).max(MIN_EPSILON);

        if episode % 5 == 0 {
            println!(
                "📦 Episode: {} | Reward: {:.2} | Epsilon: {:.3}",
                episode, total_reward, epsilon
            );
        }
    }
}
