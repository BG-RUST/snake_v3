mod utils;       // RNG and misc helpers.
mod log;         // Simple logging.
mod db;          // CSV for episode results.
mod snake;       // Snake data structure.
mod food;        // Food.
mod game;        // Game logic & observation.
mod event_loop;  // Window/render for manual/AI preview.
mod network;     // Neural net.
mod dqn;         // DQN agent.

use std::env;
use crate::game::{Game, StepOutcome};
use crate::dqn::{DQNAgent, AgentConfig};

fn main() {
    // Parse flags after the program name.
    let args: Vec<String> = env::args().skip(1).collect();

    // Select mode.
    let mode = if args.contains(&"--best".to_string()) {
        "best"
    } else if args.contains(&"--train".to_string()) {
        "train"
    } else {
        "run"
    };

    // Board size (tweak if needed).
    let w = 24usize;
    let h = 16usize;

    match mode {
        // Manual play with arrows (no learning).
        "run" => {
            let game = Game::new(w, h);
            if let Err(e) = event_loop::run_manual(game) {
                eprintln!("fatal: {e}");
            }
        }

        // Preview the trained/best model in a window (no learning).
        "best" => {
            // Create game and agent. We set eps_start = eps_end = 0.05 for near-greedy play.
            let mut game = Game::new(w, h);
            let cfg = AgentConfig {
                obs_dim: game.observation_dim(),
                act_dim: 3,
                hidden: 64,
                buffer_capacity: 100_000,
                batch_size: 128,
                gamma: 0.99,
                lr: 1e-3,
                eps_start: 0.05,
                eps_end: 0.05,
                eps_decay_steps: 1,
                tau: 0.005,
                learn_start: 10_000,
                updates_per_step: 1,
                seed: 42,
            };
            let mut agent = DQNAgent::new(cfg);

            // Force epsilon to eps_end (in case a saved agent_state overwrote it).
            agent.on_step(u64::MAX / 2);

            // Open a window and let the agent act. No training happens here.
            if let Err(e) = event_loop::run_ai_preview(game, agent) {
                eprintln!("fatal: {e}");
            }
        }

        // Headless training loop (fast as possible).
        "train" => {
            let mut game = Game::new(w, h);
            let cfg = AgentConfig {
                obs_dim: game.observation_dim(),
                act_dim: 3,
                hidden: 64,
                buffer_capacity: 100_000,
                batch_size: 128,
                gamma: 0.99,
                lr: 1e-3,
                eps_start: 1.0,
                eps_end: 0.05,
                eps_decay_steps: 100_000,
                tau: 0.005,
                learn_start: 5_000,
                updates_per_step: 2,
                seed: 1234567,
            };
            let mut agent = DQNAgent::new(cfg);

            // Episode counters.
            let mut episode_idx: u64 = 0;
            let mut episode_return: f32 = 0.0;
            let mut episode_steps: u64 = 0;

            // We'll maintain our own global step counter for epsilon schedule.
            // (It doesn't strictly continue from a previously saved steps_done,
            // but that's fine for practical training.)
            let mut global_steps: u64 = 0;

            loop {
                // Observe.
                let obs = game.observe();
                // Act (eps-greedy).
                let a = agent.select_action(&obs);
                // Environment step.
                let StepOutcome { reward, done } = game.step_ai(a);
                // Next observation.
                let next_obs = game.observe();
                // Store transition.
                agent.remember(&obs, a, reward, &next_obs, done);
                // Learn (if buffer is warm enough).
                agent.maybe_learn();

                // Counters and epsilon schedule.
                episode_return += reward;
                episode_steps += 1;
                global_steps += 1;
                agent.on_step(global_steps);

                // Episode end -> log & reset.
                if done {
                    let _ = db::append_episode_result(
                        "results.csv",
                        episode_idx,
                        episode_return,
                        episode_steps,
                    );
                    log::info(&format!(
                        "EP {:5} | ret {:7.3} | steps {:4} | eps {:.3} | loss {:.4} | buffer {}",
                        episode_idx,
                        episode_return,
                        episode_steps,
                        agent.current_epsilon(),
                        agent.last_loss,          // public field
                        agent.replay_len(),       // assumes you have this getter in dqn.rs
                    ));
                    episode_idx += 1;
                    episode_return = 0.0;
                    episode_steps = 0;
                    game.reset();
                }

                // Periodic save.
                if global_steps % 5000 == 0 {
                    agent.save_all();
                }
            }
        }

        _ => {}
    }
}
