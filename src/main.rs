mod game;
mod snake;
mod food;
mod network;
mod dqn;

use std::fs::File;
use std::io::BufReader;
use std::io::BufRead;
use winit::event_loop::EventLoop;

use crate::dqn::DQNAgent;
use crate::network::Network;
use crate::game::{start, GameResult};

fn main() {
    // 1. Создаём event loop один раз
    let mut event_loop = EventLoop::new();

    // 2. Пытаемся загрузить веса сети или создаём новую
    let network = Network::load_weights_csv("weights.csv")
        .unwrap_or_else(|_| Network::new(11, 64, 4, 0.001));

    let (epsilon, mut episode): (f32, usize) = load_agent_state_or_default("agent_state.csv");

    // 3. Создаём агента с этой сетью
    let mut agent = DQNAgent::new_with_network(network);
    agent.epsilon = epsilon;

    // 4. Основной цикл игры
    loop {
        if agent.network.is_valid() {
            if let Err(e) = agent.network.save_weights_csv("weights.csv") {
                eprintln!("Ошибка при сохранении весов: {}", e);
            }
        } else {
            eprintln!("⚠ Весовая матрица содержит NaN или inf. Пропуск сохранения.");
        }
        let (result, _) = start(&mut agent, &mut event_loop, episode);

        // 5. Сохраняем веса после каждого эпизода
        if let Err(e) = agent.network.save_weights_csv("weights.csv") {
            eprintln!("Ошибка при сохранении весов: {}", e);
        }
        if let Err(e) = save_agent_state("agent_state.csv", agent.epsilon, episode) {
            eprintln!("Ошибка при сохранении состояния агента: {}", e);
        }

        if result == GameResult::Exit {
            break;
        }

        println!("Игра завершена. Перезапуск...");
        agent.epsilon *= 0.999; // Коэффициент спада (можно 0.99 или 0.999)
        if agent.epsilon < 0.005 {
            agent.epsilon = 0.05; // Минимальный уровень ε
        }
        episode += 1;
    }
}

fn save_agent_state (path: &str, epsilon: f32, episode: usize) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = File::create(path)?;
    writeln!(file, "epsilon,episode")?;
    writeln!(file, "{:.6},{}", epsilon, episode)?;
    Ok(())
}

fn load_agent_state_or_default (path: &str) -> (f32, usize) {
    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        let mut lines = reader.lines().skip(1); // пропускаем заголовок
        if let Some (Ok(line)) = lines.next() {
            let parts: Vec<&str> = line.split(",").collect();
            if parts.len() == 2 {
                let epsilon = parts[0].parse().unwrap_or(1.0);
                let episode = parts[1].parse().unwrap_or(0);
                return (epsilon, episode);
            }
        }
    }
    (1.0, 0)
}