mod game;
mod snake;
mod food;
mod net;
mod ga;
mod web;

use winit::{event_loop::EventLoop, window::WindowBuilder};
use winit::dpi::PhysicalSize;
use std::sync::{Arc, Mutex};
use std::thread;
use std::fs;

use crate::ga::run_ga_training;
use crate::net::NeuralNet;

fn main() {
    // Очищаем старый лог
    if fs::remove_file("fitness_log.csv").is_ok() {
        println!("Старый файл fitness_log.csv удалён.");
    }

    // Создаём окно
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake AI - Genetic Algorithm")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    // Загружаем или создаём нейросеть
    let initial_net = if let Some(net) = NeuralNet::load_from_file("best_weights.txt") {
        println!("Загружены сохранённые веса нейросети.");
        net
    } else {
        println!("Файл весов не найден. Инициализация новой нейросети.");
        NeuralNet::new(net::INPUT_COUNT, net::HIDDEN_COUNT, net::OUTPUT_COUNT)
    };

    let best_net = Arc::new(Mutex::new(initial_net));
    let best_net_for_training = Arc::clone(&best_net);

    // ✅ Запускаем веб-сервер
    thread::spawn(|| {
        tokio::runtime::Runtime::new().unwrap().block_on(web::start_web_server());
    });

    // ✅ Запускаем обучение
    thread::spawn(move || {
        run_ga_training(best_net_for_training);
    });

    // Запускаем игру
    game::run_game_loop(window, event_loop, best_net);
}
