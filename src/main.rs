mod game;
mod snake;
mod food;
mod net;
mod ga;

use winit::{event_loop::EventLoop, window::WindowBuilder};
use winit::dpi::PhysicalSize;
use std::sync::{Arc, Mutex};
use std::thread;
use crate::ga::run_ga_training;
use crate::net::NeuralNet;

fn main() {
    // Создаем событийный цикл и окно для игры (как и раньше)
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake AI - Genetic Algorithm")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build(&event_loop)
        .unwrap();

    // Загружаем сохраненные веса нейронной сети, если файл существует, иначе создаем новую сеть
    let initial_net = if let Some(net) = NeuralNet::load_from_file("best_weights.txt") {
        println!("Загружены сохраненные веса нейросети из файла.");
        net
    } else {
        println!("Сохраненные веса не найдены, инициализируем новую нейросеть случайно.");
        // Создаем новую нейросеть с заданным числом входов, скрытых нейронов и выходов
        NeuralNet::new(net::INPUT_COUNT, net::HIDDEN_COUNT, net::OUTPUT_COUNT)
    };

    // Оборачиваем лучшую нейронную сеть в Arc<Mutex>, чтобы ее разделять между потоками (игровым и обучающим)
    let best_net = Arc::new(Mutex::new(initial_net));

    // Клонируем указатель на лучшую сеть для обучающего потока (генетического алгоритма)
    let best_net_for_ga = Arc::clone(&best_net);
    // Запускаем генетический алгоритм в фоновом потоке.
    // Он будет эволюционировать население нейросетей и обновлять лучшую сеть.
    thread::spawn(move || {
        run_ga_training(best_net_for_ga);
    });

    // Запускаем основной игровой цикл, передавая ему указатель на лучшую сеть
    game::run_game_loop(window, event_loop, best_net);
}
