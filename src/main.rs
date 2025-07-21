use std::env;

mod brain;
mod db;
mod event_loop;
mod game;
mod game_input;
mod replay_buffer;
mod train;
mod utils;
mod snake;
mod food;

use brain::Brain;
use db::load_checkpoint;
use event_loop::{run_brain_play, run_human_play};
use train::train;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.contains(&"--train".to_string()) {
        println!("🚀 Запуск обучения модели");
        let (brain, epsilon, start_episode) = if let Some(ckpt) = load_checkpoint("checkpoint.json") {
            (
                Brain::from_model(&ckpt.model),
                ckpt.meta.epsilon,
                ckpt.meta.episode + 1,
            )
        } else {
            (Brain::new_random(), 1.0, 0)
        };
        train(brain, epsilon, start_episode);
    } else if args.contains(&"--best".to_string()) {
        println!("🎯 Запуск лучшей модели");
        run_brain_play();
    } else if args.contains(&"--run".to_string()) {
        println!("🎮 Игра с клавиатуры");
        run_human_play();
    } else {
        println!("❓ Укажите аргумент запуска: --train | --best | --run");
    }
}
