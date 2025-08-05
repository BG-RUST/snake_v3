mod snake;
mod utils;
mod food;
mod game;
mod network;
mod log;
mod db;
mod event_loop;
mod autodiff;
mod rl;

use std::env;

fn main(){
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("usage: {} -- run | --train | --best", args[0]);
        return;
    }
    let mode = args[1].as_str();
    match mode {
        "--run" => {
            let game = game::Game::new();
            event_loop::run_manual(game);
        },
        "--train" => {
            rl::train();
        },
        "--best" => {
            if let Err(err) = event_loop::run_best() {
                eprintln!("Error: {}", err);
            }
        },
        _ => {
            eprintln!("Неизвестный режим: {}. Используйте --run, --train или --best.", mode);
        }
    }
}