mod event_loop;
mod snake;
mod food;
mod utils;
mod log;
mod db;
mod network;
mod dqn;
mod game;
use std::env;


fn main() {
    //read arguments after --
    let args: Vec<String> = env::args().skip(1).collect();

    let mode = if args.contains(&"--best".to_string()) {
        "best"
    } else if args.contains(&"--train".to_string()) {
        "train"
    } else {
        "run"
    };

    //board size
    let grid_w = 24usize;
    let grid_h = 16usize;

    match mode {
        "run" => {
            let game = game::Game::new(grid_w, grid_h);
            if let Err(e) = event_loop::run_manual(game) {
                eprintln!("fatal: {e}");
            }
        }
        "best" => {
            eprintln!("")
        }
        "train" => {
            eprintln!("")
        }
        _ => {}
    }
}