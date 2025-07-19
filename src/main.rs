mod game;
mod snake;
mod food;
mod event_loop;
mod genome;
mod brain;
mod evolution;
mod game_input;
mod training;
mod utils;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.contains(&"--train".to_string()) {
        training::run_training(20, 20);
    } else {
        event_loop::run();
    }
}