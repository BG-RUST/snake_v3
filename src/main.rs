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
mod db;
use crate::event_loop::*;
use crate::{
    db::{init_db, get_best_individual},
    event_loop::run_best_individual,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.contains(&"--train".to_string()) {
        training::run_training(20, 20);
    } else if args.contains(&"--best".to_string()) {
        let conn = init_db("my_snake_ai.db");
        if let Some(best) = get_best_individual(&conn) {
            run_best_individual(best.into());
        } else {
            eprintln!("❌ Не удалось загрузить лучшую особь из базы данных.");
        }
    } else {
        event_loop::run(); // обычная ручная игра
    }
}