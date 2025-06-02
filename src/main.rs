mod game;
mod snake;
mod food;
mod network;
mod dqn;

//use crate::game::GameResult;
use winit::event_loop::EventLoop;

use crate::dqn::DQNAgent;
use crate::game::{start, GameResult};

fn main() {
    let mut agent = DQNAgent::new(4, 64, 4, 0.01);

    loop {
        let result = game::start(&mut agent);
        if result == GameResult::Exit {
            break;
        }
        println!("Игра завершена. Перезапуск...");
    }
}