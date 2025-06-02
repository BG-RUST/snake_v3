mod game;
mod snake;
mod food;
use crate::game::GameResult;
use winit::event_loop::EventLoop;

fn main() {
    let mut event_loop = EventLoop::new(); // ← один раз тут

    game::start(&mut event_loop);
}
