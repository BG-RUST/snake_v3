mod game;
mod snake;
mod food;
mod border;

use game::Game;
use std::time::Duration;
use winit::event_loop::EventLoop;

fn main() {
    let mut event_loop = EventLoop::new(); // ← создаётся один раз

    loop {
        let mut game = Game::new(640, 640);
        game.run(&mut event_loop); // ← передаём ссылку
        println!("🔁 Перезапуск через 1 секунду...");
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}