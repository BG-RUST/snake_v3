mod game;
mod snake;
mod food;
mod net;

use winit::{event_loop::EventLoop, window::WindowBuilder};
use winit::dpi::PhysicalSize;

fn main() {
    let mut event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rustrover Projects")
        .with_inner_size(PhysicalSize::new(800, 600))
        .build(&event_loop).unwrap();

    game::run_game_loop(window, event_loop);
}