use winit::event_loop as other_event_loop;

mod game;
mod snake;
mod food;
mod event_loop;

fn main() {
    event_loop::run();
}