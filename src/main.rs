mod game;
mod snake;
mod food;
mod border;

use game::Game;

fn main() {
    let mut game = Game::new(640, 640);
    game.run();
}