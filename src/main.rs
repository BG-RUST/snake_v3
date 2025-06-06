mod game;
mod snake;

use game::Game;

fn main() {
    let mut game = Game::new(640, 640);
    game.run();
}