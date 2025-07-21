use crate::snake::{Direction, Snake, Point};
use crate::food::Food;
use winit::event::VirtualKeyCode;

pub struct Game {
    width: usize,
    height: usize,
    snake: Snake,
    food: Food,
}

impl Game {
    pub fn new(width: usize, height: usize) -> Self {
        let snake = Snake::new(width / 2, height / 2);
        let food = Food::new_random(width, height);
        Self{width, height, snake, food}
    }

    /// one game step (called every frame)
    /// snake is moving
    /// if eat food  -> it grows and new food appears
    /// if crash the game start over
    pub fn update(&mut self) {
        self.snake.step();


        if self.snake.head() == self.food.pos {
            self.snake.grow();
            self.food = Food::new_random(self.width, self.height);
        }

        if self.snake.is_dead(self.width, self.height) {
            //restart
            self.snake = Snake::new(self.width / 2, self.height / 2);
            self.food = Food::new_random(self.width, self.height);
        }
    }

    ///set new snake direction
    pub fn set_direction(&mut self, dir: Direction) {
        self.snake.set_direction(dir);
    }

    ///converts key to direction
    pub fn key_to_direction(&self, key: VirtualKeyCode) -> Option<Direction> {
        match key {
            VirtualKeyCode::Up => Some(Direction::Up),
            VirtualKeyCode::Down => Some(Direction::Down),
            VirtualKeyCode::Left => Some(Direction::Left),
            VirtualKeyCode::Right => Some(Direction::Right),
            _ => None,
        }
    }

    ///access to food coordinates (used for rendering)
    pub fn food(&self) -> Point {
        self.food.pos
    }

    ///access to snake (head and body)
    pub fn snake(&self) -> &Snake {
        &self.snake
    }

    ///board size
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    pub fn set_action_index(&mut self, action: usize) {
        let dir = match action {
            1 => self.snake.direction().turn_left(),
            2 => self.snake.direction().turn_right(),
            _ => self.snake.direction(),
        };
        self.snake.set_direction(dir);
    }

    pub fn calc_reward(&self) -> f32 {
        if self.snake().is_dead(self.width, self.height) {
            -1.0
        } else if self.snake().head() == self.food() {
            2.0 // Увеличим награду
        } else {
            -0.005 // Поменьше штраф
        }
    }
}