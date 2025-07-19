use crate::snake::*;
use crate::game::*;

pub struct GameInput;

impl GameInput {
    pub fn from_game(game: &Game, last_action: usize, start_pos: Point, steps_since_eat: usize) -> [f32; 38] {
        let mut input = [0.0f32; 38];
        let head = game.snake().head();
        let food = game.food();
        let width = game.width();
        let height = game.height();
        let snake = game.snake();

        input[0] = (food.x as f32 - head.x as f32) / width as f32;
        input[1] = (food.y as f32 / height as f32) / width as f32;

        //current direction
        match snake.direction() {
            Direction::Up => input[2] = 1.0,
            Direction::Down => input[3] = 1.0,
            Direction::Left => input[4] = 1.0,
            Direction::Right => input[5] = 1.0,
        }

        //last action
        if last_action < 3 {
            input[6 + last_action] = 1.0;
        }

        //snake vision - 8 directions * 3 signs
        let directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ];

        for (i, (dx, dy)) in directions.iter().enumerate() {
            input[9 + i * 3 + 0] = Self::vision_wall(head, *dx, *dy, width, height);
            input[9 + i * 3 + 1] = Self::vision_body(head, *dx, *dy, width, height, snake);
            input[9 + i * 3 + 2] = Self::vision_food(head, *dx, *dy, food);
        }

        //snake len
        input[34] = snake.body().len() as f32 / (width * height) as f32;

        //amount food eaten
        input[35] = (snake.body().len().saturating_sub(1)) as f32 / 30.0;

        //HUNGRY
        input[36] = steps_since_eat as f32 / 100.0;

        //start position
        input[37] = ((start_pos.x + start_pos.y) as f32) / (width + height) as f32;

        input
    }

    ///distance to wall
    fn vision_wall(head: Point, dx: i32, dy: i32, width: usize, height: usize) -> f32 {
        let mut distance = 0;
        let (mut x, mut y) = (head.x as i32, head.y as i32);
        while x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
            x += dx;
            y += dy;
            distance += 1;
        }
        1.0 / distance as f32
    }

    ///distance to body snake
    fn vision_body(head: Point, dx: i32, dy: i32, width: usize, height: usize, snake: &Snake) -> f32 {
        let mut distance = 1;
        let (mut x, mut y) = (head.x as i32 + dx, head.y as i32 + dy);
        while x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
            if snake.body().contains(&Point { x: x as usize, y: y as usize }) {
                return 1.0 / distance as f32;
            }
            x += dx;
            y += dy;
            distance += 1;
        }
        0.0
    }

    //distance to food
    fn vision_food(head: Point, dx: i32, dy: i32, food: Point) -> f32 {
        let mut distance = 1;
        let (mut x, mut y) = (head.x as i32 + dx, head.y as i32 + dy);

        while x >= 0 && y >= 0 && x < 1000 && y < 1000 {
            if x as usize == food.x && y as usize == food.y {
                return 1.0 / distance as f32;
            }
            x += dx;
            y += dy;
            distance += 1;
        }

        0.0
    }
}