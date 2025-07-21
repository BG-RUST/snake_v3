use crate::snake::*;
use crate::game::*;

pub struct GameInput;

impl GameInput {
    pub fn from_game(game: &Game, last_action: usize, _start_pos: Point, steps_since_eat: usize) -> [f32; 38] {
        let mut input = [0.0f32; 38];
        let head = game.snake().head();
        let food = game.food();
        let width = game.width() as i32;
        let height = game.height() as i32;
        let snake = game.snake();

        // 🍎 Расстояние до еды по x и y
        input[0] = (food.x - head.x) as f32 / width as f32;
        input[1] = (food.y - head.y) as f32 / height as f32;

        // 🧭 Текущее направление змейки
        match snake.direction() {
            Direction::Up => input[2] = 1.0,
            Direction::Down => input[3] = 1.0,
            Direction::Left => input[4] = 1.0,
            Direction::Right => input[5] = 1.0,
        }

        // 🔁 Последнее действие (one-hot)
        if last_action < 3 {
            input[6 + last_action] = 1.0;
        }

        // 👀 Обзор в 8 направлениях
        let directions = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ];

        for (i, (dx, dy)) in directions.iter().enumerate() {
            input[9 + i * 3 + 0] = Self::vision_wall(head, *dx, *dy, width, height);
            input[9 + i * 3 + 1] = Self::vision_body(head, *dx, *dy, width, height, snake);
            input[9 + i * 3 + 2] = Self::vision_food(head, *dx, *dy, food, width, height);
        }

        // 🧬 Доп. признаки
        input[34] = snake.body().len() as f32 / (width * height) as f32;  // относительная длина
        input[35] = (snake.body().len().saturating_sub(1)) as f32 / 30.0; // отъеденность
        input[36] = (steps_since_eat as f32).min(100.0) / 100.0;          // нормализованный голод
        input[37] = (head.x + head.y) as f32 / (width + height) as f32;   // положение головы

        input
    }

    fn vision_wall(head: Point, dx: i32, dy: i32, width: i32, height: i32) -> f32 {
        let max_dist = ((width * width + height * height) as f32).sqrt();
        let mut distance = 0.0;
        let (mut x, mut y) = (head.x as f32, head.y as f32);
        while x >= 0.0 && y >= 0.0 && x < width as f32 && y < height as f32 {
            x += dx as f32;
            y += dy as f32;
            distance += 1.0;
        }
        (1.0 - (distance / max_dist)).clamp(0.0, 1.0)
    }

    fn vision_body(head: Point, dx: i32, dy: i32, width: i32, height: i32, snake: &Snake) -> f32 {
        let max_dist = ((width * width + height * height) as f32).sqrt();
        let mut distance = 1.0;
        let (mut x, mut y) = (head.x + dx, head.y + dy);
        while x >= 0 && y >= 0 && x < width && y < height {
            if snake.body().contains(&Point { x, y }) {
                return (1.0 - (distance / max_dist)).clamp(0.0, 1.0);
            }
            x += dx;
            y += dy;
            distance += 1.0;
        }
        0.0
    }

    fn vision_food(head: Point, dx: i32, dy: i32, food: Point, width: i32, height: i32) -> f32 {
        let max_dist = ((width * width + height * height) as f32).sqrt();
        let mut distance = 1.0;
        let (mut x, mut y) = (head.x + dx, head.y + dy);
        while distance < max_dist {
            if x == food.x && y == food.y {
                return (1.0 - (distance / max_dist)).clamp(0.0, 1.0);
            }
            x += dx;
            y += dy;
            distance += 1.0;
        }
        0.0
    }
}
