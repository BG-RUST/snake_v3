use rand::Rng;
use crate::snake::*;
use crate::food::*;
use crate::utils::*;
use crate::network::*;

pub const WIDTH: i32 = 20;
pub const HEIGHT: i32 = 20;

pub struct Game {
    pub width: i32,
    pub height: i32,
    pub snake: Snake,
    pub food: Food,
    pub score: u32,
}

impl Game {
    pub fn new() -> Game {
        let width = WIDTH;
        let height = HEIGHT;
        let start_x = width / 2;
        let start_y = height / 2;
        let head = Point { x: start_x, y: start_y };
        let snake = Snake::new(head, 3, Direction::Right);

        let mut game = Game {
            width,
            height,
            snake,
            food: Food { position: Point { x: 0, y: 0 } },
            score: 0,
        };
        game.spawn_food();
        game
    }

    pub fn spawn_food(&mut self) {
        loop {
            let p = random_point(self.width, self.height);
            if !p.is_in(&self.snake.body) {
                self.food.position = p;
                break;
            }
        }
    }

    /// Выполняет один шаг обновления игры.
    /// Возвращает (съедена ли еда, завершена ли игра)
    pub fn update(&mut self, new_dir: Direction) -> (f32, bool) {
        if new_dir != self.snake.direction.opposite() {
            self.snake.direction = new_dir;
        }

        let head = self.snake.head();
        let (dx, dy) = self.snake.direction.delta();
        let new_head = Point { x: head.x + dx, y: head.y + dy };

        // Проверка выхода за границы
        if new_head.x < 0 || new_head.x >= self.width || new_head.y < 0 || new_head.y >= self.height {
            println!("💀 Смерть: удар об стену");
            return (-1.0, true);
        }

        // Только эта проверка нужна!
        if new_head.is_in(&self.snake.body) {
            let tail = *self.snake.body.last().unwrap();
            let eat = new_head == self.food.position;

            if !(new_head == tail && !eat) {
                println!("💀 Смерть: столкновение с телом");
                return (-1.0, true);
            }
        }

        let eat = new_head == self.food.position;
        self.snake.move_forward(eat);

        if eat {
            self.score += 1;
            self.spawn_food();
            return (1.0, false);
        } else {
            return (-0.1, false);
        }
    }

    /*
    /// Выполняет один шаг обновления игры.
    /// Возвращает (съедена ли еда, завершена ли игра)
    pub fn update(&mut self, new_dir: Direction) -> (f32, bool) {
        if new_dir != self.snake.direction.opposite() {
            self.snake.direction = new_dir;
        }

        let head = self.snake.head();
        let (dx, dy) = self.snake.direction.delta();
        let new_head = Point { x: head.x + dx, y: head.y + dy };

        if new_head.x < 0 || new_head.x >= self.width || new_head.y < 0 || new_head.y >= self.height {
            return (-1.0, true);
        }

        if new_head.is_in(&self.snake.body) {
            let tail_index = self.snake.body.len() - 1;
            if !(new_head == self.snake.body[tail_index] && self.food.position != new_head) {
                return (-1.0, true);
            }
        }

        let eat = new_head == self.food.position;
        self.snake.move_forward(eat);

        if eat {
            self.score += 1;
            self.spawn_food();
            return (1.0, false);
        } else {
            return (-0.1, false);
        }
    }

*/
    /// Возвращает признаки (state) для нейросети
/*    pub fn get_state(&self) -> [f32; 19] {
        let mut state = [0.0; 19];
        let head = self.snake.head();
        let dir = self.snake.direction;

        let left_dir = dir.left();
        let right_dir = dir.right();
        let (dx_f, dy_f) = dir.delta();
        let (dx_l, dy_l) = left_dir.delta();
        let (dx_r, dy_r) = right_dir.delta();

        let front_point = Point { x: head.x + dx_f, y: head.y + dy_f };
        let left_point = Point { x: head.x - dx_l, y: head.y + dy_l };
        let right_point = Point { x: head.x + dx_r, y: head.y - dy_r };

        // [0..3] danger
        state[0] = is_danger(&front_point, &self.snake, self.width, self.height);
        state[1] = is_danger(&left_point, &self.snake, self.width, self.height);
        state[2] = is_danger(&right_point, &self.snake, self.width, self.height);

        // [3..7] direction
        state[3] = if dir == Direction::Up { 1.0 } else { 0.0 };
        state[4] = if dir == Direction::Down { 1.0 } else { 0.0 };
        state[5] = if dir == Direction::Left { 1.0 } else { 0.0 };
        state[6] = if dir == Direction::Right { 1.0 } else { 0.0 };

        // [7..11] food direction (bits)
        state[7] = if self.food.position.y < head.y { 1.0 } else { 0.0 };
        state[8] = if self.food.position.y > head.y { 1.0 } else { 0.0 };
        state[9] = if self.food.position.x < head.x { 1.0 } else { 0.0 };
        state[10] = if self.food.position.x > head.x { 1.0 } else { 0.0 };

        // [11] manhattan food distance (normalized)
        let dist = ((head.x - self.food.position.x).abs() + (head.y - self.food.position.y).abs()) as f32;
        state[11] = dist / (self.width + self.height) as f32;

        // [12] length ratio
        state[12] = self.snake.body.len() as f32 / (self.width * self.height) as f32;

        // [13..15] normalized distance to wall (max 5 tiles)
        state[13] = look_distance(&self.snake, self.width, self.height, dir);
        state[14] = look_distance(&self.snake, self.width, self.height, dir.left());
        state[15] = look_distance(&self.snake, self.width, self.height, dir.right());

        // [16..18] is food visible within 5 tiles in those directions
        state[16] = is_food_visible(&self.snake.head(), self.food.position, dir, 5);
        state[17] = is_food_visible(&self.snake.head(), self.food.position, dir.left(), 5);
        state[18] = is_food_visible(&self.snake.head(), self.food.position, dir.right(), 5);

        state
    }
}*/
    /// Возвращает признаки [0..12]:
    /// Для каждого из направлений: [0] расстояние до стены, [1] видит ли еду, [2] видит ли тело
    pub fn get_state(&self) -> [f32; 12] {
        let mut state = [0.0; 12];
        let head = self.snake.head();

        let directions = [
            Direction::Up,
            Direction::Down,
            Direction::Left,
            Direction::Right,
        ];

        for (i, dir) in directions.iter().enumerate() {
            state[i * 3 + 0] = look_for_wall(head, *dir, self.width, self.height) / 5.0;
            state[i * 3 + 1] = is_food_visible(head, self.food.position, *dir, 5);
            state[i * 3 + 2] = look_for_tail(head, *dir, &self.snake, 5);
        }

        state
    }

    fn look_distance(snake: &crate::snake::Snake, width: i32, height: i32, dir: Direction) -> f32 {
    let mut pos = snake.head();
    let (dx, dy) = dir.delta();
    for i in 1..=5 {
        pos.x += dx;
        pos.y += dy;
        if pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height || pos.is_in(&snake.body) {
            return i as f32 / 5.0;
        }
    }
    1.0
}

fn is_food_visible(head: &Point, food: Point, dir: Direction, max_dist: i32) -> f32 {
    let (dx, dy) = dir.delta();
    for i in 1..=max_dist {
        let pos = Point { x: head.x + dx * i, y: head.y + dy * i };
        if pos == food {
            return 1.0;
        }
    }
    0.0
}

/// Проверка опасности в заданной точке
fn is_danger(p: &Point, snake: &Snake, width: i32, height: i32) -> f32 {
    if p.x < 0 || p.x >= width || p.y < 0 || p.y >= height {
        return 1.0;
    } else if p.is_in(&snake.body) && *p != *snake.body.last().unwrap() {
        return 1.0;
    }
    0.0
}}

/// Расстояние до стены (максимум 5, нормализовано)
fn look_for_wall(mut pos: Point, dir: Direction, width: i32, height: i32) -> f32 {
    let (dx, dy) = dir.delta();
    for i in 1..=5 {
        pos.x += dx;
        pos.y += dy;
        if pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height {
            return i as f32;
        }
    }
    5.0
}

/// Видит ли еду в направлении (на расстоянии ≤ max_dist)
fn is_food_visible(head: Point, food: Point, dir: Direction, max_dist: i32) -> f32 {
    let (dx, dy) = dir.delta();
    for i in 1..=max_dist {
        let pos = Point { x: head.x + dx * i, y: head.y + dy * i };
        if pos == food {
            return 1.0;
        }
    }
    0.0
}

/// Есть ли тело змеи в этом направлении (в пределах max_dist)
fn look_for_tail(head: Point, dir: Direction, snake: &Snake, max_dist: i32) -> f32 {
    let (dx, dy) = dir.delta();
    for i in 1..=max_dist {
        let pos = Point { x: head.x + dx * i, y: head.y + dy * i };
        if pos.is_in(&snake.body) {
            return 1.0;
        }
    }
    0.0
}
