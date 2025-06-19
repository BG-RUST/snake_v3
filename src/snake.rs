// src/snake.rs
use crate::game::*;

#[derive(PartialEq, Clone, Copy)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

pub struct Snake {
    pub body: Vec<(u32, u32)>,
    pub next_dir: Direction,
}

impl Snake {
    pub fn new() -> Self {
        Self {
            body: vec![(10, 10)],
            next_dir: Direction::Right,
        }
    }

    pub fn change_dir(&mut self, dir: Direction) {
        // Предотвращаем разворот в обратную сторону
        match (self.next_dir, dir) {
            (Direction::Up, Direction::Down) |
            (Direction::Down, Direction::Up) |
            (Direction::Left, Direction::Right) |
            (Direction::Right, Direction::Left) => {},
            _ => self.next_dir = dir,
        }
    }

    pub fn update(&mut self) {
        let (head_x, head_y) = self.body[0];
        let new_head = match self.next_dir {
            Direction::Up => (head_x, head_y.saturating_sub(1)),
            Direction::Down => (head_x, head_y + 1),
            Direction::Left => (head_x.saturating_sub(1), head_y),
            Direction::Right => (head_x + 1, head_y),
        };

        self.body.insert(0, new_head);
        self.body.pop();
    }

    pub fn grow(&mut self) {
        let tail = *self.body.last().unwrap();
        self.body.push(tail);
    }

    pub fn draw(&self, frame: &mut [u8], cell_size: u32, screen_width: u32) {
        for &(x, y) in &self.body {
            draw_cell(frame, x, y, cell_size, screen_width, [0, 255, 0, 255]);
        }
    }
    pub fn is_colliding_with_self(&self) -> bool {
        if self.body.len() < 2 {
            return false;
        }
        let head = self.body[0];
        self.body[1..].contains(&head)
    }
}

