pub const BOARD_WIDTH: u32 = 20;
pub const BOARD_HEIGHT: u32 = 20;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

pub struct Snake {
    pub body: Vec<(u32, u32)>,
    pub direction: Direction,
}

impl Snake {
    pub fn new() -> Self {
        let start_x = BOARD_WIDTH / 2;
        let start_y = BOARD_HEIGHT / 2;

        Snake {
            body: vec![(start_x, start_y)],
            direction: Direction::Right,
        }
    }

    pub fn change_direction(&mut self, new_dir: Direction) {
        let opposite = match self.direction {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        };
        if new_dir == opposite && self.body.len() > 1 {
            return;
        }
        self.direction = new_dir;
    }

    pub fn move_forvard(&mut self, new_head: (u32, u32), grow: bool) {
        self.body.insert(0, new_head);
        if !grow {
            self.body.pop();
        }
    }

    pub fn cheack_self_collision(&self) -> bool {
        let head = self.body[0];
        self.body[1..].contains(&head)
    }
}