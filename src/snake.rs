use crate::game::draw_cell;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Up, Down, Left, Right,
}
pub struct Snake {
    pub body: Vec<(u32, u32)>,
    pub dir: Direction,
}

impl Snake {
    pub fn new() -> Self {
        Self {
            body: vec![(5, 5), (4, 5), (3, 5)],
            dir: Direction::Right,
        }
    }

    pub fn draw(&self, frame: &mut [u8], cell_size: u32, screen_width: u32) {
        for &(x, y) in &self.body {
            crate::game::draw_cell(frame, x, y, cell_size, screen_width, [0, 255, 0, 255]);
        }
    }
gi
    pub fn update(&mut self) {
        let (head_x, head_y) = self.body[0];
        let new_head = match self.dir {
            Direction::Up => (head_x, head_y - 1),
            Direction::Down => (head_x, head_y + 1),
            Direction::Left => (head_x - 1, head_y),
            Direction::Right => (head_x + 1, head_y),
        };

        self.body.insert(0, new_head);
        self.body.pop();
    }

    pub fn grow(&mut self) {
        let tail = *self.body.last().unwrap();
        self.body.push(tail);
    }

    pub fn change_dir(&mut self, new_dir: Direction) {
        if (self.dir == Direction::Up && new_dir != Direction::Right)
            || (self.dir == Direction::Right && new_dir != Direction::Left)
            || (self.dir == Direction::Up && new_dir != Direction::Down)
            || (self.dir == Direction::Down && new_dir != Direction::Up)
        {
            self.dir = new_dir;
        }
    }
}