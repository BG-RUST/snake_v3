/*use winit::event::VirtualKeyCode::P;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

///snake struct:
/// - body (vector points from head to tail)
/// - current direction
/// - flag - do we need to grow
pub struct Snake{
    body: Vec<Point>,
    dir: Direction,
    grow: bool,
}

impl Snake {
    ///create new snake with start pozition
    pub fn new(x: usize, y: usize) -> Self {
        Self {
            body: vec![Point { x, y }],
            dir: Direction::Right,
            grow: false,
        }
    }

    ///return head point
    pub fn head(&self) -> Point {
        self.body[0]
    }

    ///move snake on one cell in current direction
    pub fn step(&mut self) {
        let mut new_head = self.head();

        //change coordinates depending on direction
        match self.dir {
            Direction::Up => {
                if new_head.y > 0 {
                    new_head.y -= 1;
                }
            }
            Direction::Down => {
                new_head.y += 1;
            }
            Direction::Left => {
                if new_head.x > 0 {
                    new_head.x -= 1;
                }
            }
            Direction::Right => {
                new_head.x += 1;
            }
        }

        //insert head in the body begining
        self.body.insert(0, new_head);

        //if not grow - delete tail(moving without grow)
        if !self.grow {
            self.body.pop();
        } else {
            self.grow = false;
        }
    }

    ///change new direction if this not opposite
    pub fn set_direction(&mut self, dir: Direction) {
        let opposite = match self.dir {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        };

        if dir != opposite {
            self.dir = dir;
        }
    }

    pub fn direction(&self) -> Direction {
        self.dir
    }

    ///tell the snake that it should grow up
    pub fn grow(&mut self) {
        self.grow = true;
    }

    ///death check (hit at wall or herself)
    pub fn is_dead(&self, width: usize, height: usize) -> bool {
        let head = self.head();

        // 1. Столкновение со стеной
        if head.x >= width || head.y >= height || head.x < 0 || head.y < 0 {
            return true;
        }

        // 2. Столкновение с телом
        self.body[1..].contains(&head)
    }
    ///access body
    pub fn body(&self) -> &Vec<Point> {
        &self.body
    }
}

impl Direction {
    pub fn turn_left(&self) -> Self {
        match self {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }

    pub fn turn_right(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }
}

 */
use winit::event::VirtualKeyCode;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

pub struct Snake {
    body: Vec<Point>,
    dir: Direction,
    grow: bool,
}

impl Snake {
    pub fn new(x: usize, y: usize) -> Self {
        Self {
            body: vec![Point {
                x: x as i32,
                y: y as i32,
            }],
            dir: Direction::Right,
            grow: false,
        }
    }

    pub fn head(&self) -> Point {
        self.body[0]
    }

    pub fn step(&mut self) {
        let mut new_head = self.head();

        match self.dir {
            Direction::Up => new_head.y -= 1,
            Direction::Down => new_head.y += 1,
            Direction::Left => new_head.x -= 1,
            Direction::Right => new_head.x += 1,
        }

        self.body.insert(0, new_head);

        if !self.grow {
            self.body.pop();
        } else {
            self.grow = false;
        }
    }

    pub fn set_direction(&mut self, dir: Direction) {
        let opposite = match self.dir {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        };

        if dir != opposite {
            self.dir = dir;
        }
    }

    pub fn direction(&self) -> Direction {
        self.dir
    }

    pub fn grow(&mut self) {
        self.grow = true;
    }

    pub fn is_dead(&self, width: usize, height: usize) -> bool {
        let head = self.head();

        if head.x < 0 || head.y < 0 || head.x >= width as i32 || head.y >= height as i32 {
            return true;
        }

        self.body[1..].contains(&head)
    }

    pub fn body(&self) -> &Vec<Point> {
        &self.body
    }
}

impl Direction {
    pub fn turn_left(&self) -> Self {
        match self {
            Direction::Up => Direction::Left,
            Direction::Left => Direction::Down,
            Direction::Down => Direction::Right,
            Direction::Right => Direction::Up,
        }
    }

    pub fn turn_right(&self) -> Self {
        match self {
            Direction::Up => Direction::Right,
            Direction::Right => Direction::Down,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
        }
    }
}


