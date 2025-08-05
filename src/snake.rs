
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    // Возвращает противоположное направление (180°).
    pub fn opposite(&self) -> Direction {
        match self {
            Direction::Up => Direction::Down,
            Direction::Down => Direction::Up,
            Direction::Left => Direction::Right,
            Direction::Right => Direction::Left,
        }
    }

    // Возвращает направление налево (поворот на 90° влево).
    pub fn left(&self) -> Direction {
        match self {
            Direction::Up => Direction::Left,
            Direction::Down => Direction::Right,
            Direction::Left => Direction::Down,
            Direction::Right => Direction::Up,
        }
    }

    // Возвращает направление направо (поворот на 90° вправо).
    pub fn right(&self) -> Direction {
        match self {
            Direction::Up => Direction::Right,
            Direction::Down => Direction::Left,
            Direction::Left => Direction::Up,
            Direction::Right => Direction::Down,
        }
    }

    // Возвращает смещение координат (dx, dy) для шага в данном направлении.
    pub fn delta(&self) -> (i32, i32) {
        match self {
            Direction::Up => (0, -1),
            Direction::Down => (0, 1),
            Direction::Left => (-1, 0),
            Direction::Right => (1, 0),
        }
    }
}

pub struct Snake {
    pub body: Vec<crate::utils::Point>, // list of coordinates, body[0] - head
    pub direction: Direction, // current direction
}

impl Snake {
    pub fn new(head_position: crate::utils::Point, init_length: usize, direction: Direction) -> Snake {
        let mut body = Vec::new();
        body.push(head_position);
        let tail_dir = direction.opposite();
        let (dx, dy) = tail_dir.delta();
        for i in 1..init_length {
            // Каждый следующий сегмент располагается на 1 шаг в противоположном направлении от движения головы
            let x = head_position.x + dx * (i as i32);
            let y = head_position.y + dy * (i as i32);
            body.push(crate::utils::Point { x, y });
        }
        Snake { body, direction }
    }
    ///return current direction head coordinates
    pub fn head(&self) -> crate::utils::Point {
        self.body[0]
    }

    // Перемещает змейку на один шаг вперед.
    // Если grow == true, змейка растет (хвост не удаляется).
    // Если grow == false, змейка движется обычно (удаляется последний сегмент хвоста).
    pub fn move_forward(&mut self, grow: bool) {
        //calculate new head position based on current direction
        let (dx, dy) = self.direction.delta();
        let head = self.head();
        let new_head = crate::utils::Point { x: head.x + dx, y: head.y + dy };
        //add new head to beginning of the segment list
        self.body.insert(0, new_head);
        if !grow {
            self.body.pop();
        }
    }
}
