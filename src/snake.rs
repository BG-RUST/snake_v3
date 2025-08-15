use std::collections::VecDeque;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Dir {Up, Down, Left, Right}

impl Dir {
    //check for opposite directions
    fn is_opposite(self, other: Dir) -> bool {
        matches!(
            (self, other),
            (Dir::Up, Dir::Down) | (Dir::Down, Dir::Up) | (Dir::Left, Dir::Right) | (Dir::Right, Dir::Left)
        )
    }

    //offset in direction
    fn delta(self) -> (i32, i32) {
        match self {
            Dir::Up => (0, -1),
            Dir::Down => (0, 1),
            Dir::Left => (-1, 0),
            Dir::Right => (1, 0),
        }
    }
}

pub struct Snake {
    //segments ate stored from tail to head
    body: VecDeque<(i32, i32)>,
    //current direction
    dir: Dir,
    //growth counter (how many steps to not remove the tail)
    grow: usize,

}

impl Snake {
    //create snake of length 3 in the center, facing right
    pub fn new(cx: i32, cy: i32) -> Self {
        let mut body = VecDeque::new();
        body.push_back((cx - 1, cy));
        body.push_back((cx, cy));
        body.push_back((cx + 1, cy));
        Self { body, dir: Dir::Right, grow: 0 }
    }

    //current head
    pub fn head(&self) -> (i32, i32) {
        *self.body.back().unwrap()
    }
    //segments from render

    //convenience method: get a copy pf the segments as a vec
    pub fn segments_vec(&self) -> Vec<(i32, i32)>{
        self.body.iter().copied().collect()
    }
    //current direction - needed for RL "relative" actions and observation
    pub fn dir(&self) -> Dir { self.dir }
    pub fn len(&self) -> usize { self.body.len() }


    //apply the desired direction if it is not opposite to the current one
    pub fn apply_dir(&mut self, want: Dir) {
        if !want.is_opposite(self.dir) {
            self.dir = want;
        }
    }
    //take one step: add a new head; remove the tail if we are not growing
    pub fn advance(&mut self) {
        let (dx, dy) = self.dir.delta();
        let (hx, hy) = self.head();
        let new_head = (hx + dx, hy + dy);
        self.body.push_back(new_head);
        if self.grow > 0 {
            self.grow -= 1;
        } else {
            self.body.pop_front();
        }

    }

    //mark that it needs to grow by 1 segment
    pub fn feed(&mut self) { self.grow += 1}

    //check if the snake occupies cell(x,y)
    pub fn occupies(&self, x: i32, y: i32) -> bool {
        self.body.iter().any(|&(sx, sy)| sx == x && sy == y)
    }

    //head collision with body (the last segmenr is head, we dont count it)
    pub fn self_collision(&self) -> bool {
        let head = self.head();
        self.body.iter().take(self.body.len() - 1).any(|&p| p == head)
    }
}