use crate::snake::Point;
use rand::Rng;

pub struct Food {
    pub pos: Point,
}

impl Food {
    pub fn new_random(width: usize, height: usize) -> Self {
        let mut rng = rand::thread_rng();

        let x = rng.gen_range(0..width) as i32;
        let y = rng.gen_range(0..height) as i32;

        Self {
            pos: Point { x, y },
        }
    }
}
