use crate::snake::Point;
use rand::Rng;

pub struct Food{
    pub pos: Point,
}

impl Food{
    pub fn new_random(width: usize, height: usize) -> Self {
        let mut rng = rand::thread_rng();

        let x = rng.gen_range(0..width);
        let y = rng.gen_range(0..height);

        Self {
            pos: Point { x, y },
        }
    }
}