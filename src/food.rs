use rand::{Rng, thread_rng};

#[derive(Debug)]
pub struct Food {
    pub position: (i32, i32),
}

impl Food {
    pub fn new() -> Self {
        Self {
            position: (5, 5),
        }
    }

    pub fn respawn(&mut self, snake_body: &[(i32, i32)]) {

        let mut rng = rand::thread_rng();
        loop {
            let pos = (rng.gen_range(1..19), rng.gen_range(1..19));
            if !snake_body.contains(&pos) {
                self.position = pos;
                break;
            }
        }

    }

    pub fn draw(&self, frame: &mut [u8]) {
        let cell_size = 32;
        let (x, y) = self.position;
        for dy in 0..cell_size {
            for dx in 0..cell_size {
                let px = (x * cell_size + dx) as usize;
                let py = (y * cell_size + dy) as usize;
                let i = (py * 640 + px) * 4;
                if i + 4 <= frame.len() {
                    frame[i..i + 4].copy_from_slice(&[255, 0, 0, 255]); // красная еда
                }
            }
        }
    }
}