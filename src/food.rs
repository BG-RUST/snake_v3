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

    pub fn respawn(&mut self, snake_body: &[(i32, i32)], width: u32, height: u32) {
        let mut rng = rand::thread_rng();

        loop {
            let pos = (
                rng.gen_range(1..(width as i32 - 1)),
                rng.gen_range(1..(height as i32 - 1)),
            );
            println!("🔄 trying pos: {:?}", pos);
            if !snake_body.contains(&pos) {
                self.position = pos;
                println!("✅ set pos: {:?}", self.position);
                break;
            }
        }
    }

    pub fn draw(&self, frame: &mut [u8], screen_width: usize) {
        let cell_size = 32;
        let (x, y) = self.position;
        for dy in 0..cell_size {
            for dx in 0..cell_size {
                let px = (x * cell_size + dx) as usize;
                let py = (y * cell_size + dy) as usize;
                let i = (py * screen_width + px) * 4;
                if i + 4 <= frame.len() {
                    frame[i..i + 4].copy_from_slice(&[255, 0, 0, 255]); // красная еда
                }
            }
        }
    }
}