use rand::Rng;
use crate::game::draw_cell;

#[derive(Debug)]
pub struct Food {
    pub position: (u32, u32),
}

impl Food {
    pub fn new(width: u32, height: u32, cell_size: u32) -> Self {
        let cols = width / cell_size;
        let rows = height / cell_size;
        let mut rng = rand::thread_rng();

        let x = rng.gen_range(1..cols - 1);
        let y = rng.gen_range(1..rows - 1);

        Self { position: (x, y) }
    }

    pub fn draw(&self, frame: &mut [u8], cell_size: u32, screen_width: u32) {
        draw_cell(
            frame,
            self.position.0,
            self.position.1,
            cell_size,
            screen_width,
            [255, 0, 0, 255],
        );
    }
}