#[derive(Debug)]
pub struct Snake {
    pub body: Vec<(i32, i32)>,
    dir: (i32, i32),
}

impl Snake {
    pub fn new() -> Self {
        Self {
            body: vec![(10, 10)],
            dir: (1, 0),
        }
    }

    pub fn head(&self) -> (i32, i32) {
        self.body[0]
    }

    pub fn update(&mut self) {
        let mut new_head = self.head();
        new_head.0 += self.dir.0;
        new_head.1 += self.dir.1;
        self.body.insert(0, new_head);
        self.body.pop();
    }

    pub fn grow(&mut self) {
        let tail = *self.body.last().unwrap();
        self.body.push(tail);
    }

    pub fn set_dir(&mut self, dir: (i32, i32)) {
        // предотвращаем разворот на 180°
        if (self.dir.0 + dir.0, self.dir.1 + dir.1) != (0, 0) {
            self.dir = dir;
        }
    }

    //столкновение с собой
    pub fn is_colliding_with_self(&self) -> bool {
        let head = self.head();
        self.body[1..].contains(&head)
    }
    pub fn draw(&self, frame: &mut [u8]) {
        let cell_size = 32;
        for &(x, y) in &self.body {
            for dy in 0..cell_size {
                for dx in 0..cell_size {
                    let px = (x * cell_size + dx) as usize;
                    let py = (y * cell_size + dy) as usize;
                    let i = (py * 640 + px) * 4;
                    if i + 4 <= frame.len() {
                        frame[i..i + 4].copy_from_slice(&[0, 255, 0, 255]); // зелёная змейка
                    }
                }
            }
        }
    }
}