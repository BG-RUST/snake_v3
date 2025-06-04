use rand::Rng;
use crate::snake::*;

pub struct Food {
    pub position: (u32, u32),
}

impl Food {
    /// Генерирует новую еду в случайной позиции на поле
    ///
    /// - Учитывает размеры поля `BOARD_WIDTH`, `BOARD_HEIGHT`
    /// - Проверяет, чтобы еда не появилась внутри тела змейки
    /// - Использует цикл `loop`, чтобы искать подходящую свободную клетку
    pub fn new(snake_body: &[(u32, u32)]) -> Self {
        let mut rng = rand::thread_rng();
        let mut pos;

        loop {
            pos = (rng.gen_range(0..BOARD_WIDTH), rng.gen_range(0..BOARD_HEIGHT));
            // Продолжаем, пока позиция ЗАНЯТА
            if !snake_body.contains(&pos) {
                break;
            }
        }

        Food { position: pos }
    }

}