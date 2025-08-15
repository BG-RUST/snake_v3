//simplest deterministic RNG without thrid party crates
pub struct LcgRng { state: u64 }


impl LcgRng {
    pub fn new(seed: u64) -> Self { Self { state: seed } }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        // Параметры: Numerical Recipes.
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        self.state
    }

    pub fn next_f32(&mut self) -> f32 {
        let x = self.next_u64() >> 40;                 // 24 бита.
        (x as f32) / ((1u32 << 24) as f32)            // [0,1)
    }

    // Случайное число в [0, n).
    pub fn gen_range_u32(&mut self, n: u32) -> u32 {
        // Берём 32 бита и берём модуль.
        (self.next_u64() as u32) % n
    }
}

pub fn has_non_finite(xs: &[f32]) -> bool {
    // Идём по всем значениям и проверяем is_finite().
    xs.iter().any(|&v| !v.is_finite())
}

/// Простейшая статистика по вектору — min / max / mean.
pub struct Stats { pub min: f32, pub max: f32, pub mean: f32 }

/// Подсчёт статистики по срезу f32.
pub fn vec_stats(xs: &[f32]) -> Stats {
    // Пустой вектор не ожидается (но на всякий случай дадим нули).
    if xs.is_empty() { return Stats { min: 0.0, max: 0.0, mean: 0.0 }; }
    // Инициализируем минимум/максимум первым элементом.
    let mut mn = xs[0];
    let mut mx = xs[0];
    // Накопитель суммы для среднего.
    let mut sum = 0.0f32;
    // Проходим все элементы.
    for &v in xs {
        if v < mn { mn = v; }
        if v > mx { mx = v; }
        sum += v;
    }
    // Возвращаем структуру со сводкой.
    Stats { min: mn, max: mx, mean: sum / (xs.len() as f32) }
}
pub fn now_millis() -> u128 {
    let now = std::time::SystemTime::now();
    let dur = now.duration_since(std::time::UNIX_EPOCH).unwrap();
    (dur.as_secs() as u128) * 1000 + (dur.subsec_nanos() as u128) / 1_000_000
}