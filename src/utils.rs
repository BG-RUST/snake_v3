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

    // Случайное число в [0, n).
    pub fn gen_range_u32(&mut self, n: u32) -> u32 {
        // Берём 32 бита и берём модуль.
        (self.next_u64() as u32) % n
    }
}