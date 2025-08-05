// log.rs

// Простые функции для логирования
pub fn info(message: &str) {
    println!("[INFO] {}", message);
}

pub fn episode(ep: usize, score: u32, epsilon: f32) {
    println!("[Эпизод {}] Очки: {}, Эпсилон: {:.3}", ep, score, epsilon);
}

pub fn reward_log(ep: usize, total: f32) {
    if ep % 100 == 0 {
        println!("[REWARD] Эпизод {}: total_reward = {:.2}", ep, total);
    }
}