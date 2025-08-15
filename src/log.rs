use std::fs::OpenOptions;                 // Открываем/создаём файл для дозаписи.
use std::io::Write;                       // Трейт для записи байт/строк.
use crate::utils::now_millis;             // Метка времени (мс со старта эпохи).

// Вспомогательный общий писатель: один раз формируем строку и пишем куда нужно.
fn write_line(level: &str, msg: &str) {
    // Собираем строку: [timestamp] LEVEL: сообщение + перевод строки.
    let line = format!("[{}] {}: {}\n", now_millis(), level, msg);
    // Дублируем в stdout (сразу видно в консоли).
    print!("{line}");
    // Пишем в файл train.log (создаём при отсутствии, дописываем в конец).
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("train.log") {
        let _ = f.write_all(line.as_bytes()); // Игнорируем ошибку записи, чтобы лог не «ронял» процесс.
    }
}

// Информационные сообщения — нормальный «зелёный» поток.
pub fn info(msg: &str)  { write_line("INFO",  msg); }
// Предупреждения — что-то подозрительное, но обучение можно продолжать.
pub fn warn(msg: &str)  { write_line("WARN",  msg); }
// Ошибки — что-то сломалось/NaN — нужна реакция (скип шага, сейв и т.д.).
pub fn error(msg: &str) { write_line("ERROR", msg); }

// Удобно логировать числовые метрики в едином формате (для последующего парсинга).
pub fn scalar(step: u64, name: &str, value: f32) {
    // Формат: SCALAR step=<..> name=<..> value=<..> — легко grep/parse.
    let line = format!("[{}] SCALAR step={} name={} value={:.6}\n", now_millis(), step, name, value);
    print!("{line}");
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open("train.log") {
        let _ = f.write_all(line.as_bytes());
    }
}
