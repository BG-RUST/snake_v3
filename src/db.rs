use std::fs::OpenOptions;                 // Файл с дозаписью.
use std::io::Write;                       // Запись строк.

pub fn append_episode_result(path: &str, ep: u64, ret: f32, steps: u64) -> Result<(), String> {
    // Открываем/создаём CSV.
    let mut f = OpenOptions::new().create(true).append(true).open(path)
        .map_err(|e| format!("open csv: {}", e))?;
    // Если файл только что создан — можно было бы написать заголовок; опустим для простоты.
    // Пишем строку.
    let line = format!("{},{},{},{}\n", ep, ret, steps, now_ts());
    f.write_all(line.as_bytes()).map_err(|e| format!("write csv: {}", e))?;
    Ok(())
}

// Временная метка в секундах (для CSV).
fn now_ts() -> u64 {
    let now = std::time::SystemTime::now();
    now.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()
}