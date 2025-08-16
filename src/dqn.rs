// dqn.rs — исправленная версия
// В ЭТОЙ ВЕРСИИ главная правка — порядок вызовов forward в learn_once():
//   1) forward( s' ) на online — только для argmax(a*), кеши можно перетирать;
//   2) forward( s' ) на target — для таргета y (кеши другого нетворка);
//   3) forward( s   ) на online — ПОСЛЕДНИЙ перед backward, чтобы градиент шёл по s.
// Плюс: добавил лог q_abs_max для диагностики масштаба выходов.

use crate::network::Net;     // Подключаем нашу MLP-сеть.
use crate::utils::*;         // RNG и числовые утилиты.
use crate::log;              // Логгер (info/warn/error/scalar).
use std::fs::File;           // Файлы — для сохранения/загрузки состояния агента.
use std::io::{Read, Write};  // Трейты чтения/записи байтов.

// ---------------- Гиперпараметры агента ----------------

/// Конфиг: размеры, буфер, батч, дисконт, шаги eps, софт-апдейт, прогрев, апдейты, сид.
pub struct AgentConfig {
    pub obs_dim: usize,          // Размер наблюдения.
    pub act_dim: usize,          // Кол-во действий (3).
    pub hidden: usize,           // Ширина скрытых слоёв.
    pub buffer_capacity: usize,  // Вместимость реплея.
    pub batch_size: usize,       // Размер минибатча.
    pub gamma: f32,              // Дисконт γ.
    pub lr: f32,                 // Скорость обучения.
    pub eps_start: f32,          // Стартовое ε.
    pub eps_end: f32,            // Финальное ε.
    pub eps_decay_steps: u64,    // За сколько шагов дойти до ε_end.
    pub tau: f32,                // Коэф. софт-апдейта таргета.
    pub learn_start: usize,      // Сколько транзиций накопить до обучения.
    pub updates_per_step: usize, // Сколько SGD-апдейтов на шаг среды.
    pub seed: u64,               // Сид RNG.
}

/// Одна транзиция (s, a, r, s', done).
struct Transition {
    s: Vec<f32>,     // Состояние s.
    a: u8,           // Действие a.
    r: f32,          // Награда r.
    s2: Vec<f32>,    // Следующее состояние s'.
    done: bool,      // Флаг терминальности.
}

/// Кольцевой реплей-буфер.
struct ReplayBuffer {
    cap: usize,              // Вместимость.
    buf: Vec<Transition>,    // Данные.
    idx: usize,              // Куда писать при переполнении.
}
impl ReplayBuffer {
    fn new(capacity: usize) -> Self {            // Создаём буфер c заданной ёмкостью.
        Self { cap: capacity, buf: Vec::with_capacity(capacity), idx: 0 }
    }
    fn len(&self) -> usize { self.buf.len() }    // Текущая длина.
    fn push(&mut self, tr: Transition) {         // Добавление (с перезаписью по кругу).
        if self.buf.len() < self.cap {
            self.buf.push(tr);
        } else {
            self.buf[self.idx] = tr;
            self.idx = (self.idx + 1) % self.cap;
        }
    }
    fn sample_indices(&self, rng: &mut LcgRng, batch: usize) -> Vec<usize> { // Семплируем индексы.
        let n = self.buf.len() as u32;
        let mut out = Vec::with_capacity(batch);
        for _ in 0..batch { out.push(rng.gen_range_u32(n) as usize); }
        out
    }
}

// ---------------- Сам агент DQN/Double-DQN ----------------

pub struct DQNAgent {
    cfg: AgentConfig, // Конфиг.

    pub online: Net,  // Обучаемая сеть Q_θ.
    pub target: Net,  // Замороженная сеть Q_{θ^-}.

    replay: ReplayBuffer, // Реплей-буфер.

    rng: LcgRng,      // RNG для семплинга и ε-жадности.
    eps: f32,         // Текущее ε.
    pub steps_done: u64, // Сколько шагов обучили — для расписаний.

    pub last_loss: f32,  // Последний усреднённый лосс — для логов.
}

// ---------------- Страховочные константы ----------------

const MAX_GRAD_NORM: f32 = 1.0;   // Глобальный клип нормы градиента.
const WEIGHT_DECAY:   f32 = 1e-4; // AdamW-декей на весах.
const PARAM_CLIP:     f32 = 10.0; // Жёсткая обрезка параметров после шага.
const REWARD_CLIP:    f32 = 1.0;  // Клип наград в [-1, 1].
const TARGET_CLIP:    f32 = 10.0; // Клип таргета y в [-10, 10].

impl DQNAgent {
    /// Конструктор: создаём/инициализируем сети, буфер, RNG; пробуем загрузить веса/состояние.
    pub fn new(cfg: AgentConfig) -> Self {
        let seed            = cfg.seed;                     // Берём сид.
        let obs_dim         = cfg.obs_dim;                  // Размер входа.
        let act_dim         = cfg.act_dim;                  // Кол-во действий.
        let hidden          = cfg.hidden;                   // Ширина скрытых слоёв.
        let buffer_capacity = cfg.buffer_capacity;          // Вместимость реплея.

        let online = Net::new(obs_dim, hidden, hidden, act_dim, LcgRng::new(seed)); // Online-сеть.
        let mut target = Net::new(obs_dim, hidden, hidden, act_dim, LcgRng::new(seed ^ 0xA5A5_5A5A)); // Target-сеть.
        target.copy_from(&online);                          // Жёсткая копия online → target.

        let replay_rng_seed = 0xDEAD_BEEFu64 ^ seed;        // Сид для реплея.

        let mut ag = Self {                                 // Собираем структуру агента.
            cfg,
            online,
            target,
            replay: ReplayBuffer::new(buffer_capacity),
            rng: LcgRng::new(replay_rng_seed),
            eps: 0.0,
            steps_done: 0,
            last_loss: 0.0,
        };

        if ag.online.load("weights.bin").is_ok() {          // Пытаемся подгрузить веса.
            ag.target.copy_from(&ag.online);                // Синхронизируем target.
            log::info("loaded weights.bin");
        }
        if let Ok((eps, steps)) = load_agent_state("agent_state.bin") { // Пытаемся подгрузить eps/steps.
            ag.eps = eps;
            ag.steps_done = steps;
            log::info(&format!("loaded agent_state.bin (eps={:.3}, steps={})", eps, steps));
        } else {
            ag.eps = ag.cfg.eps_start;                      // Если нет состояния — стартуем с eps_start.
        }
        ag
    }

    /// Текущее ε.
    pub fn current_epsilon(&self) -> f32 { self.eps }

    /// Длина реплея.
    pub fn replay_len(&self) -> usize { self.replay.len() }

    /// ε-жадное действие по наблюдению.
    pub fn select_action(&mut self, obs: &[f32]) -> u8 {
        if self.rng.next_f32() < self.eps {                 // С вероятностью ε — случайное действие.
            return self.rng.gen_range_u32(self.cfg.act_dim as u32) as u8;
        }
        let q = self.online.forward(obs);                   // Иначе — forward и берём argmax.
        if has_non_finite(&q) {                             // Защита от NaN/Inf.
            log::error("Q contains NaN/Inf in select_action — fallback to random");
            return self.rng.gen_range_u32(self.cfg.act_dim as u32) as u8;
        }
        argmax(&q) as u8                                    // Индекс максимального Q.
    }

    /// Кладём транзицию в реплей.
    pub fn remember(&mut self, s: &[f32], a: u8, r: f32, s2: &[f32], done: bool) {
        self.replay.push(Transition { s: s.to_vec(), a, r, s2: s2.to_vec(), done });
    }

    /// Если реплей прогрелся — учимся (несколько апдейтов на шаг).
    pub fn maybe_learn(&mut self) {
        if self.replay.len() < self.cfg.learn_start { return; } // Ждём прогрева.
        for _ in 0..self.cfg.updates_per_step {                  // Делаем N апдейтов.
            self.learn_once();
        }
    }

    /// Один шаг обучения по минибатчу.
    fn learn_once(&mut self) {
        let idxs = self.replay.sample_indices(&mut self.rng, self.cfg.batch_size); // Семплируем индексы.
        self.online.zero_grad();                            // Сбрасываем градиенты.

        let mut loss_acc = 0.0f32;                          // Аккумулятор лосса (среднее по батчу).
        let mut td_errs: Vec<f32> = Vec::with_capacity(self.cfg.batch_size); // Для статистики TD-ошибок.
        let mut q_sel:   Vec<f32> = Vec::with_capacity(self.cfg.batch_size); // Для статистики Q выбранных действий.

        for &k in &idxs {                                   // Итерируем по батчу индексов.
            let tr = &self.replay.buf[k];                   // Берём транзицию.

            // ---------- ВАЖНАЯ ЧАСТЬ: порядок вызовов forward ----------

            // (1) Сначала считаем a* = argmax_a Q_online(s', a).
            //     Это перетирает кеши online — и нам это сейчас безразлично.
            let q_s2_online = self.online.forward(&tr.s2);
            let a_star = argmax(&q_s2_online);

            // (2) Теперь строим таргет через TARGET-сеть: y = r + γ * Q_target(s', a*)
            //     Кеши target независимы, они не мешают backward по online-сети.
            let mut y = tr.r.clamp(-REWARD_CLIP, REWARD_CLIP);
            if !tr.done {
                let q_s2_targ = self.target.forward(&tr.s2);
                y += self.cfg.gamma * q_s2_targ[a_star];
            }
            let y = y.clamp(-TARGET_CLIP, TARGET_CLIP);

            // (3) И ТОЛЬКО ТЕПЕРЬ делаем forward по s на ONLINE-сети.
            //     Этот forward ДОЛЖЕН быть ПОСЛЕДНИМ перед backward,
            //     чтобы кеши соответствовали вычислению Q(s,·).
            let q_s = self.online.forward(&tr.s);
            if has_non_finite(&q_s) {                       // На всякий случай — пропустим плохие сэмплы.
                continue;
            }

            // TD-ошибка по выбранному действию e = Q(s,a) − y.
            let a = tr.a as usize;
            let e = q_s[a] - y;

            // Сохраняем для статистики.
            td_errs.push(e);
            q_sel.push(q_s[a]);

            // Градиент Huber (δ=1): dL/dQ = clip(e, -1, 1).
            let g = if e.abs() <= 1.0 { e } else { e.signum() };
            let g_scaled = g / (self.cfg.batch_size as f32); // Усредняем по батчу.

            // Градиент по выходу: только по выбранному действию (один хот).
            let mut d_q = vec![0.0f32; self.cfg.act_dim];
            d_q[a] = g_scaled;

            // Backward: накопим градиенты в слоях online-сети.
            self.online.backward_from_output_grad(d_q);

            // Значение Huber-лосса для логов.
            let l = if e.abs() <= 1.0 { 0.5 * e * e } else { e.abs() - 0.5 };
            loss_acc += l / (self.cfg.batch_size as f32);
        }

        // Если весь батч оказался «плохим» — пропускаем шаг.
        if td_errs.is_empty() {
            log::warn("learn_once: batch had only bad/NaN samples — skipping update");
            return;
        }

        // Проверка здоровья до шага оптимизатора.
        let grad_l2 = self.online.grad_l2_sum_all().sqrt();         // Норма градиента (до клипа).
        if self.online.non_finite_any() || !grad_l2.is_finite() {    // Если NaN/Inf — пропускаем шаг.
            log::error(&format!("non-finite grads/params before step (||g||={}) — skip", grad_l2));
            self.online.zero_grad();
            return;
        }

        // Глобальный клип нормы и шаг AdamW.
        let scale = self.online.clip_grad_norm(MAX_GRAD_NORM);       // Масштаб клипа (≤1).
        self.online.step_adam(self.cfg.lr, 0.9, 0.999, 1e-8, scale, WEIGHT_DECAY); // Шаг оптимизатора.

        // Жёсткий клип параметров (доп. ремень безопасности).
        self.online.clamp_params(PARAM_CLIP);

        // Софт-апдейт таргет-сети: θ^- ← (1−τ)θ^- + τ θ.
        self.target.soft_update_from(&self.online, self.cfg.tau);

        // Сохраняем усреднённый лосс по батчу.
        self.last_loss = loss_acc;

        // Логи метрик батча.
        let td = vec_stats(&td_errs);                                  // Статы TD-ошибок.
        let qs = vec_stats(&q_sel);                                     // Статы Q выбранных действий.
        let q_abs_max: f32 = q_sel.iter().copied().map(f32::abs).fold(0.0f32, f32::max);// Диагностический максимум |Q|.
        log::scalar(self.steps_done, "loss",       loss_acc);           // Лосс.
        log::scalar(self.steps_done, "grad_norm",  grad_l2);            // Норма градиента (до клипа).
        log::scalar(self.steps_done, "td_mean",    td.mean);            // Средний TD.
        log::scalar(self.steps_done, "td_min",     td.min);             // Мин TD.
        log::scalar(self.steps_done, "td_max",     td.max);             // Макс TD.
        log::scalar(self.steps_done, "q_sel_mean", qs.mean);            // Средний Q выбранных действий.
        log::scalar(self.steps_done, "q_sel_min",  qs.min);             // Мин Q выбранных действий.
        log::scalar(self.steps_done, "q_sel_max",  qs.max);             // Макс Q выбранных действий.
        log::scalar(self.steps_done, "q_abs_max",  q_abs_max);          // Новый лог масштаба |Q|.
        log::scalar(self.steps_done, "epsilon",    self.eps);           // Текущее ε.
    }

    /// Обновляем расписание ε по номеру шага.
    pub fn on_step(&mut self, global_steps: u64) {
        self.steps_done = global_steps;                                        // Обновляем счётчик шагов.
        let t = (self.steps_done as f32 / self.cfg.eps_decay_steps as f32).min(1.0); // Нормируем 0..1.
        self.eps = self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start); // Линейный спуск ε.
    }

    /// Сохранение весов и состояния агента на диск.
    pub fn save_all(&self) {
        if self.online.save("weights.bin").is_ok() {           // Пишем веса online-сети.
            log::info("saved weights.bin");
        }
        if save_agent_state("agent_state.bin", self.eps, self.steps_done).is_ok() { // Пишем ε и шаги.
            log::info("saved agent_state.bin");
        }
    }
}

// ---------------- Вспомогательные функции ----------------

/// Индекс максимума.
fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0;                 // Текущий лучший индекс.
    let mut best_v = v[0];              // Текущее лучшее значение.
    for i in 1..v.len() {               // Проходим массив.
        if v[i] > best_v {              // Нашли больше — обновляем.
            best_v = v[i];
            best_i = i;
        }
    }
    best_i                               // Возвращаем индекс.
}

/// Сохраняем ε и steps в бинарный файл.
fn save_agent_state(path: &str, eps: f32, steps_done: u64) -> std::io::Result<()> {
    let mut f = File::create(path)?;                  // Создаём/переписываем файл.
    f.write_all(&eps.to_le_bytes())?;                 // Пишем 4 байта f32 в LE.
    f.write_all(&steps_done.to_le_bytes())?;          // Пишем 8 байт u64 в LE.
    Ok(())                                            // Ок.
}

/// Загружаем ε и steps из бинарного файла.
fn load_agent_state(path: &str) -> std::io::Result<(f32, u64)> {
    let mut f = File::open(path)?;                    // Открываем файл.
    let mut buf = [0u8; 12];                          // 4 + 8 байт.
    f.read_exact(&mut buf)?;                          // Читаем строго 12 байт.
    let mut fe = [0u8; 4]; fe.copy_from_slice(&buf[0..4]);   // Первые 4 — f32 ε.
    let mut fs = [0u8; 8]; fs.copy_from_slice(&buf[4..12]);  // Следующие 8 — u64 steps.
    Ok((f32::from_le_bytes(fe), u64::from_le_bytes(fs)))     // Возвращаем распакованные значения.
}
