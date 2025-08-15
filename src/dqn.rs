/// agent, replay buffer, Double DQN, learning, save/load

use crate::network::Net;                                // our MLP network
use crate::utils::*;  // RNG + numeric helpers
use crate::log;                                         // logging (info/warn/error/scalar)
use std::fs::File;
use std::io::{Read, Write};

/// Hyperparameters for the agent.
pub struct AgentConfig {
    pub obs_dim: usize,            // observation size (Game::observation_dim())
    pub act_dim: usize,            // number of actions (3: left/straight/right)
    pub hidden: usize,             // hidden width (e.g., 64)
    pub buffer_capacity: usize,    // replay buffer capacity
    pub batch_size: usize,         // minibatch size
    pub gamma: f32,                // discount factor
    pub lr: f32,                   // learning rate
    pub eps_start: f32,            // initial epsilon
    pub eps_end: f32,              // final epsilon
    pub eps_decay_steps: u64,      // decay horizon for epsilon
    pub tau: f32,                  // soft-update rate for target network
    pub learn_start: usize,        // warmup size before learning starts
    pub updates_per_step: usize,   // number of SGD updates per environment step
    pub seed: u64,                 // RNG seed
}

/// A single experience tuple (s, a, r, s', done).
struct Transition {
    s: Vec<f32>,
    a: u8,
    r: f32,
    s2: Vec<f32>,
    done: bool,
}

/// Ring replay buffer.
struct ReplayBuffer {
    cap: usize,
    buf: Vec<Transition>,
    idx: usize, // next overwrite position
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self { cap: capacity, buf: Vec::with_capacity(capacity), idx: 0 }
    }

    fn len(&self) -> usize { self.buf.len() }

    fn push(&mut self, tr: Transition) {
        if self.buf.len() < self.cap {
            self.buf.push(tr);
        } else {
            self.buf[self.idx] = tr;
            self.idx = (self.idx + 1) % self.cap;
        }
    }

    /// Sample random indices for a minibatch.
    fn sample_indices(&self, rng: &mut LcgRng, batch: usize) -> Vec<usize> {
        let n = self.buf.len() as u32;
        let mut out = Vec::with_capacity(batch);
        for _ in 0..batch {
            out.push(rng.gen_range_u32(n) as usize);
        }
        out
    }
}

/// DQN / Double-DQN agent.
pub struct DQNAgent {
    cfg: AgentConfig,
    pub online: Net,   // online network: used for argmax and updated by SGD
    pub target: Net,   // target network: used for stable bootstrap targets
    replay: ReplayBuffer,

    rng: LcgRng,       // epsilon-greedy + sampling RNG
    eps: f32,          // current epsilon
    pub steps_done: u64, // number of env steps elapsed (for epsilon schedule)

    // monitoring
    pub last_loss: f32, // last averaged batch loss
}

impl DQNAgent {
    /// Build networks/buffer; try to load weights and agent state if present.
    pub fn new(cfg: AgentConfig) -> Self {
        // Pull out the config fields we need *before* moving `cfg` into the struct.
        let seed             = cfg.seed;
        let obs_dim          = cfg.obs_dim;
        let act_dim          = cfg.act_dim;
        let hidden           = cfg.hidden;
        let buffer_capacity  = cfg.buffer_capacity;

        // Two independent RNG states for weight init (avoid cloning).
        let online = Net::new(obs_dim, hidden, hidden, act_dim, LcgRng::new(seed));
        let mut target = Net::new(obs_dim, hidden, hidden, act_dim, LcgRng::new(seed ^ 0xA5A5_5A5A));
        target.copy_from(&online);

        // Separate RNG for epsilon-greedy & sampling.
        let replay_rng_seed = 0xDEAD_BEEFu64 ^ seed;

        let mut ag = Self {
            cfg, // move happens here exactly once
            online,
            target,
            replay: ReplayBuffer::new(buffer_capacity),
            rng: LcgRng::new(replay_rng_seed),
            eps: 0.0,
            steps_done: 0,
            last_loss: 0.0,
        };

        // Load saved weights/state if available (resume training).
        if ag.online.load("weights.bin").is_ok() {
            ag.target.copy_from(&ag.online);
            log::info("loaded weights.bin");
        }
        if let Ok((eps, steps)) = load_agent_state("agent_state.bin") {
            ag.eps = eps;
            ag.steps_done = steps;
            log::info(&format!("loaded agent_state.bin (eps={:.3}, steps={})", eps, steps));
        } else {
            ag.eps = ag.cfg.eps_start;
        }
        ag
    }

    /// Current epsilon (for UI/logs).
    pub fn current_epsilon(&self) -> f32 { self.eps }

    /// Convenience: expose replay length for logs.
    pub fn replay_len(&self) -> usize { self.replay.len() }

    /// Epsilon-greedy action: with prob ε — random action; else argmax Q(s, ·).
    pub fn select_action(&mut self, obs: &[f32]) -> u8 {
        let u = self.rng.next_f32();
        if u < self.eps {
            return self.rng.gen_range_u32(self.cfg.act_dim as u32) as u8;
        }
        let q = self.online.forward(obs);
        if has_non_finite(&q) {
            log::error("Q contains NaN/Inf in select_action — falling back to random action");
            return self.rng.gen_range_u32(self.cfg.act_dim as u32) as u8;
        }
        argmax(&q) as u8
    }

    /// Store transition in replay.
    pub fn remember(&mut self, s: &[f32], a: u8, r: f32, s2: &[f32], done: bool) {
        self.replay.push(Transition { s: s.to_vec(), a, r, s2: s2.to_vec(), done });
    }

    /// Optionally perform training updates (once buffer is warm).
    pub fn maybe_learn(&mut self) {
        if self.replay.len() < self.cfg.learn_start { return; }
        for _ in 0..self.cfg.updates_per_step {
            self.learn_once();
        }
    }

    /// One SGD step on a minibatch.
    fn learn_once(&mut self) {
        let idxs = self.replay.sample_indices(&mut self.rng, self.cfg.batch_size);
        self.online.zero_grad();

        // Accumulators for logging.
        let mut loss_acc = 0.0f32;
        let mut td_errs: Vec<f32> = Vec::with_capacity(self.cfg.batch_size);
        let mut q_sel:   Vec<f32> = Vec::with_capacity(self.cfg.batch_size);

        for &k in &idxs {
            let tr = &self.replay.buf[k];

            // Q(s, ·)
            let q_s = self.online.forward(&tr.s);
            if has_non_finite(&q_s) {
                log::error("NaN/Inf in Q(s,·) during learn_once — skipping this sample");
                continue;
            }

            // Double-DQN target:
            // a* = argmax_a' Q_online(s', a'), y = r + γ Q_target(s', a*)
            let q_s2_online = self.online.forward(&tr.s2);
            let a_star = argmax(&q_s2_online);
            let mut y = tr.r;
            if !tr.done {
                let q_s2_targ = self.target.forward(&tr.s2);
                y += self.cfg.gamma * q_s2_targ[a_star];
            }

            // TD error on chosen action.
            let a = tr.a as usize;
            let e = q_s[a] - y;
            td_errs.push(e);
            q_sel.push(q_s[a]);

            // Huber gradient (δ=1).
            let g = if e.abs() <= 1.0 { e } else { e.signum() };
            let g_scaled = g / (self.cfg.batch_size as f32);

            // dL/dQ(s, ·) is zero except at selected action.
            let mut d_q = vec![0.0f32; self.cfg.act_dim];
            d_q[a] = g_scaled;
            self.online.backward_from_output_grad(d_q);

            // Huber loss value (for logging).
            let l = if e.abs() <= 1.0 { 0.5 * e * e } else { e.abs() - 0.5 };
            loss_acc += l / (self.cfg.batch_size as f32);
        }

        if td_errs.is_empty() {
            log::warn("learn_once: batch had only bad/NaN samples — skipping update");
            return;
        }

        // Health checks before optimizer step.
        let grad_l2 = self.online.grad_l2_sum_all().sqrt();
        if self.online.non_finite_any() || !grad_l2.is_finite() {
            log::error(&format!("non-finite grads/params detected before step (||g||={}) — skipping update", grad_l2));
            self.online.zero_grad();
            return;
        }

        // Global grad-norm clip and Adam step.
        let scale = self.online.clip_grad_norm(1.0);
        self.online.step_adam(self.cfg.lr, 0.9, 0.999, 1e-8, scale);

        // Soft-update target network.
        self.target.soft_update_from(&self.online, self.cfg.tau);

        // Store averaged loss for external logs.
        self.last_loss = loss_acc;

        // Post-step health check (optional, good for debugging).
        if self.online.non_finite_any() {
            log::error("non-finite detected right after Adam step — parameters blew up");
        }

        // Batch stats for plots.
        let td = vec_stats(&td_errs);
        let qs = vec_stats(&q_sel);
        log::scalar(self.steps_done, "loss",       loss_acc);
        log::scalar(self.steps_done, "grad_norm",  grad_l2);
        log::scalar(self.steps_done, "td_mean",    td.mean);
        log::scalar(self.steps_done, "td_min",     td.min);
        log::scalar(self.steps_done, "td_max",     td.max);
        log::scalar(self.steps_done, "q_sel_mean", qs.mean);
        log::scalar(self.steps_done, "q_sel_min",  qs.min);
        log::scalar(self.steps_done, "q_sel_max",  qs.max);
        log::scalar(self.steps_done, "epsilon",    self.eps);
    }

    /// Called every env step: updates epsilon according to a linear schedule.
    pub fn on_step(&mut self, global_steps: u64) {
        self.steps_done = global_steps;
        let t = (self.steps_done as f32 / self.cfg.eps_decay_steps as f32).min(1.0);
        self.eps = self.cfg.eps_start + t * (self.cfg.eps_end - self.cfg.eps_start);
    }

    /// Save weights and agent state (epsilon/steps) to disk.
    pub fn save_all(&self) {
        if self.online.save("weights.bin").is_ok() {
            log::info("saved weights.bin");
        }
        if save_agent_state("agent_state.bin", self.eps, self.steps_done).is_ok() {
            log::info("saved agent_state.bin");
        }
    }
}

// -------- helpers (module-private) --------

/// Argmax over a slice, returning the index of the maximum element.
fn argmax(v: &[f32]) -> usize {
    let mut best_i = 0;
    let mut best_v = v[0];
    for i in 1..v.len() {
        if v[i] > best_v {
            best_v = v[i];
            best_i = i;
        }
    }
    best_i
}

/// Save epsilon and the step counter to a simple binary file.
fn save_agent_state(path: &str, eps: f32, steps_done: u64) -> std::io::Result<()> {
    let mut f = File::create(path)?;
    f.write_all(&eps.to_le_bytes())?;
    f.write_all(&steps_done.to_le_bytes())?;
    Ok(())
}

/// Load epsilon and the step counter from a binary file.
/// Layout: [f32 eps][u64 steps]  => 4 + 8 = 12 bytes total.
fn load_agent_state(path: &str) -> std::io::Result<(f32, u64)> {
    let mut f = File::open(path)?;
    let mut buf = [0u8; 12];
    f.read_exact(&mut buf)?;
    let mut fe = [0u8; 4]; fe.copy_from_slice(&buf[0..4]);
    let mut fs = [0u8; 8]; fs.copy_from_slice(&buf[4..12]);
    Ok((f32::from_le_bytes(fe), u64::from_le_bytes(fs)))
}
