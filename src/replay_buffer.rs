use rand::{seq::SliceRandom, thread_rng};
#[derive(Clone)]
pub struct Transition {
    pub state: [f32; 38],
    pub action: usize,
    pub reward: f32,
    pub next_state: [f32; 38],
    pub done: bool,
}

pub struct ReplayBuffer {
    buffer: Vec<Transition>,
    capacity: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
        }
    }

    ///add new transition
    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() == self.capacity {
            self.buffer.remove(0);
        }
        self.buffer.push(transition);
    }

    ///find how many transition there are in total
    pub fn ken(&self) -> usize {
        self.buffer.len()
    }

    /// is it possible to learn (are there at least batch_size elements)
    pub fn is_ready(&self, batch_size: usize) -> bool {
        self.buffer.len() >= batch_size
    }

    ///return random batch_transition (without replacement)
    pub fn sample(&self, batch_size: usize) -> Vec<Transition> {
        let mut rng = thread_rng();
        self.buffer
            .choose_multiple(&mut rng, batch_size)
            .cloned()
            .collect()
    }
}

impl ReplayBuffer {
    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}