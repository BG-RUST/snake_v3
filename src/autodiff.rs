use std::borrow::BorrowMut;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::ops::{Add, Sub, Mul, Div};
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(1);

#[derive(Clone)]
pub struct Var {
    pub value: f32,
    pub grad: Rc<RefCell<f32>>,
    backward_ops: Rc<RefCell<Vec<Box<dyn FnMut(f32)>>>>,
    parents: Rc<RefCell<Vec<Var>>>,
    id: usize,
}

impl Var {
    pub fn new(value: f32) -> Self {
        Self {
            value,
            grad: Rc::new(RefCell::new(0.0)),
            backward_ops: Rc::new(RefCell::new(Vec::new())),
            parents: Rc::new(RefCell::new(Vec::new())),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn from_vec(values: &[f32]) -> Vec<Var> {
        values.iter().map(|&v| Var::new(v)).collect()
    }

    pub fn backward(&self) {
        *self.grad.borrow_mut() = 1.0;

        let mut visited = HashSet::new();
        let mut stack = vec![self.clone()];

        while let Some(v) = stack.pop() {
            if !visited.insert(v.id) {
                continue;
            }

            let grad_val = *v.grad.borrow();
            for op in v.backward_ops.borrow_mut().iter_mut() {
                op(grad_val);
            }

            for parent in v.parents.borrow().iter() {
                stack.push(parent.clone());
            }
        }
    }

    pub fn zero_grad(&self) {
        *self.grad.borrow_mut() = 0.0;
    }

    pub fn grad(&self) -> f32 {
        *self.grad.borrow()
    }
    pub fn set_grad(&mut self, new_grad: f32) {
        *self.grad.borrow_mut() = new_grad;
    }

    fn add_backward_op<F: 'static + FnMut(f32)>(&self, op: F) {
        self.backward_ops.borrow_mut().push(Box::new(op));
    }

    fn add_parent(&self, parent: Var) {
        self.parents.borrow_mut().push(parent);
    }
}

impl Add for Var {
    type Output = Var;
    fn add(self, rhs: Var) -> Var {
        let out = Var::new(self.value + rhs.value);
        let a = self.clone();
        let b = rhs.clone();
        out.add_parent(a.clone());
        out.add_parent(b.clone());
        out.add_backward_op(move |grad| {
            *a.grad.borrow_mut() += grad;
            *b.grad.borrow_mut() += grad;
        });
        out
    }
}

impl Sub for Var {
    type Output = Var;
    fn sub(self, rhs: Var) -> Var {
        let out = Var::new(self.value - rhs.value);
        let a = self.clone();
        let b = rhs.clone();
        out.add_parent(a.clone());
        out.add_parent(b.clone());
        out.add_backward_op(move |grad| {
            *a.grad.borrow_mut() += grad;
            *b.grad.borrow_mut() -= grad;
        });
        out
    }
}

impl Mul for Var {
    type Output = Var;
    fn mul(self, rhs: Var) -> Var {
        let out = Var::new(self.value * rhs.value);
        let a = self.clone();
        let b = rhs.clone();
        out.add_parent(a.clone());
        out.add_parent(b.clone());
        out.add_backward_op(move |grad| {
            *a.grad.borrow_mut() += grad * b.value;
            *b.grad.borrow_mut() += grad * a.value;
        });
        out
    }
}

impl Div for Var {
    type Output = Var;
    fn div(self, rhs: Var) -> Var {
        let out = Var::new(self.value / rhs.value);
        let a = self.clone();
        let b = rhs.clone();
        out.add_parent(a.clone());
        out.add_parent(b.clone());
        out.add_backward_op(move |grad| {
            *a.grad.borrow_mut() += grad / b.value;
            *b.grad.borrow_mut() -= grad * a.value / (b.value * b.value);
        });
        out
    }
}

pub fn relu(x: Var) -> Var {
    let val = if x.value > 0.0 { x.value } else { 0.0 };
    let out = Var::new(val);
    let x_clone = x.clone();
    out.add_parent(x_clone.clone());
    out.add_backward_op(move |grad| {
        if x_clone.value > 0.0 {
            *x_clone.grad.borrow_mut() += grad;
        }
    });
    out
}

pub fn powi(x: Var, power: i32) -> Var {
    let val = x.value.powi(power);
    let out = Var::new(val);
    let x_clone = x.clone();
    out.add_parent(x_clone.clone());
    out.add_backward_op(move |grad| {
        *x_clone.grad.borrow_mut() += grad * (power as f32) * x_clone.value.powi(power - 1);
    });
    out
}

pub fn abs(x: Var) -> Var {
    let val = x.value.abs();
    let out = Var::new(val);
    let x_clone = x.clone();
    out.add_parent(x_clone.clone());
    out.add_backward_op(move |grad| {
        if x_clone.value >= 0.0 {
            *x_clone.grad.borrow_mut() += grad;
        } else {
            *x_clone.grad.borrow_mut() -= grad;
        }
    });
    out
}

pub fn dot_var(weights: &[Var], vars: &[Var]) -> Var {
    weights.iter().zip(vars.iter()).fold(Var::new(0.0), |acc, (w, v)| acc + w.clone() * v.clone())
}
///все еще полнейшая хуйня которая не работает
/// }