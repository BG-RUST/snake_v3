use crate::snake::{Snake, Direction, BOARD_WIDTH, BOARD_HEIGHT};
use crate::food::Food;
use crate::dqn::{DQNAgent, get_state};
use winit::platform::run_return::EventLoopExtRunReturn;

use winit::{
    event::{Event, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use pixels::{Pixels, SurfaceTexture};
use winit_input_helper::WinitInputHelper;

use std::time::{Duration, Instant};

const CELL_SIZE: u32 = 32;
const WIDTH: u32 = BOARD_WIDTH * CELL_SIZE;
const HEIGHT: u32 = BOARD_HEIGHT * CELL_SIZE;

#[derive(PartialEq, Copy, Clone)]
pub enum GameResult {
    Restart,
    Exit,
}

pub fn start(agent: &mut DQNAgent) -> GameResult {
    let mut event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_title("Snake AI with DQN")
        .with_inner_size(winit::dpi::LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();

    let mut pixels = Pixels::new(WIDTH, HEIGHT, SurfaceTexture::new(WIDTH, HEIGHT, &window)).unwrap();
    let mut input = WinitInputHelper::new();

    // 🧠 Объекты игры
    let mut snake = Snake::new();
    let mut food = Food::new(&snake.body);
    let mut last_update = Instant::now();
    let update_interval = Duration::from_millis(100);
    let mut score = 0;

    let mut result = GameResult::Restart;

    event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.close_requested() {
                *control_flow = ControlFlow::Exit;
                result = GameResult::Exit;
                return;
            }
        }

        if last_update.elapsed() >= update_interval {
            // 1. Получаем текущее состояние
            let state = get_state(&snake, &food);

            // 2. Агент выбирает действие
            let action = agent.select_action(&state);
            match action {
                0 => snake.change_direction(Direction::Up),
                1 => snake.change_direction(Direction::Down),
                2 => snake.change_direction(Direction::Left),
                3 => snake.change_direction(Direction::Right),
                _ => (),
            }

            // 3. Вычисляем новую позицию головы
            let (head_x, head_y) = snake.body[0];
            let new_head = match snake.direction {
                Direction::Up => (head_x, head_y.saturating_sub(1)),
                Direction::Down => (head_x, head_y + 1),
                Direction::Left => (head_x.saturating_sub(1), head_y),
                Direction::Right => (head_x + 1, head_y),
            };

            let dead = new_head.0 >= BOARD_WIDTH || new_head.1 >= BOARD_HEIGHT;
            let ate = new_head == food.position;
            let reward = if ate { 1.0 } else if dead { -1.0 } else { 0.0 };

            // 4. Двигаем змейку
            snake.move_forvard(new_head, ate);
            if ate {
                food = Food::new(&snake.body);
                score += 1;
            }

            // 5. Получаем следующее состояние
            let next_state = get_state(&snake, &food);

            // 6. Сохраняем опыт
            agent.store_experience(state, action, reward, next_state);
            agent.train();

            // 7. Обработка проигрыша
            if dead {
                println!("Snake crashed. Score: {}", score);
                agent.log_episode(0, score);
                *control_flow = ControlFlow::Exit;
                result = GameResult::Restart;
                return;
            }

            last_update = Instant::now();
        }

        // Отрисовка
        if let Event::RedrawRequested(_) = event {
            let frame = pixels.frame_mut();
            for pixel in frame.chunks_exact_mut(4) {
                pixel.copy_from_slice(&[0, 0, 0, 255]);
            }

            for &(x, y) in &snake.body {
                draw_cell(x, y, [0, 255, 0, 255], frame);
            }

            let (fx, fy) = food.position;
            draw_cell(fx, fy, [255, 0, 0, 255], frame);

            pixels.render().unwrap();
        }

        window.request_redraw();
    });

    result
}

fn draw_cell(x: u32, y: u32, color: [u8; 4], frame: &mut [u8]) {
    let start_x = x * CELL_SIZE;
    let start_y = y * CELL_SIZE;

    for dy in 0..CELL_SIZE {
        for dx in 0..CELL_SIZE {
            let idx = (((start_y + dy) * WIDTH) + (start_x + dx)) as usize * 4;
            if idx + 4 <= frame.len() {
                frame[idx..idx + 4].copy_from_slice(&color);
            }
        }
    }
}
