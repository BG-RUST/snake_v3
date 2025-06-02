use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent, VirtualKeyCode},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use winit_input_helper::WinitInputHelper;
use winit::platform::run_return::EventLoopExtRunReturn;

use std::time::{Duration, Instant};
use pixels::{Pixels, SurfaceTexture};

//game modules
use crate::food::Food;
use crate::snake::{Direction, Snake, BOARD_HEIGHT, BOARD_WIDTH};
//size 1 cell in pixels

const CELL_SIZE: u32 = 32;
const WIDTH: u32 = BOARD_WIDTH * CELL_SIZE;
const HEIGHT: u32 = BOARD_HEIGHT * CELL_SIZE;

#[derive(PartialEq, Copy, Clone)]
pub enum GameResult {
    Restart,
    Exit,
}

pub fn start(event_loop: &mut EventLoop<()>) -> GameResult {
    //window create
    //let mut event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Snake in Rust + Pixels")
        .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
        .build(&event_loop)
        .unwrap();
    //let window = event_loop.create_window(window_attributes).unwrap();

    //setup paint buffer

    let surface_texture = SurfaceTexture::new(WIDTH, HEIGHT, &window);
    let mut pixels = Pixels::new(WIDTH, HEIGHT, surface_texture).unwrap();
    let mut input = WinitInputHelper::new();

    let mut result = GameResult::Restart;

    // creagte snake and food

    let mut snake = Snake::new();
    let mut food = Food::new(&snake.body);

    //timer for logic update
    let mut last_update = Instant::now();
    let update_interval = Duration::from_millis(150); //snake speed

    //start game loop
    event_loop.run_return(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
//обработка нажатия клавиш
        if input.update(&event) {
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }
            // обновляем направления движения
            if input.key_pressed(VirtualKeyCode::Up) {
                snake.change_direction(Direction::Up);
            }
            if input.key_pressed(VirtualKeyCode::Down) {
                snake.change_direction(Direction::Down);
            }
            if input.key_pressed(VirtualKeyCode::Left) {
                snake.change_direction(Direction::Left);
            }
            if input.key_pressed(VirtualKeyCode::Right) {
                snake.change_direction(Direction::Right);
            }
        }

        //двигаем змейку по таймеру
        if last_update.elapsed() > update_interval {
            let (head_x, head_y) = snake.body[0];

            //new head position
            let new_head = match snake.direction {
                Direction::Up => (head_x, head_y.saturating_sub(1)),
                Direction::Down => (head_x, head_y + 1),
                Direction::Left => (head_x.saturating_sub(1), head_y),
                Direction::Right => (head_x + 1, head_y),
            };
            //проверка границ
            if new_head.0 >= BOARD_WIDTH || new_head.1 >= BOARD_HEIGHT {
                println!("Snake game over!");
                result = GameResult::Restart;
                *control_flow = ControlFlow::Exit;
               // return;
            }

            let ate = new_head == food.position;
            if ate {
                food = Food::new(&snake.body);
            }

            //двигаем змейку с учетом поедания еді
            snake.move_forvard(new_head, ate);
            last_update = Instant::now();
        }

        //отрисовываем текущий кадр
        if let Event::RedrawRequested(_) = event {
            let frame = pixels.frame_mut();
            for pixel in frame.chunks_exact_mut(4) {
                pixel.copy_from_slice(&[0x00, 0x00, 0x00, 0xff]); // чёрный фон

            }

            //paint snake body
            for &(x, y) in &snake.body {
                draw_cell(x, y, [0x00, 0xff, 0x00, 0xff], frame);
            }

            // Рисуем еду красным
            let (fx, fy) = food.position;
            draw_cell(fx, fy, [0xff, 0x00, 0x00, 0xff], frame);

            pixels.render().unwrap();
        }

        // Явно запрашиваем перерисовку окна
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
            frame[idx..idx + 4].copy_from_slice(&color);
        }
    }
}