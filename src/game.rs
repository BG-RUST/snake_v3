use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};
use pixels::{Pixels, SurfaceTexture};
use std::sync::{Arc, Mutex};

use crate::snake::*;
use crate::food::*;
use crate::net::NeuralNet;

pub fn run_game_loop<T>(window: Window, event_loop: EventLoop<T>, best_net: Arc<Mutex<NeuralNet>>)
where
    T: 'static,
{
    let window_size = window.inner_size();
    let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
    let cell_size = 20;

    // Инициализируем механизм рисования пикселей
    let mut pixels = Pixels::new(window_size.width, window_size.height, surface_texture)
        .expect("Не удалось создать Pixel buffer");

    // Создаем змейку и еду
    let mut snake = Snake::new();
    let mut food = Food::new(window_size.width, window_size.height, cell_size);

    let mut last_update = std::time::Instant::now();
    // Можно уменьшить tick_rate для более быстрой игры,
    // чтобы быстрее наблюдать обучение (например, 50 мс вместо 150 мс).
    let tick_rate = std::time::Duration::from_millis(150);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll; // Постоянно перерисовывать без задержки (Poll)

        match event {
            // Обработка событий окна
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    println!("Окно закрывается...");
                    *control_flow = ControlFlow::Exit;
                }
                _ => {} // Пропускаем остальные события окна
            },

            // Главное обновление логики игры
            Event::MainEventsCleared => {
                if last_update.elapsed() > tick_rate {
                    // Каждые tick_rate миллисекунд делаем шаг игры
                    // Получаем решение от нейросети: в каком направлении двигаться змейке
                    let direction = {
                        // Блокируем мьютекс, чтобы получить доступ к лучшей нейросети
                        let net = best_net.lock().unwrap();
                        // Вычисляем направление движения на основе текущего состояния игры (змейки и еды)
                        net.decide_direction(&snake, &food, window_size.width, window_size.height)
                    };
                    // Меняем направление движения змейки на предложенное нейросетью
                    snake.change_dir(direction);
                    // Обновляем состояние змейки (перемещаем ее на клетку вперед)
                    snake.update();

                    // Проверяем, съела ли змейка еду (совпадает ли голова с позицией еды)
                    if snake.body[0] == food.position {
                        snake.grow(); // увеличиваем длину змейки
                        food = Food::new(window_size.width, window_size.height, cell_size); // создаем новую еду в случайной позиции
                    }
                    //буду по новой переделівать
                    // Проверяем столкновения со стенами (границами поля) или своим телом
                    let (head_x, head_y) = snake.body[0];
                    let cols = window_size.width / cell_size;
                    let rows = window_size.height / cell_size;
                    // Столкновение с границей: если голова вышла на рамку (координаты 1 или крайние)
                    let hit_wall = head_x == 1 || head_y == 1 || head_x >= cols - 1 || head_y >= rows - 1;
                    // Столкновение с собственным телом: если голова совпала с любой другой частью тела
                    let hit_self = snake.is_colliding_with_self();
                    if hit_wall || hit_self {
                        println!("Game Over! (змейка погибла)");
                        // Перезапускаем игру: создаем новую змейку и новую еду
                        snake = Snake::new();
                        food = Food::new(window_size.width, window_size.height, cell_size);
                        // (Мы НЕ выходим из игры, а продолжаем, чтобы наблюдать непрерывное обучение)
                    }

                    // Обновляем время последнего шага
                    last_update = std::time::Instant::now();
                }
                // Запрашиваем перерисовку окна после обновления логики
                window.request_redraw();
            }

            // Рендеринг (отрисовка) каждого кадра
            Event::RedrawRequested(_) => {
                let frame = pixels.frame_mut();
                // Заполняем фон темно-серым цветом
                fill_background(frame, [30, 30, 30, 255]);
                // Рисуем рамку (границы поля) другим цветом (светло-красным)
                draw_borders(frame, window_size.width, window_size.height, cell_size, [255, 120, 120, 233]);

                // Рисуем змейку (зеленым цветом) и еду (красным) на текущем кадре
                snake.draw(frame, cell_size, window_size.width);
                food.draw(frame, cell_size, window_size.width);

                // Отправляем буфер кадра в окно
                pixels.render().expect("Ошибка при отрисовке frame");
            }

            _ => {}
        }
    });
}


fn fill_background(frame: &mut [u8], color: [u8; 4]) {
    for chunk in frame.chunks_exact_mut(4) {
        chunk.copy_from_slice(&color);
    }
}

fn draw_borders(frame: &mut [u8], width: u32, height: u32, cell_size: u32, color: [u8; 4]) {
    let screen_width = width as usize;
    let screen_height = height as usize;
    let cell = cell_size as usize;

    for y in 0..screen_height / cell {
        for x in 0..screen_width / cell {
            let is_border = x == 0 || y == 0 || x == (screen_width / cell - 1) || y == (screen_height / cell - 1);

            if is_border {
                for py in 0..cell {
                    for px in 0..cell {
                        let fx = x * cell + px;
                        let fy = y * cell + py;
                        let index = (fy * screen_width + fx) * 4;

                        if index + 3 < frame.len() {
                            frame[index..index + 4].copy_from_slice(&color);
                        }
                    }
                }
            }
        }
    }
}

pub fn draw_cell(
    frame: &mut [u8],
    x: u32,
    y: u32,
    cell_size: u32,
    screen_width: u32,
    color: [u8; 4],
) {
    let cell = cell_size as usize;
    let width = screen_width as usize;

    for py in 0..cell {
        for px in 0..cell {
            let fx = (x * cell_size + px as u32) as usize;
            let fy = (y * cell_size + py as u32) as usize;
            let index = (fy * width + fx) * 4;

            if index + 3 < frame.len() {
                frame[index..index + 4].copy_from_slice(&color);
            }
        }
    }
}