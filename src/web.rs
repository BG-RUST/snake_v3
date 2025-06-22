use warp::Filter;
use std::fs;
use serde::Serialize;

#[derive(Serialize)]
struct FitnessPoint {
    generation: u32,
    average: f32,
    best: f32,
}

pub async fn start_web_server() {
    let chart_route = warp::path("chart").map(|| {
        warp::reply::html(include_str!("chart.html"))
    });

    let data_route = warp::path("data").map(|| {
        let content = fs::read_to_string("fitness_log.csv").unwrap_or_default();
        let data: Vec<FitnessPoint> = content.lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() == 3 {
                    Some(FitnessPoint {
                        generation: parts[0].parse().ok()?,
                        average: parts[1].parse().ok()?,
                        best: parts[2].parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
        warp::reply::json(&data)
    });

    let routes = chart_route.or(data_route);

    println!("📈 Веб-сервер запущен на http://localhost:3030/chart");
    warp::serve(routes).run(([127, 0, 0, 1], 3030)).await;
}
