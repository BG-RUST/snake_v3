use rusqlite::{ Connection, params};
use serde::{ Serialize, Deserialize };
use chrono::{ Utc };

#[derive(Serialize, Deserialize, Debug)]
pub struct Individual {
    pub generation: usize,
    pub genome: Vec<f32>,//weights
    pub fitness: f32,
    pub steps: usize,
    pub eaten: usize,
}

pub fn init_db(path: &str) -> Connection {
    let conn = Connection::open(path).expext("Failed to open DB");
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS individuals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,   -- уникальный ID особи
            generation  INTEGER NOT NULL,                    -- номер поколения
            genome      TEXT NOT NULL,                       -- сериализованный JSON со списком весов
            fitness     REAL NOT NULL,                       -- оценка
            steps       INTEGER NOT NULL,                    -- кол-во шагов
            eaten       INTEGER NOT NULL,                    -- сколько еды съела
            timestamp   TEXT NOT NULL                        -- когда была записана
        );
        "
    ).expect("Failed to create DB");
    conn
}

pub fn insert_individual(conn: &Connection, ind: &Individual) {
    let genome_json = serde_json::to_string(&ind.genome).expect("Serialization failed");
    //get current date\time
    let now = Utc::now().to_rfc3339();
    conn.execute(
        "
        INSERT INTO individuals (generation, genome, fitness, steps, eaten, timestamp)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6);
        ",
        params![
            ind.generation,  // ?1
            genome_json,     // ?2
            ind.fitness,     // ?3
            ind.steps,       // ?4
            ind.eaten,       // ?5
            now              // ?6
        ],
    ).expect("Insert failed");
}

pub fn get_best_individual(c0nn: &Connection) -> Option<Individual> {
    let mut stmt = conn.prepare(
        "
        SELECT generation, genome, fitness, steps, eaten
        FROM individuals
        ORDER BY fitness DESC
        LIMIT 1;
        "
    ).ok()?;

    let result = stmt.query_row([], |row| {
        let genome_json: String = row.get(1)?;
        let genome: Vec<f32> = serde_json::from_str(&genome_json).unwrap_or_default();

        Ok(Individual {
            generation: row.get(0)?,
            genome,
            fitness: row.get(2)?,
            steps: row.get(3)?,
            eaten: row.get(4)?,
        })
    }).ok();

    result
}