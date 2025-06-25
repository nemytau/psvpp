use std::time::{Instant, Duration};
use std::fs::File;
use std::io::{Write, BufWriter};

#[derive(Debug, Clone)]
pub struct IterationLog {
    pub iteration: usize,
    pub current_cost: f64,
    pub best_cost: f64,
    pub destroy_idx: usize,
    pub repair_idx: usize,
    pub destroy_weight: f64,
    pub repair_weight: f64,
    pub accepted: bool,
    pub temperature: f64,
    pub duration_ms: u128,
}

pub struct AlnsLogger {
    logs: Vec<IterationLog>,
    start_time: Instant,
}

impl AlnsLogger {
    pub fn new() -> Self {
        AlnsLogger {
            logs: Vec::new(),
            start_time: Instant::now(),
        }
    }

    pub fn log_iteration(&mut self,
        iteration: usize,
        current_cost: f64,
        best_cost: f64,
        destroy_idx: usize,
        repair_idx: usize,
        destroy_weight: f64,
        repair_weight: f64,
        accepted: bool,
        temperature: f64,
        duration: Duration,
    ) {
        let log = IterationLog {
            iteration,
            current_cost,
            best_cost,
            destroy_idx,
            repair_idx,
            destroy_weight,
            repair_weight,
            accepted,
            temperature,
            duration_ms: duration.as_millis(),
        };
        self.logs.push(log);
    }

    pub fn export_csv(&self, path: &str) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "iteration,current_cost,best_cost,destroy_idx,repair_idx,destroy_weight,repair_weight,accepted,temperature,duration_ms")?;
        for log in &self.logs {
            writeln!(
                writer,
                "{},{},{},{},{},{},{},{},{},{}",
                log.iteration,
                log.current_cost,
                log.best_cost,
                log.destroy_idx,
                log.repair_idx,
                log.destroy_weight,
                log.repair_weight,
                log.accepted,
                log.temperature,
                log.duration_ms
            )?;
        }
        Ok(())
    }
}
