use std::fs::File;
use std::io::Write;

use serde::Serialize;

use crate::structs::constants::HOURS_IN_PERIOD;
use crate::structs::context::Context;
use crate::structs::distance_manager;
use crate::structs::solution::Solution;
use crate::structs::vessel::Vessel;
use crate::structs::voyage::Voyage;

use super::tsp_solver;

// Structs for explicit schedule visualization
#[derive(Serialize)]
struct ExplicitStage {
    #[serde(rename = "Vessel")]
    vessel: String,
    #[serde(rename = "Start")]
    start: f64,
    #[serde(rename = "End")]
    end: f64,
    #[serde(rename = "Action")]
    action: String,
    #[serde(rename = "Description")]
    description: String,
}

#[derive(Serialize)]
struct ExplicitSchedule {
    stages: Vec<ExplicitStage>,
    cost: f64,
}

// Structs for simplified schedule visualization
#[derive(Serialize)]
struct VizVoyage {
    #[serde(rename = "Vessel")]
    vessel: String,
    #[serde(rename = "Route")]
    route: String,
    #[serde(rename = "Start")]
    start: f64,
    #[serde(rename = "End")]
    end: f64,
    #[serde(rename = "Load")]
    load: String,
}

#[derive(Serialize)]
struct VizSchedule {
    voyages: Vec<VizVoyage>,
    cost: f64,
}

fn push_stage_cyclic(
    stages: &mut Vec<ExplicitStage>,
    vessel: &str,
    start: f64,
    end: f64,
    action: &str,
    description: &str,
) {
    if end <= HOURS_IN_PERIOD as f64 {
        stages.push(ExplicitStage {
            vessel: vessel.to_string(),
            start,
            end,
            action: action.to_string(),
            description: description.to_string(),
        });
    } else {
        stages.push(ExplicitStage {
            vessel: vessel.to_string(),
            start,
            end: HOURS_IN_PERIOD as f64,
            action: action.to_string(),
            description: description.to_string(),
        });
        stages.push(ExplicitStage {
            vessel: vessel.to_string(),
            start: 0.0,
            end: end - HOURS_IN_PERIOD as f64,
            action: action.to_string(),
            description: description.to_string(),
        });
    }
}

fn advance_time(current: f64, delta: f64) -> f64 {
    let next = current + delta;
    if next > HOURS_IN_PERIOD as f64 {
        next - HOURS_IN_PERIOD as f64
    } else {
        next
    }
}

pub fn dump_explicit_schedule_to_json(
    solution: &Solution,
    context: &Context,
    output_path: &str,
) {
    let mut stages = Vec::new();
    let base = &context.problem.base;
    let vessels = &context.problem.vessels;
    let distance_manager = &context.problem.distance_manager;
    let tsp_solver = &context.tsp_solver;

    for voyage_cell in &solution.voyages {
        let voyage = voyage_cell.borrow();
        let vessel = vessels
            .iter()
            .find(|v| v.id == voyage.vessel_id.unwrap())
            .unwrap();
        let visits = voyage.visit_ids
            .iter()
            .filter_map(|&idx| solution.visit(idx).cloned())
            .collect::<Vec<_>>();
        let mut current_time = voyage.start_time().unwrap_or(0.0);
        let speed = vessel.speed;

        // Service at base before departure
        push_stage_cyclic(
            &mut stages,
            &vessel.name,
            current_time - base.service_time,
            current_time,
            "Service at base",
            "base",
        );
        current_time = advance_time(current_time, 0.0);

        for (i, visit) in visits.iter().enumerate() {
            let from_inst_id = if i > 0 {
                visits[i - 1].installation_id()
            } else {
                base.id
            };
            let to_inst_id = visit.installation_id();
            let distance = distance_manager.distance(from_inst_id, to_inst_id);
            let travel_time = distance / speed;

            push_stage_cyclic(
                &mut stages,
                &vessel.name,
                current_time,
                current_time + travel_time,
                "Sailing",
                &format!("{}→{}", from_inst_id, to_inst_id),
            );
            current_time = advance_time(current_time, travel_time);

            let arrival_time = current_time;
            if let Some(wait) = tsp_solver.compute_wait_time(to_inst_id, arrival_time) {
                if wait > 0.0 {
                    push_stage_cyclic(
                        &mut stages,
                        &vessel.name,
                        current_time,
                        current_time + wait,
                        "Waiting",
                        &format!("{}", to_inst_id),
                    );
                    current_time = advance_time(current_time, wait);
                }
            }

            let service_time = context.problem.installations[to_inst_id - 1].service_time;
            push_stage_cyclic(
                &mut stages,
                &vessel.name,
                current_time,
                current_time + service_time,
                "Service",
                &format!("{}", to_inst_id),
            );
            current_time = advance_time(current_time, service_time);
        }

        if let Some(last_visit) = visits.last() {
            let distance = distance_manager.distance(last_visit.installation_id(), base.id);
            let travel_time = distance / speed;
            push_stage_cyclic(
                &mut stages,
                &vessel.name,
                current_time,
                current_time + travel_time,
                "Sailing",
                &format!("{}→base", last_visit.installation_id()),
            );
            current_time = advance_time(current_time, travel_time);

            if let Some(wait) = tsp_solver.compute_wait_time(base.id, current_time) {
                if wait > 0.0 {
                    push_stage_cyclic(
                        &mut stages,
                        &vessel.name,
                        current_time,
                        current_time + wait,
                        "Waiting",
                        &format!("{}", base.id),
                    );
                    current_time = advance_time(current_time, wait);
                }
            }
        }
    }

    // Calculate cost with context, not just solution.total_cost
    let cost = solution.cost_with_context(context);
    let json = serde_json::to_string_pretty(&ExplicitSchedule { stages, cost }).unwrap();
    let mut file = File::create(output_path).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}

pub fn dump_schedule_to_json(solution: &Solution, vessels: &[Vessel], output_path: &str, context: &Context) {
    let mut viz_voyages = Vec::new();

    for voyage_cell in &solution.voyages {
        let voyage = voyage_cell.borrow();
        let vessel_id = voyage.vessel_id.expect("Voyage missing vessel ID");
        let vessel = vessels.iter().find(|v| v.id == vessel_id).expect("Vessel not found");

        let route = voyage.visit_ids
            .iter()
            .filter_map(|&idx| solution.visit(idx).map(|v| v.installation_id()))
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join("→");
        let load = format!("{}/{}", voyage.load().unwrap_or(0), vessel.deck_capacity);

        if voyage.end_time_at_base > Some(HOURS_IN_PERIOD as f64) {
            viz_voyages.push(VizVoyage {
                vessel: vessel.name.clone(),
                route: route.clone(),
                start: voyage.start_time().unwrap_or(0.0),
                end: HOURS_IN_PERIOD as f64,
                load: load.clone(),
            });
            viz_voyages.push(VizVoyage {
                vessel: vessel.name.clone(),
                route: route,
                start: 0.0,
                end: voyage.end_time_at_base.unwrap() - HOURS_IN_PERIOD as f64,
                load: load,
            });
        } else {
            viz_voyages.push(VizVoyage {
                vessel: vessel.name.clone(),
                route: route,
                start: voyage.start_time().unwrap_or(0.0),
                end: voyage.end_time_at_base.unwrap(),
                load: load,
            });
        }
    }
    
    // Calculate cost with context, not just solution.total_cost
    let cost = solution.cost_with_context(context);
    let json_schedule = VizSchedule { voyages: viz_voyages, cost };
    let json = serde_json::to_string_pretty(&json_schedule).expect("Failed to serialize");
    let mut file = File::create(output_path).expect("Failed to create file");
    file.write_all(json.as_bytes()).expect("Failed to write to file");
}