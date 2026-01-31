use crate::operators::repair::k_regret_insertion::KRegretInsertion;
use crate::operators::traits::{ImprovementOperator, RepairOperator};
use crate::structs::constants::HOURS_IN_PERIOD;
use crate::structs::{context::Context, solution::Solution};
use crate::utils::serialization::dump_schedule_to_json;
use log::{info, warn};
use rand::RngCore;
use std::collections::{HashMap, HashSet};
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;

pub struct FleetAndCostReduction;

impl FleetAndCostReduction {
    pub const LAMBDA_MAX: u32 = 72;
}

struct DebugSnapshotWriter {
    root: PathBuf,
}

impl DebugSnapshotWriter {
    fn new() -> Option<Self> {
        let candidate = std::env::var_os("ALNS_MOVES_DIR")
            .or_else(|| std::env::var_os("ALNS_DEBUG_MOVES_DIR"))
            .or_else(|| std::env::var_os("ALNS_DEBUG_DIR"));
        let Some(path) = candidate else {
            return None;
        };
        let root = PathBuf::from(path);
        if root.as_os_str().is_empty() {
            return None;
        }
        if let Err(err) = std::fs::create_dir_all(&root) {
            warn!(
                "fleet_and_cost_reduction: unable to create iteration snapshot directory {}: {}",
                root.display(),
                err
            );
            return None;
        }
        info!(
            "fleet_and_cost_reduction: writing iteration snapshots to {}",
            root.display()
        );
        Some(Self { root })
    }

    fn capture(&self, iteration: usize, stage: &str, solution: &Solution, context: &Context) {
        let filename = format!("iteration_{:03}_{}.json", iteration, stage);
        let path = self.root.join(filename);
        if let Some(parent) = path.parent() {
            if let Err(err) = std::fs::create_dir_all(parent) {
                warn!(
                    "fleet_and_cost_reduction: failed to create snapshot parent directory {}: {}",
                    parent.display(),
                    err
                );
                return;
            }
        }
        let Some(path_str) = path.to_str() else {
            warn!(
                "fleet_and_cost_reduction: snapshot path is not valid UTF-8: {}",
                path.display()
            );
            return;
        };
        let vessels = &context.problem.vessels;
        let write_result = std::panic::catch_unwind(AssertUnwindSafe(|| {
            dump_schedule_to_json(solution, vessels, path_str, context);
        }));
        if write_result.is_err() {
            warn!(
                "fleet_and_cost_reduction: failed to dump snapshot for iteration {} stage {}",
                iteration, stage
            );
        }
    }
}

impl ImprovementOperator for FleetAndCostReduction {
    fn apply(&self, solution: &mut Solution, context: &Context, rng: &mut dyn RngCore) {
        solution.ensure_consistency_updated(context);
        let mut previous_cost = solution.cost_with_context(context);
        let debug_writer = DebugSnapshotWriter::new();
        info!(
            "fleet_and_cost_reduction: starting iteration with cost {:.2}",
            previous_cost
        );
        let describe_voyage = |sol: &Solution, voyage_id: usize| -> String {
            let schedule = &sol.schedule;
            let start = schedule
                .voyage_start_times
                .get(&voyage_id)
                .copied()
                .unwrap_or(f64::NAN);
            let end = schedule
                .voyage_end_times
                .get(&voyage_id)
                .copied()
                .unwrap_or(f64::NAN);
            let visits = sol
                .voyages
                .iter()
                .find_map(|cell| {
                    let voyage = cell.borrow();
                    if voyage.id == voyage_id {
                        Some((
                            voyage.vessel_id,
                            voyage.departure_day,
                            voyage.visit_ids.clone(),
                        ))
                    } else {
                        None
                    }
                })
                .unwrap_or((None, None, Vec::new()));
            let installation_ids: Vec<usize> = visits
                .2
                .iter()
                .filter_map(|visit_id| sol.visit(*visit_id).map(|visit| visit.installation_id()))
                .collect();
            format!(
                "voyage {} vessel {:?} dep_day {:?} installs {:?} visit_ids {:?} start {:.2} end {:.2}",
                voyage_id, visits.0, visits.1, installation_ids, visits.2, start, end
            )
        };
        struct OverlapInfo {
            target_voyage_id: usize,
            target_start: f64,
            target_end: f64,
            target_is_empty: bool,
            overlap_hours: f64,
        }
        struct OverlapRecord {
            origin_vessel_id: usize,
            target_vessel_id: usize,
            voyage_id: usize,
            voyage_start: f64,
            voyage_end: f64,
            voyage_is_empty: bool,
            overlaps: Vec<OverlapInfo>,
            total_overlap_hours: f64,
        }
        let mut iteration = 0usize;
        loop {
            iteration += 1;
            if let Some(writer) = &debug_writer {
                writer.capture(iteration, "before", solution, context);
            }
            let mut used_vessels = Vec::new();
            for vessel_id in 0..context.problem.vessels.len() {
                let assigned_voyages = solution.schedule.get_all_voyages_for_vessel(vessel_id);
                if assigned_voyages.iter().any(|voyage_id| {
                    solution.voyages.iter().any(|cell| {
                        let v = cell.borrow();
                        v.id == *voyage_id && !v.visit_ids.is_empty()
                    })
                }) {
                    used_vessels.push((vessel_id, assigned_voyages));
                }
            }

            let mut overlap_records: Vec<OverlapRecord> = Vec::new();
            let mut least_overlap_targets: Vec<(usize, Option<(usize, f64)>)> = Vec::new();
            let mut origin_least_overlap_totals: Vec<(usize, f64)> = Vec::new();
            for (origin_vessel_id, origin_voyages) in &used_vessels {
                let mut vessel_min_overlap_sum = 0.0;
                for voyage_id in origin_voyages {
                    let Some(&voyage_start) = solution.schedule.voyage_start_times.get(voyage_id)
                    else {
                        continue;
                    };
                    let Some(&voyage_end) = solution.schedule.voyage_end_times.get(voyage_id)
                    else {
                        continue;
                    };
                    let voyage_is_empty = solution.is_empty_voyage_by_id(*voyage_id);
                    info!(
                        "fleet_and_cost_reduction iteration {}: origin vessel {} considering {}",
                        iteration,
                        origin_vessel_id,
                        describe_voyage(solution, *voyage_id)
                    );
                    let mut best_target: Option<(usize, f64)> = None;
                    for (target_vessel_id, target_voyages) in &used_vessels {
                        if target_vessel_id == origin_vessel_id {
                            continue;
                        }
                        let mut overlaps = Vec::new();
                        let mut total_overlap_hours = 0.0;
                        for other_voyage_id in target_voyages {
                            if other_voyage_id == voyage_id {
                                continue;
                            }
                            let Some(&other_start) =
                                solution.schedule.voyage_start_times.get(other_voyage_id)
                            else {
                                continue;
                            };
                            let Some(&other_end) =
                                solution.schedule.voyage_end_times.get(other_voyage_id)
                            else {
                                continue;
                            };
                            let overlap_hours = cyclic_overlap_duration(
                                voyage_start,
                                voyage_end,
                                other_start,
                                other_end,
                                HOURS_IN_PERIOD as f64,
                            );
                            if overlap_hours > f64::EPSILON {
                                let target_is_empty =
                                    solution.is_empty_voyage_by_id(*other_voyage_id);
                                total_overlap_hours += overlap_hours;
                                overlaps.push(OverlapInfo {
                                    target_voyage_id: *other_voyage_id,
                                    target_start: other_start,
                                    target_end: other_end,
                                    target_is_empty,
                                    overlap_hours,
                                });
                            }
                        }
                        let candidate_target_id = *target_vessel_id;
                        let candidate_overlap = total_overlap_hours;
                        let should_replace = match &best_target {
                            None => true,
                            Some((best_id, best_overlap)) => {
                                if candidate_overlap + f64::EPSILON < *best_overlap {
                                    true
                                } else {
                                    let overlap_diff = (candidate_overlap - *best_overlap).abs();
                                    overlap_diff <= f64::EPSILON && candidate_target_id < *best_id
                                }
                            }
                        };
                        if should_replace {
                            best_target = Some((candidate_target_id, candidate_overlap));
                        }
                        overlap_records.push(OverlapRecord {
                            origin_vessel_id: *origin_vessel_id,
                            target_vessel_id: *target_vessel_id,
                            voyage_id: *voyage_id,
                            voyage_start,
                            voyage_end,
                            voyage_is_empty,
                            overlaps,
                            total_overlap_hours,
                        });
                    }
                    if let Some((_, overlap)) = best_target {
                        vessel_min_overlap_sum += overlap;
                    }
                    least_overlap_targets.push((*voyage_id, best_target));
                }
                origin_least_overlap_totals.push((*origin_vessel_id, vessel_min_overlap_sum));
            }
            let origin_with_least_overlap = origin_least_overlap_totals
                .iter()
                .min_by(|a, b| {
                    let diff = (a.1 - b.1).abs();
                    if diff <= f64::EPSILON {
                        a.0.cmp(&b.0)
                    } else if a.1 < b.1 {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Greater
                    }
                })
                .copied();
            if overlap_records.is_empty() {
                // Placeholder: overlap data will inform upcoming move selection logic
            }
            if least_overlap_targets.is_empty() {
                // Placeholder: target selection will guide reassignment decisions
            }
            if origin_least_overlap_totals.is_empty() {
                // Placeholder: per-vessel totals enable prioritizing origin vessels for reassignment
            }
            if let Some((origin_vessel_id, overlap_total)) = origin_with_least_overlap {
                info!(
					"fleet_and_cost_reduction iteration {}: evaluating origin vessel {} with min_overlap {:.2}",
					iteration,
					origin_vessel_id,
					overlap_total
				);
                if overlap_total + f64::EPSILON < Self::LAMBDA_MAX as f64 {
                    let target_lookup: HashMap<usize, (usize, f64)> = least_overlap_targets
                        .iter()
                        .filter_map(|(voyage_id, best)| best.map(|target| (*voyage_id, target)))
                        .collect();
                    let Some(origin_entry) = used_vessels
                        .iter()
                        .find(|(vessel_id, _)| *vessel_id == origin_vessel_id)
                    else {
                        if let Some(writer) = &debug_writer {
                            writer.capture(iteration, "after", solution, context);
                        }
                        continue;
                    };
                    let mut candidate_solution = solution.clone();
                    let mut reassignment_failed = false;
                    info!(
						"fleet_and_cost_reduction iteration {}: attempting reassignment of vessel {} to {} candidates",
						iteration,
						origin_vessel_id,
						target_lookup.len()
					);
                    let mut removed_visit_ids: HashSet<usize> = HashSet::new();
                    for voyage_id in origin_entry.1.clone() {
                        if solution.is_empty_voyage_by_id(voyage_id) {
                            continue;
                        }
                        let origin_voyage_before = describe_voyage(solution, voyage_id);
                        let Some(&(target_vessel_id, target_overlap)) =
                            target_lookup.get(&voyage_id)
                        else {
                            warn!(
								"fleet_and_cost_reduction iteration {}: no target vessel found for voyage {}",
								iteration,
								voyage_id
							);
                            reassignment_failed = true;
                            break;
                        };
                        let Some(voyage_index) = candidate_solution
                            .voyages
                            .iter()
                            .position(|cell| cell.borrow().id == voyage_id)
                        else {
                            reassignment_failed = true;
                            break;
                        };
                        let (previous_vessel, departure_day) = {
                            let mut voyage = candidate_solution.voyages[voyage_index].borrow_mut();
                            let prev_vessel = voyage.vessel_id;
                            let dep_day = voyage.departure_day;
                            voyage.vessel_id = Some(target_vessel_id);
                            (prev_vessel, dep_day)
                        };
                        let mut displaced_voyage: Option<usize> = None;
                        if let Some(day) = departure_day {
                            if let Some(old_vessel) = previous_vessel {
                                candidate_solution
                                    .schedule
                                    .vessel_day_voyages
                                    .remove(&(old_vessel, day));
                            }
                            displaced_voyage = candidate_solution
                                .schedule
                                .vessel_day_voyages
                                .insert((target_vessel_id, day), voyage_id);
                        }
                        candidate_solution.schedule.set_need_update(true);
                        let mut start_time = match candidate_solution
                            .schedule
                            .voyage_start_times
                            .get(&voyage_id)
                        {
                            Some(value) => *value,
                            None => {
                                reassignment_failed = true;
                                break;
                            }
                        };
                        let mut end_time =
                            match candidate_solution.schedule.voyage_end_times.get(&voyage_id) {
                                Some(value) => *value,
                                None => {
                                    reassignment_failed = true;
                                    break;
                                }
                            };
                        if target_overlap > f64::EPSILON {
                            let origin_voyage_after =
                                describe_voyage(&candidate_solution, voyage_id);
                            info!(
                                "fleet_and_cost_reduction iteration {}: origin {} reassigns to {} and overlaps with target vessel {} (overlap {:.2}h)",
                                iteration,
                                origin_voyage_before,
                                origin_voyage_after,
                                target_vessel_id,
                                target_overlap
                            );
                            loop {
                                let mut has_overlap =
                                    candidate_solution.schedule.overlaps_with_other_voyages(
                                        target_vessel_id,
                                        voyage_id,
                                        start_time,
                                        end_time,
                                        Some(|id| candidate_solution.is_empty_voyage_by_id(id)),
                                    );
                                if !has_overlap {
                                    if let Some(displaced) = displaced_voyage {
                                        if displaced != voyage_id
                                            && !candidate_solution.is_empty_voyage_by_id(displaced)
                                        {
                                            if let (Some(&other_start), Some(&other_end)) = (
                                                candidate_solution
                                                    .schedule
                                                    .voyage_start_times
                                                    .get(&displaced),
                                                candidate_solution
                                                    .schedule
                                                    .voyage_end_times
                                                    .get(&displaced),
                                            ) {
                                                if cyclic_overlap_duration(
                                                    start_time,
                                                    end_time,
                                                    other_start,
                                                    other_end,
                                                    HOURS_IN_PERIOD as f64,
                                                ) > f64::EPSILON
                                                {
                                                    has_overlap = true;
                                                }
                                            }
                                        }
                                    }
                                }
                                if !has_overlap {
                                    break;
                                }
                                let mut overlapping_voyages: Vec<usize> = candidate_solution
                                    .schedule
                                    .get_all_voyages_for_vessel(target_vessel_id)
                                    .into_iter()
                                    .filter(|other_id| *other_id != voyage_id)
                                    .filter(|other_id| {
                                        if let (Some(&other_start), Some(&other_end)) = (
                                            candidate_solution
                                                .schedule
                                                .voyage_start_times
                                                .get(other_id),
                                            candidate_solution
                                                .schedule
                                                .voyage_end_times
                                                .get(other_id),
                                        ) {
                                            cyclic_overlap_duration(
                                                start_time,
                                                end_time,
                                                other_start,
                                                other_end,
                                                HOURS_IN_PERIOD as f64,
                                            ) > f64::EPSILON
                                        } else {
                                            false
                                        }
                                    })
                                    .collect();
                                if let Some(displaced) = displaced_voyage {
                                    if displaced != voyage_id
                                        && !candidate_solution.is_empty_voyage_by_id(displaced)
                                    {
                                        if let (Some(&other_start), Some(&other_end)) = (
                                            candidate_solution
                                                .schedule
                                                .voyage_start_times
                                                .get(&displaced),
                                            candidate_solution
                                                .schedule
                                                .voyage_end_times
                                                .get(&displaced),
                                        ) {
                                            if cyclic_overlap_duration(
                                                start_time,
                                                end_time,
                                                other_start,
                                                other_end,
                                                HOURS_IN_PERIOD as f64,
                                            ) > f64::EPSILON
                                                && !overlapping_voyages
                                                    .iter()
                                                    .any(|id| *id == displaced)
                                            {
                                                overlapping_voyages.push(displaced);
                                            }
                                        }
                                    }
                                }
                                if overlapping_voyages.is_empty() {
                                    warn!(
                                        "fleet_and_cost_reduction iteration {}: expected overlapping voyages but none found",
                                        iteration
                                    );
                                    reassignment_failed = true;
                                    break;
                                }
                                let overlaps_are_empty = overlapping_voyages
                                    .iter()
                                    .all(|id| candidate_solution.is_empty_voyage_by_id(*id));
                                let overlapping_targets: Vec<String> = overlapping_voyages
                                    .iter()
                                    .map(|id| describe_voyage(&candidate_solution, *id))
                                    .collect();
                                let origin_voyage_current =
                                    describe_voyage(&candidate_solution, voyage_id);
                                for other_id in &overlapping_voyages {
                                    info!(
                                        "fleet_and_cost_reduction iteration {}: overlap between origin {} (was {}) and target {}",
                                        iteration,
                                        origin_voyage_current,
                                        origin_voyage_before,
                                        describe_voyage(&candidate_solution, *other_id)
                                    );
                                }
                                let mut start_candidates: Vec<(usize, f64)> = Vec::new();
                                start_candidates.push((voyage_id, start_time));
                                for other_id in &overlapping_voyages {
                                    if let Some(&other_start) =
                                        candidate_solution.schedule.voyage_start_times.get(other_id)
                                    {
                                        start_candidates.push((*other_id, other_start));
                                    }
                                }
                                let latest_start = start_candidates
                                    .iter()
                                    .fold(f64::NEG_INFINITY, |acc, (_, value)| acc.max(*value));
                                if !latest_start.is_finite() {
                                    warn!(
                                        "fleet_and_cost_reduction iteration {}: unable to determine latest start for overlap pruning",
                                        iteration
                                    );
                                    reassignment_failed = true;
                                    break;
                                }
                                let mut removal_voyage_id: Option<usize> = None;
                                for (candidate_id, candidate_start) in start_candidates {
                                    if (candidate_start - latest_start).abs() <= f64::EPSILON {
                                        removal_voyage_id = match removal_voyage_id {
                                            Some(current) => Some(current.min(candidate_id)),
                                            None => Some(candidate_id),
                                        };
                                    }
                                }
                                let Some(removal_target) = removal_voyage_id else {
                                    warn!(
                                        "fleet_and_cost_reduction iteration {}: no voyage selected for removal despite overlap candidates",
                                        iteration
                                    );
                                    reassignment_failed = true;
                                    break;
                                };
                                if displaced_voyage == Some(removal_target) {
                                    displaced_voyage = None;
                                }
                                let mut visits_to_remove = Vec::new();
                                if let Some(cell) = candidate_solution
                                    .voyages
                                    .iter()
                                    .find(|cell| cell.borrow().id == removal_target)
                                {
                                    let voyage = cell.borrow();
                                    visits_to_remove.extend(voyage.visit_ids.iter().copied());
                                }
                                visits_to_remove.sort_unstable();
                                visits_to_remove.dedup();
                                if visits_to_remove.is_empty() {
                                    warn!(
                                        "fleet_and_cost_reduction iteration {}: no visits selected for removal despite overlap for origin {} (targets: {}), all_overlaps_empty={}",
                                        iteration,
                                        origin_voyage_before,
                                        overlapping_targets.join("; "),
                                        overlaps_are_empty
                                    );
                                    reassignment_failed = true;
                                    break;
                                }
                                let mut installation_ids_to_remove: Vec<usize> = visits_to_remove
                                    .iter()
                                    .filter_map(|visit_id| {
                                        candidate_solution
                                            .visit(*visit_id)
                                            .map(|visit| visit.installation_id())
                                    })
                                    .collect();
                                installation_ids_to_remove.sort_unstable();
                                installation_ids_to_remove.dedup();
                                info!(
                                    "fleet_and_cost_reduction iteration {}: removing {} visit(s) due to overlap (installations {:?})",
                                    iteration,
                                    visits_to_remove.len(),
                                    installation_ids_to_remove
                                );
                                for visit in &visits_to_remove {
                                    let installation = candidate_solution
                                        .visit(*visit)
                                        .map(|v| v.installation_id());
                                    info!(
                                        "fleet_and_cost_reduction iteration {}: marking visit {} (installation {:?}) for removal",
                                        iteration,
                                        visit,
                                        installation
                                    );
                                }
                                for visit in &visits_to_remove {
                                    removed_visit_ids.insert(*visit);
                                }
                                candidate_solution.unassign_visits(&visits_to_remove);
                                candidate_solution.schedule.set_need_update(true);
                                candidate_solution.ensure_consistency_updated(context);
                                if candidate_solution
                                    .schedule
                                    .voyage_start_times
                                    .get(&voyage_id)
                                    .is_none()
                                {
                                    break;
                                }
                                if candidate_solution.is_empty_voyage_by_id(voyage_id) {
                                    break;
                                }
                                if let (Some(&updated_start), Some(&updated_end)) = (
                                    candidate_solution
                                        .schedule
                                        .voyage_start_times
                                        .get(&voyage_id),
                                    candidate_solution.schedule.voyage_end_times.get(&voyage_id),
                                ) {
                                    start_time = updated_start;
                                    end_time = updated_end;
                                } else {
                                    break;
                                }
                                continue;
                            }
                            if reassignment_failed {
                                break;
                            }
                        }
                        if reassignment_failed {
                            break;
                        }
                    }
                    if !reassignment_failed {
                        if !removed_visit_ids.is_empty() {
                            info!(
								"fleet_and_cost_reduction iteration {}: applying K-Regret repair to {} removed visit(s)",
								iteration,
								removed_visit_ids.len()
							);
                            candidate_solution.add_idle_vessel_and_add_empty_voyages(context);
                            let repair_operator = KRegretInsertion { k: 2 };
                            repair_operator.apply(&mut candidate_solution, context, rng);
                            if !candidate_solution.get_unassigned_visits().is_empty() {
                                warn!(
									"fleet_and_cost_reduction iteration {}: repair failed, {} visit(s) remain unassigned",
									iteration,
									candidate_solution.get_unassigned_visits().len()
								);
                                reassignment_failed = true;
                            }
                        }
                    }
                    if reassignment_failed {
                        info!(
                            "fleet_and_cost_reduction iteration {}: reassignment aborted",
                            iteration
                        );
                        if let Some(writer) = &debug_writer {
                            writer.capture(iteration, "after", solution, context);
                        }
                        break;
                    }
                    candidate_solution.ensure_consistency_updated(context);
                    let candidate_cost = candidate_solution.cost_with_context(context);
                    if candidate_cost + f64::EPSILON < previous_cost {
                        info!(
							"fleet_and_cost_reduction iteration {}: improvement accepted {:.2} -> {:.2}",
							iteration,
							previous_cost,
							candidate_cost
						);
                        if let Some(writer) = &debug_writer {
                            writer.capture(iteration, "after", &candidate_solution, context);
                        }
                        *solution = candidate_solution;
                        previous_cost = candidate_cost;
                        continue;
                    }
                    info!(
						"fleet_and_cost_reduction iteration {}: move rejected (candidate cost {:.2})",
						iteration,
						candidate_cost
					);
                    if let Some(writer) = &debug_writer {
                        writer.capture(iteration, "after", solution, context);
                    }
                    break;
                }
            }

            let new_cost = solution.cost_with_context(context);
            if new_cost + f64::EPSILON < previous_cost {
                info!(
					"fleet_and_cost_reduction iteration {}: cost decreased after loop {:.2} -> {:.2}",
					iteration,
					previous_cost,
					new_cost
				);
                if let Some(writer) = &debug_writer {
                    writer.capture(iteration, "after", solution, context);
                }
                previous_cost = new_cost;
                continue;
            }
            info!(
                "fleet_and_cost_reduction iteration {}: no further improvement",
                iteration
            );
            if let Some(writer) = &debug_writer {
                writer.capture(iteration, "after", solution, context);
            }
            break;
        }
    }
}

fn cyclic_overlap_duration(
    start1: f64,
    mut end1: f64,
    start2: f64,
    mut end2: f64,
    period: f64,
) -> f64 {
    fn normalize_interval(start: f64, mut end: f64, period: f64) -> Vec<(f64, f64)> {
        let mut normalized_start = start % period;
        if normalized_start < 0.0 {
            normalized_start += period;
        }
        if end < start {
            end += period;
        }
        let mut remaining = (end - start).max(0.0);
        if remaining <= f64::EPSILON {
            return Vec::new();
        }
        let mut segments = Vec::new();
        let mut current_start = normalized_start;
        while remaining > f64::EPSILON {
            let segment_end = if current_start + remaining > period {
                period
            } else {
                current_start + remaining
            };
            if segment_end > current_start + f64::EPSILON {
                segments.push((current_start, segment_end));
            }
            remaining -= segment_end - current_start;
            current_start = 0.0;
        }
        segments
    }

    fn segment_overlap(start_a: f64, end_a: f64, start_b: f64, end_b: f64) -> f64 {
        let overlap_start = start_a.max(start_b);
        let overlap_end = end_a.min(end_b);
        if overlap_end > overlap_start {
            overlap_end - overlap_start
        } else {
            0.0
        }
    }

    if end1 < start1 {
        end1 += period;
    }
    if end2 < start2 {
        end2 += period;
    }
    let segments1 = normalize_interval(start1, end1, period);
    let segments2 = normalize_interval(start2, end2, period);
    let mut total = 0.0;
    for (s1, e1) in &segments1 {
        for (s2, e2) in &segments2 {
            total += segment_overlap(*s1, *e1, *s2, *e2);
        }
    }
    total
}
