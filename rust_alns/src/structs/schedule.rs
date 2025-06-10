use std::rc::Rc;
use crate::structs::node::{Installation, Base, Node, HasLocation};
use crate::structs::voyage::Voyage;
use crate::structs::vessel::Vessel;
use crate::structs::visit::Visit;
use crate::structs::distance_manager::DistanceManager;
use crate::structs::transaction::Transaction;
use std::collections::{BTreeSet, HashMap};
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use crate::structs::problem_data::ProblemData;
use crate::utils::utils::cyclic_intervals_overlap;

use super::vessel;

#[derive(Debug, Clone)]
pub struct Schedule {
    /// voyage_id → departure time
    pub voyage_start_times: HashMap<usize, f64>,
    /// voyage_id → end time
    pub voyage_end_times: HashMap<usize, f64>,
    /// (vessel_id, day) → voyage_id (only one voyage per vessel per day)
    pub vessel_day_voyages: HashMap<(usize, usize), usize>,

    /// installation_id → set of visit_ids
    pub departures_by_installation: HashMap<usize, BTreeSet<usize>>,

    /// Indicates if the schedule needs an update
    need_update: bool, // Made private
}

impl Schedule {
    pub fn empty() -> Self {
        Schedule {
            voyage_start_times: HashMap::new(),
            voyage_end_times: HashMap::new(),
            vessel_day_voyages: HashMap::new(),
            departures_by_installation: HashMap::new(),
            need_update: false,
        }
    }
    pub fn assign_voyage(&mut self, voyage: &Voyage, visits: &[Visit]) {
        self.voyage_start_times.insert(voyage.id, voyage.start_time().unwrap());
        self.voyage_end_times.insert(voyage.id, voyage.end_time().unwrap());
        self.vessel_day_voyages
            .insert((voyage.vessel_id.unwrap(), voyage.departure_day.unwrap()), voyage.id);
        for visit_id in &voyage.visit_ids {
            self.departures_by_installation
                .entry(visits[*visit_id].installation_id())
                .or_insert_with(BTreeSet::new)
                .insert(*visit_id);
        }
    }

    /// Getter for need_update
    pub fn needs_update(&self) -> bool {
        self.need_update
    }

    /// Setter for need_update
    pub fn set_need_update(&mut self, value: bool) {
        self.need_update = value;
    }

    pub fn is_vessel_available_for_period(
        &self, 
        vessel_id: usize, 
        start_time: f64, 
        end_time: f64
    ) -> bool {
        let vessel_voyages = self.get_all_voyages_for_vessel(vessel_id);
        for voyage_id in vessel_voyages {
            if let (Some(start), Some(end)) = (
                self.voyage_start_times.get(&voyage_id),
                self.voyage_end_times.get(&voyage_id),
            ) {
                if cyclic_intervals_overlap(*start, *end, start_time, end_time, HOURS_IN_PERIOD as f64) {
                    return false;
                }
            }
        }
        true
    }
    pub fn is_vessel_available_for_voyage(
        &mut self, 
        vessel_id: usize, 
        voyage: &Voyage
    ) -> bool {
        if let (Some(start), Some(end)) = (voyage.start_time(), voyage.end_time()) {
            self.is_vessel_available_for_period(vessel_id, start, end)
        } else {
            false
        }
    }
    pub fn get_all_voyages_for_vessel(
        &self,
        vessel_id: usize,
    ) -> Vec<usize> {
        let mut voyages = Vec::new();
        for day in 0..DAYS_IN_PERIOD {
            if let Some(&voyage_id) = self.vessel_day_voyages.get(&(vessel_id, day as usize)) {
                voyages.push(voyage_id);
            }
        }
        voyages
    }
    /// Checks for overlaps, skipping empty voyages if a closure is provided.
    pub fn overlaps_with_other_voyages<F>(
        &self,
        vessel_id: usize,
        voyage_id: usize,
        start_time: f64,
        end_time: f64,
        is_empty: Option<F>,
    ) -> bool
    where
        F: Fn(usize) -> bool,
    {
        let voyage_ids = self.get_all_voyages_for_vessel(vessel_id);
        for id in voyage_ids {
            if id != voyage_id {
                if let Some(ref is_empty_fn) = is_empty {
                    if is_empty_fn(id) {
                        continue;
                    }
                }
                if let (Some(start), Some(end)) = (
                    self.voyage_start_times.get(&id),
                    self.voyage_end_times.get(&id),
                ) {
                    if cyclic_intervals_overlap(
                        *start,
                        *end,
                        start_time,
                        end_time,
                        HOURS_IN_PERIOD as f64,
                    ) {
                        return true;
                    }
                }
            }
        }
        false
    }
}