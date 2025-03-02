use crate::structs::node::{Base, Installation};
use crate::structs::distance_manager::DistanceManager;
use crate::structs::visit::Visit;
use crate::structs::vessel::Vessel;
use std::rc::Rc;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use std::clone::Clone;

#[derive(Debug, Clone)]
pub struct Voyage {
    vessel: Option<Rc<Vessel>>,            // Vessel assigned to this voyage, if any
    start_time: Option<u32>,               // Start time in hours, if assigned
    start_day: Option<u32>,                // Start day, if assigned
    deck_load: u32,                        // Current deck load, should not exceed vessel's capacity
    end_time: Option<u32>,                 // End time in hours, if start_time and vessel are assigned
    distance_manager: DistanceManager,     // Distance manager instance
    base: Base,                            // Regular Base instance
    route: Vec<Visit>,                     // Sequence of owned visits
    is_scheduled: bool,                    // Indicates if the voyage is complete and scheduled
}

impl Voyage {
    /// Creates a new `Voyage` instance with references to the base and distance manager.
    pub fn new(distance_manager: DistanceManager, base: Base) -> Self {
        Voyage {
            vessel: None,
            start_time: None,
            start_day: None,
            deck_load: 0,
            end_time: None,
            distance_manager,
            base,
            route: vec![],
            is_scheduled: false,
        }
    }

    /// Assigns a vessel to the voyage and updates the scheduled status.
    pub fn set_vessel(&mut self, vessel: Rc<Vessel>) {
        self.vessel = Some(vessel);
        self.update_scheduled_status();
    }

    /// Sets the start day for the voyage.
    pub fn set_start_day(&mut self, start_day: u32) {
        self.start_day = Some(start_day);
        self.update_start_time();
    }

    /// Calculates and updates the start time based on `start_day` and `REL_DEPARTURE_TIME`.
    fn update_start_time(&mut self) {
        if let Some(day) = self.start_day {
            self.start_time = Some(day * HOURS_IN_DAY + REL_DEPARTURE_TIME);
            self.update_scheduled_status();
        }
    }

    /// Marks the voyage as scheduled if it has a vessel, start time, and start day.
    fn update_scheduled_status(&mut self) {
        self.is_scheduled = self.vessel.is_some() && self.start_time.is_some() && self.start_day.is_some() && !self.route.is_empty();
    }

    /// Adds a visit to the route without optimizing sequence or checking load.
    pub fn add_visit(&mut self, visit: Visit) {
        self.route.push(visit);
    }

    /// Returns the current deck load.
    pub fn get_deck_load(&self) -> u32 {
        self.deck_load
    }

    /// Returns the scheduled status.
    pub fn is_scheduled(&self) -> bool {
        self.is_scheduled
    }

    /// Returns the start time, if assigned.
    pub fn get_start_time(&self) -> Option<u32> {
        self.start_time
    }

    /// Returns the end time, if calculated.
    pub fn get_end_time(&self) -> Option<u32> {
        self.end_time
    }

    /// Returns the current route of visits.
    pub fn get_route(&self) -> &Vec<Visit> {
        &self.route
    }

    pub fn get_vessel_idx(&self) -> Option<u32> {
        self.vessel.as_ref().map(|v| v.get_idx())
    }

    /// Helper function to check if two time intervals overlap.
    fn times_overlap(start1: u32, end1: u32, start2: u32, end2: u32) -> bool {
        start1 < end2 && end1 > start2
    }

    pub fn overlaps_with(&self, other: &Voyage) -> bool {
        // Check if the voyages have overlapping each other
        if let (Some(start_time), Some(end_time), Some(other_start_time), Some(other_end_time)) = (self.start_time, self.end_time, other.start_time, other.end_time) {
            // Original times
            if Voyage::times_overlap(start_time, end_time, other_start_time, other_end_time) {
                return true;
            }
            // Adjusted by horizon length
            if Voyage::times_overlap(start_time - HOURS_IN_PERIOD, end_time - HOURS_IN_PERIOD, other_start_time, other_end_time) {
                return true;
            }
            if Voyage::times_overlap(start_time, end_time, other_start_time - HOURS_IN_PERIOD, other_end_time - HOURS_IN_PERIOD) {
                return true;
            }
        }
        false
    }

    pub fn optimize_route(&mut self) {
        // LATER: Implement optimization algorithm
    }
}