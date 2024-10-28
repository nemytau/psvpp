use crate::structs::node::{Base, Installation};
use crate::structs::distance_manager::DistanceManager;
use crate::structs::visit::Visit;
use crate::structs::vessel::Vessel;
use std::rc::Rc;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};

#[derive(Debug, Clone)]
pub struct Voyage {
    vessel: Option<Rc<Vessel>>,            // Vessel assigned to this voyage, if any
    start_time: Option<u32>,               // Start time in hours, if assigned
    start_day: Option<u32>,                // Start day, if assigned
    deck_load: u32,                        // Current deck load, should not exceed vessel's capacity
    end_time: Option<u32>,                 // End time in hours, if start_time and vessel are assigned
    distance_manager: Rc<DistanceManager>, // Shared pointer to distance manager
    base: Rc<Base>,                        // Shared pointer to the base
    route: Vec<Visit>,                     // Sequence of owned visits
    is_scheduled: bool,                    // Indicates if the voyage is complete and scheduled
}

impl Voyage {
    /// Creates a new `Voyage` instance with references to the base and distance manager.
    pub fn new(distance_manager: Rc<DistanceManager>, base: Rc<Base>) -> Self {
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
}