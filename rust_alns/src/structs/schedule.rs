use std::rc::Rc;
use crate::structs::node::{Installation, Base, Node, HasLocation};
use crate::structs::voyage::Voyage;
use crate::structs::vessel::Vessel;
use crate::structs::visit::Visit;
use crate::structs::distance_manager::DistanceManager;
use crate::structs::transaction::Transaction;
use std::collections::HashMap;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use rand::seq::SliceRandom;


#[derive(Debug)]
pub struct Schedule {
    voyages: Vec<Voyage>,                  // A list of voyages with stable indexing
    unassigned_visits: Vec<Visit>,         // List of unassigned visits
    vessels: Vec<Rc<Vessel>>,              // List of vessels
    installations: Vec<Installation>,  // List of installations (now Rc to match Visit needs)
    base: Base,                            // Base installation
    distance_manager: DistanceManager, // Distance manager
    transaction_stack: Vec<Transaction>,   // Stack of reversible transactions
    }


impl Schedule {
    /// Creates a new Schedule instance.
    pub fn new(
        vessels: Vec<Rc<Vessel>>, 
        installations: Vec<Installation>, 
        base: Base,
    ) -> Self {
        let distance_manager = DistanceManager::new(installations.len()+1);
        
        Schedule {
            voyages: Vec::new(),
            unassigned_visits: Vec::new(),
            vessels,
            installations,
            base,
            distance_manager,
            transaction_stack: Vec::new(),
        }
    }

    // TODO: Implement the following method
    /// Initializes the schedule with a calculated distance matrix, generated visits, and initial voyage assignments.
    pub fn init_schedule(&mut self) {
        // Step 2: Calculate distances between all installations and base
        let installations: Vec<&dyn HasLocation> = self.installations.iter().map(|i| i as &dyn HasLocation).collect();
        self.distance_manager.calculate_distances(&self.base, &installations);
        
        // Step 2: Generate voyages for each day and assign them to vessels
        self.generate_initial_schedule();
    }
    
    /// Reverts the last transaction.
    pub fn revert_last_transaction(&mut self) {
        if let Some(transaction) = self.transaction_stack.pop() {
            transaction.revert(self);
        }
    }

    /// Generates an initial schedule by creating voyages and assigning them to vessels.
    fn generate_initial_schedule(&mut self) {
        let mut visit_schedule: HashMap<u32, Vec<Visit>> = HashMap::new();
        // Collects visits for each day based on installation visit scenarios
        for installation in &self.installations {
            let departure_scenario = installation.generate_visit_scenario();

            for (day, &is_visit_day) in departure_scenario.iter().enumerate() {
                if is_visit_day == 1 {
                    let visit = Visit::new(installation.clone());
                    
                    // Insert or update the visits for each day in the scenario
                    visit_schedule.entry(day as u32).or_insert_with(Vec::new).push(visit);
                }
            }
        }
        // For each day, create voyages and randomly distribute visits to them.
        // Number of voyages created is rounded up number of visits divided by max installations per voyage
        
        for day in 0..DAYS_IN_PERIOD {
            let mut day_visits = visit_schedule.get(&day).unwrap_or(&Vec::new()).clone();
            let num_of_voyages = (day_visits.len() as f64 / HOURS_IN_PERIOD as f64).ceil() as i32;
            let mut day_voyages = self.create_empty_voyages(num_of_voyages, day);
            // Randomly assign visits to voyages
            while let Some(visit) = day_visits.pop() {
                if let Some(voyage) = day_voyages.choose_mut(&mut rand::thread_rng()) {
                    voyage.add_visit(visit);  // Now using owned `visit`
                }
            }
            self.voyages.extend(day_voyages);
        }

        // Assign vessels to voyages based on availability and capacity.
        // Then optimize the route sequence for each voyage.
        /*
        let available_vessels: Vec<_> = self.vessels.iter().cloned().collect();
        for voyage in &mut self.voyages {
            if let Some(vessel) = self.find_available_vessel(&available_vessels, voyage) {
            voyage.set_vessel(vessel);
            }
            voyage.optimize_route();
        }
        */
    }

    fn create_empty_voyages(&self, num_of_voyages: i32, start_day: u32) -> Vec<Voyage> {
        let mut voyages: Vec<Voyage> = Vec::new();
        for _ in 0..num_of_voyages {
            let mut voyage = Voyage::new(self.distance_manager.clone(), self.base.clone());
            voyage.set_start_day(start_day);
            voyages.push(voyage);
        }
        voyages
    }

    /// Find an available vessel for a given day that can accommodate the voyage
    fn find_available_vessel(&self, vessels: &Vec<Rc<Vessel>>, voyage: &Voyage) -> Option<Rc<Vessel>> {
        // LATER: Sort vessels by capacity
        for vessel in vessels {
            if self.is_vessel_available(vessel, voyage) {
                return Some(vessel.clone());
            }
        }
        None
    }

    /// Checks if a vessel is available for a specific day and voyage
    fn is_vessel_available(&self, vessel: &Vessel, voyage: &Voyage) -> bool {
        // Find all voyages for the vessel by checking vessel's idx
        let vessel_voyages: Vec<&Voyage> = self.voyages.iter().filter(|v| v.get_vessel_idx() == Some(vessel.get_idx())).collect();
        // For each voyage check overlap with the new voyage
        for vessel_voyage in vessel_voyages {
            if voyage.overlaps_with(vessel_voyage) {
                return false;
            }
        }
        true
    }
}