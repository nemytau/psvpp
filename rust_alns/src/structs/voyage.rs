use crate::structs::node::{Base, Installation};
use crate::structs::distance_manager::DistanceManager;
use crate::structs::visit::Visit;
use crate::structs::vessel::Vessel;
use std::rc::Rc;
use crate::structs::constants::{HOURS_IN_PERIOD, DAYS_IN_PERIOD, HOURS_IN_DAY, REL_DEPARTURE_TIME};
use std::clone::Clone;
use std::sync::atomic::{AtomicUsize, Ordering};

static VOYAGE_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Clone)]
pub struct Voyage {
    pub id: usize, // always required
    pub vessel_id: Option<usize>,        
    pub departure_day: Option<usize>,    
    pub visit_ids: Vec<usize>,           
    pub duration: Option<f64>,           
    pub distance: Option<f64>,           
    pub is_feasible: Option<bool>,       
}

impl Voyage {
    // Creates a new voyage object
    pub fn new() -> Self {
        let id = VOYAGE_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            id,
            vessel_id: None,
            departure_day: None,
            visit_ids: Vec::new(),
            duration: None,
            distance: None,
            is_feasible: None,
        }
    }
    // Creates a new voyage object with a specific id
    pub fn new_with_id(id: usize) -> Self {
        Self {
            id,
            vessel_id: None,
            departure_day: None,
            visit_ids: Vec::new(),
            duration: None,
            distance: None,
            is_feasible: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_with_id() {
        let id = 42;
        let voyage = Voyage::new_with_id(id);
        
        assert_eq!(voyage.id, id);
        assert_eq!(voyage.vessel_id, None);
        assert_eq!(voyage.departure_day, None);
        assert!(voyage.visit_ids.is_empty());
        assert_eq!(voyage.duration, None);
        assert_eq!(voyage.distance, None);
        assert_eq!(voyage.is_feasible, None);
    }
}
