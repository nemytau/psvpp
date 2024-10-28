use crate::structs::visit::Visit;
use crate::structs::voyage::Voyage;
use crate::structs::vessel::Vessel;
use crate::structs::schedule::Schedule;

/// Enum to represent different types of transactions within the scheduling system.
#[derive(Debug, Clone)]
pub enum Transaction {
    /// Removes a visit from a voyage.
    RemoveVisitFromVoyage {
        visit_id: u32, 
        voyage_id: u32,
    },
    /// Adds a visit to a voyage.
    AddVisitToVoyage {
        visit_id: u32, 
        voyage_id: u32,
    },
    /// Changes the route sequence within a voyage.
    ChangeRouteSequence {
        voyage_id: u32,
        old_sequence: Vec<u32>,  // Sequence of visit IDs
        new_sequence: Vec<u32>,
    },
    /// Assigns a new vessel to a voyage.
    AssignVesselToVoyage {
        voyage_id: u32,
        old_vessel_id: Option<u32>,
        new_vessel_id: Option<u32>,
    },
}

impl Transaction {
    /// Revert the transaction.
    pub fn revert(self, schedule: &mut Schedule) {
        match self {
            Transaction::RemoveVisitFromVoyage { visit_id, voyage_id } => {
                if let Some(visit) = schedule.get_visit_by_id(visit_id) {
                    schedule.add_visit_to_voyage(voyage_id, visit);
                }
            },
            Transaction::AddVisitToVoyage { visit_id, voyage_id } => {
                schedule.remove_visit_from_voyage(voyage_id, visit_id);
            },
            Transaction::ChangeRouteSequence { voyage_id, old_sequence, .. } => {
                if let Some(voyage) = schedule.get_voyage_mut(voyage_id) {
                    voyage.set_route_from_ids(&old_sequence, schedule);
                }
            },
            Transaction::AssignVesselToVoyage { voyage_id, old_vessel_id, .. } => {
                if let Some(voyage) = schedule.get_voyage_mut(voyage_id) {
                    if let Some(old_vessel) = old_vessel_id.and_then(|id| schedule.get_vessel_by_id(id)) {
                        voyage.set_vessel(Some(old_vessel));
                    } else {
                        voyage.set_vessel(None);
                    }
                }
            },
        }
    }
}