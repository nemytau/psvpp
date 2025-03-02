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
                // Placeholder for removing a visit from a voyage
            },
            Transaction::AddVisitToVoyage { visit_id, voyage_id } => {
                // Placeholder for adding a visit to a voyage
            },
            Transaction::ChangeRouteSequence { voyage_id, old_sequence, .. } => {
                // Placeholder for changing the route sequence within a voyage
            },
            Transaction::AssignVesselToVoyage { voyage_id, old_vessel_id, .. } => {
                // Placeholder for assigning a new vessel to a voyage
            },
        }
    }
}
