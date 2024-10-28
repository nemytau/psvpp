#[derive(Debug)]
pub struct Schedule {
    voyages: Vec<Voyage>,                  // A list of voyages with stable indexing
    unassigned_visits: Vec<Visit>,         // List of unassigned visit IDs
    vessels: Vec<Rc<Vessel>>,              // List of vessels
    installations: Vec<Installation>,      // List of installations
    base: Base,                            // Base installation
    distance_manager: Rc<DistanceManager>, // Distance manager
    transaction_stack: Vec<Transaction>,   // Stack of reversible transactions
}

impl Schedule {
    /// Creates a new Schedule instance.
    pub fn new(
        vessels: Vec<Rc<Vessel>>, 
        installations: Vec<Installation>, 
        base: Base,
    ) -> Self {
        let distance_manager = Rc::new(DistanceManager::new(&installations, &base));
        let unassigned_visits = Schedule::generate_visits(&installations);
        
        Schedule {
            voyages: Vec::new(),
            unassigned_visits: unassigned_visits,
            vessels,
            installations,
            base,
            distance_manager,
            transaction_stack: Vec::new(),
        }
    }

    /// Assigns a visit to a specific voyage in the list by index and records the transaction.
    pub fn assign_visit_to_voyage(&mut self, visit_id: usize, voyage_index: usize) -> Result<(), String> {
        if let Some(voyage) = self.voyages.get_mut(voyage_index) {
            if let Some(visit) = self.visits.remove(&visit_id) {
                voyage.add_visit(visit_id);
                self.unassigned_visits.retain(|&id| id != visit_id);
                
                // Record transaction
                self.transaction_stack.push(Transaction::AssignVisit { visit_id, voyage_index });
                Ok(())
            } else {
                Err("Visit not found".to_string())
            }
        } else {
            Err("Voyage not found".to_string())
        }
    }

    /// Reverts the last transaction.
    pub fn revert_last_transaction(&mut self) {
        if let Some(transaction) = self.transaction_stack.pop() {
            transaction.revert(self);
        }
    }
}