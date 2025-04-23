use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Vessel {
    pub id: usize,                 
    pub name: String,
    pub deck_capacity: f64,       // Changed from u32 to f64
    pub bulk_capacity: f64,       // Changed from u32 to f64
    pub speed: f64,
    pub vessel_type: String,
    pub fcs: f64,                 // Fuel consumption sailing
    pub fcw: f64,                 // Fuel consumption waiting
    pub cost: f64,                // Changed from u32 to f64
}

impl Vessel {
    pub fn new(
        id: usize,
        name: String,
        deck_capacity: f64,       // Changed from u32 to f64
        bulk_capacity: f64,       // Changed from u32 to f64
        speed: f64,
        vessel_type: String,
        fcs: f64,
        fcw: f64,
        cost: f64,                // Changed from u32 to f64
    ) -> Self {
        Self {
            id,
            name,
            deck_capacity,
            bulk_capacity,
            speed,
            vessel_type,
            fcs,
            fcw,
            cost,
        }
    }

    pub fn id(&self) -> usize {
        self.id
    }

    pub fn builder() -> VesselBuilder {
        VesselBuilder::default()
    }
}

#[derive(Default)]
pub struct VesselBuilder {
    id: Option<usize>,
    name: Option<String>,
    deck_capacity: Option<f64>,
    bulk_capacity: Option<f64>,
    speed: Option<f64>,
    vessel_type: Option<String>,
    fcs: Option<f64>,
    fcw: Option<f64>,
    cost: Option<f64>,
}

impl VesselBuilder {
    pub fn id(mut self, id: usize) -> Self {
        self.id = Some(id);
        self
    }

    pub fn name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }

    pub fn deck_capacity(mut self, deck_capacity: f64) -> Self {
        self.deck_capacity = Some(deck_capacity);
        self
    }

    pub fn bulk_capacity(mut self, bulk_capacity: f64) -> Self {
        self.bulk_capacity = Some(bulk_capacity);
        self
    }

    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    pub fn vessel_type(mut self, vessel_type: String) -> Self {
        self.vessel_type = Some(vessel_type);
        self
    }

    pub fn fuel_consumption_sailing(mut self, fcs: f64) -> Self {
        self.fcs = Some(fcs);
        self
    }

    pub fn fuel_consumption_waiting(mut self, fcw: f64) -> Self {
        self.fcw = Some(fcw);
        self
    }

    pub fn cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    pub fn build(self) -> Result<Vessel, &'static str> {
        Ok(Vessel {
            id: self.id.ok_or("id is required")?,
            name: self.name.ok_or("name is required")?,
            deck_capacity: self.deck_capacity.ok_or("deck_capacity is required")?,
            bulk_capacity: self.bulk_capacity.ok_or("bulk_capacity is required")?,
            speed: self.speed.ok_or("speed is required")?,
            vessel_type: self.vessel_type.ok_or("vessel_type is required")?,
            fcs: self.fcs.ok_or("fuel consumption sailing (fcs) is required")?,
            fcw: self.fcw.ok_or("fuel consumption waiting (fcw) is required")?,
            cost: self.cost.ok_or("cost is required")?,
        })
    }
}