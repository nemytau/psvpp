pub fn intervals_overlap(start1: f64, end1: f64, start2: f64, end2: f64) -> bool {
    start1 < end2 && start2 < end1
}

pub fn cyclic_intervals_overlap(
    start1: f64,
    end1: f64,
    start2: f64,
    end2: f64,
    period: f64,
) -> bool {
    let intervals1 = if end1 > period {
        vec![(start1, period), (0.0, end1 - period)]
    } else {
        vec![(start1, end1)]
    };

    let intervals2 = if end2 > period {
        vec![(start2, period), (0.0, end2 - period)]
    } else {
        vec![(start2, end2)]
    };

    for (s1, e1) in &intervals1 {
        for (s2, e2) in &intervals2 {
            if s1 < e2 && s2 < e1 {
                return true;
            }
        }
    }
    false
}

//Tests:
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_intervals_overlap() {
        assert!(intervals_overlap(1.0, 3.0, 2.0, 4.0));
        assert!(!intervals_overlap(1.0, 2.0, 3.0, 4.0));
        assert!(intervals_overlap(1.0, 5.0, 2.0, 3.0));
        assert!(!intervals_overlap(1.0, 2.0, 2.0, 3.0));
    }

    #[test]
    fn test_cyclic_intervals_overlap() {
        let period = 168.0;
        assert!(cyclic_intervals_overlap(160.0, 208.0, 16.0, 64.0, period)); // overlaps due to wrap
        assert!(cyclic_intervals_overlap(160.0, 208.0, 200.0, 220.0, period)); // overlaps after wrap
        assert!(!cyclic_intervals_overlap(
            100.0, 120.0, 130.0, 150.0, period
        )); // no overlap
    }
}
