use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HealthState {
    Healthy,
    Degraded { permanent: bool },
}

/// Small, recoverable record of database and executor availability.
pub(crate) struct ServiceHealth(Mutex<HealthState>);

impl Default for ServiceHealth {
    fn default() -> Self {
        Self(Mutex::new(HealthState::Healthy))
    }
}

impl ServiceHealth {
    pub(crate) fn is_degraded(&self) -> bool {
        self.0
            .lock()
            .map(|state| !matches!(*state, HealthState::Healthy))
            .unwrap_or(true)
    }

    /// Records a runtime failure. A later successful probe can clear it.
    pub(crate) fn degrade(&self) {
        if let Ok(mut state) = self.0.lock() {
            *state = HealthState::Degraded { permanent: false };
        }
    }

    /// Records a startup/schema failure that cannot be recovered by probing.
    pub(crate) fn degrade_permanently(&self) {
        if let Ok(mut state) = self.0.lock() {
            *state = HealthState::Degraded { permanent: true };
        }
    }

    /// Clears a transient failure after a successful durable operation.
    pub(crate) fn recover(&self) {
        if let Ok(mut state) = self.0.lock()
            && matches!(*state, HealthState::Degraded { permanent: false })
        {
            *state = HealthState::Healthy;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ServiceHealth;

    #[test]
    fn transient_failure_recovers_after_probe() {
        let health = ServiceHealth::default();
        health.degrade();
        assert!(health.is_degraded());
        health.recover();
        assert!(!health.is_degraded());
    }

    #[test]
    fn permanent_failure_does_not_recover_after_probe() {
        let health = ServiceHealth::default();
        health.degrade_permanently();
        health.recover();
        assert!(health.is_degraded());
    }
}
