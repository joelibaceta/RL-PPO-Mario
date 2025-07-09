# Testing Guide

This repository includes comprehensive tests to validate that the PPO training is working correctly and that the model shows improvement over time.

## Test Requirements

The tests validate that:
- ✅ The model trains without errors
- ✅ Mean episode length increases through training episodes
- ✅ Mean reward increases through training episodes  
- ✅ Model parameters change during training (learning occurs)

## Running Tests

### Quick Demo
Run the training validation demo:
```bash
python test_demo.py
```

### Full Test Suite
Run all tests using pytest:
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_training.py::TestMarioTraining::test_training_progression_with_multiple_phases -v

# Run with coverage
pytest tests/ --cov=agent --cov-report=html
```

### Individual Tests

1. **Environment Creation**: `pytest tests/test_training.py::TestMarioTraining::test_environment_creation`
2. **Training Progress**: `pytest tests/test_training.py::TestMarioTraining::test_model_training_and_metrics_improvement`
3. **Multi-Phase Training**: `pytest tests/test_training.py::TestMarioTraining::test_training_progression_with_multiple_phases`

## Test Implementation

The tests use a mock Mario environment that simulates the characteristics of the real game environment but doesn't require the actual Mario ROM. This allows:

- **Fast execution**: Tests complete in minutes rather than hours
- **Predictable improvement**: Mock environment is designed to show clear learning progression
- **CI compatibility**: No external dependencies or ROM files needed

## Expected Results

When tests pass, you should see output similar to:
```
Training phase 1/3
Phase 1: Mean reward = 98.22, Mean episode length = 19.82, Episodes = 51
Training phase 2/3  
Phase 2: Mean reward = 376.39, Mean episode length = 24.73, Episodes = 40
Training phase 3/3
Phase 3: Mean reward = 888.10, Mean episode length = 44.68, Episodes = 22
```

This demonstrates clear improvement in both reward (98.22 → 888.10) and episode length (19.82 → 44.68) across training phases.

## Integration with Real Training

While the tests use a mock environment, the same principles apply to real Mario training:
- Monitor episode rewards and lengths during training
- Expect gradual improvement over thousands/millions of timesteps
- Use TensorBoard logs to track long-term progress

The mock environment tests validate that the training infrastructure works correctly before running expensive real training sessions.