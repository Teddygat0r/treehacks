# Verification Strategy Testing Guide

## Quick Start

### 1. Test Current Strategy (Fastest)

```bash
# Terminal 1: Start server with a strategy
cd /home/dgorb/Github/treehacks/treehacks/workers
source /home/dgorb/Github/treehacks/.venv/bin/activate
./start_target_server.sh --strategy deterministic

# Terminal 2: Run quick test
cd /home/dgorb/Github/treehacks/treehacks/workers
source /home/dgorb/Github/treehacks/.venv/bin/activate
python quick_test.py
```

The quick test will show you:
- ✅ Average acceptance rate
- 📈 Total tokens accepted/generated
- ⚡ Speed (tokens/sec)
- 💡 Interpretation of results

### 2. Compare Different Strategies

Test each strategy by restarting the server:

```bash
# Test deterministic
./start_target_server.sh --strategy deterministic
python quick_test.py
# Note the acceptance rate, then Ctrl+C the server

# Test probabilistic
./start_target_server.sh --strategy probabilistic
python quick_test.py
# Note the acceptance rate, then Ctrl+C the server

# Test threshold with different values
./start_target_server.sh --strategy threshold --threshold 0.1
python quick_test.py

./start_target_server.sh --strategy threshold --threshold 0.05
python quick_test.py
```

### 3. Debug with Verbose Logging

To see exactly what's happening token-by-token:

```bash
./start_target_server.sh --strategy deterministic --verbose
python quick_test.py
```

This shows:
- Each draft token vs target token
- Match/mismatch decisions
- Probability values (for probabilistic/threshold)

## Available Strategies

### `deterministic` (Default)
**What it does:** Accept only if draft token exactly matches target token

**When to use:**
- Good baseline to start with
- Most predictable behavior
- Best for debugging

**Expected acceptance:** 70-90% for same-family models (both OPT)

**Command:**
```bash
./start_target_server.sh --strategy deterministic
```

---

### `probabilistic` (SLED Paper)
**What it does:** Implements α = min(1, p_target/p_draft) acceptance probability

**When to use:**
- Following the research paper exactly
- Want theoretically correct output distribution
- Willing to accept some randomness

**Expected acceptance:** 60-80% for same-family models

**Command:**
```bash
./start_target_server.sh --strategy probabilistic
```

---

### `threshold`
**What it does:** Accept draft token if p_target(draft_token) > threshold

**When to use:**
- Want more lenient acceptance than deterministic
- Don't need probabilistic guarantees
- Can tune threshold to balance acceptance vs correctness

**Expected acceptance:** 75-95% for same-family models (depends on threshold)

**Commands:**
```bash
# Conservative (low threshold = more lenient)
./start_target_server.sh --strategy threshold --threshold 0.05

# Default
./start_target_server.sh --strategy threshold --threshold 0.1

# Strict (high threshold = less lenient)
./start_target_server.sh --strategy threshold --threshold 0.3
```

---

### `greedy`
**What it does:** Always use target model's token (disables speculation)

**When to use:**
- Debugging only
- Compare against baseline without speculation
- Verify target model is working

**Expected acceptance:** Varies (depends on how often models agree)

**Command:**
```bash
./start_target_server.sh --strategy greedy
```

## Troubleshooting Low Acceptance Rates

If you're getting <50% acceptance with same-family models (both OPT), try these steps:

### 1. Check Model Compatibility
```bash
# Both should be OPT models
# Draft: facebook/opt-125m
# Target: facebook/opt-1.3b
```

### 2. Enable Verbose Logging
```bash
./start_target_server.sh --strategy deterministic --verbose
python quick_test.py
```

Look for:
- Are tokens frequently mismatching even on simple prompts?
- Are the models generating completely different outputs?
- Do you see any errors in the logs?

### 3. Check for Randomness Issues
Both models use `seed=42` but may still diverge due to:
- Different sampling implementations
- Different tokenizers (check they're the same)
- Floating point precision differences

### 4. Try Different Strategies
```bash
# Try threshold with very low value (most lenient)
./start_target_server.sh --strategy threshold --threshold 0.01
python quick_test.py
```

If this gives much higher acceptance, the issue is that:
- Models agree on high-probability tokens
- Deterministic is too strict for your use case

### 5. Verify Both Models Work Independently
```bash
# Test just the draft model
cd draft_node
python client.py

# Test just the target model (check server logs)
```

## Understanding the Output

### Quick Test Output Example
```
📊 RESULTS SUMMARY
==================
🎯 Average Acceptance Rate: 73.2%
📈 Total Accepted/Generated: 44/60
⚡ Average Speed: 18.3 tokens/sec

💡 INTERPRETATION
=================
✓ GOOD: 60-80% acceptance - decent speedup expected
```

### What Each Metric Means

- **Acceptance Rate**: % of draft tokens accepted by target model
  - Higher = better speculation efficiency
  - Target: >70% for same-family models

- **Accepted/Generated**: Number of draft tokens kept vs total drafted
  - Shows absolute numbers behind the percentage

- **Speed**: Tokens per second overall
  - Includes both draft generation and verification time
  - Speculative decoding aims for 1.5-3x speedup

## Common Issues

### "Connection refused" error
**Problem:** Target server not running

**Solution:**
```bash
./start_target_server.sh --strategy deterministic
# Wait for "Verification service ready!" message
```

### Very low acceptance (<30%)
**Problem:** Models are incompatible or diverging

**Solutions:**
1. Check both models are from same family (both OPT)
2. Try threshold strategy with low threshold
3. Enable verbose logging to see what's happening

### Server crashes with OOM
**Problem:** Models too large for GPU

**Solutions:**
1. Reduce `gpu_memory_utilization` in server.py
2. Use smaller model (opt-1.3b → opt-350m)
3. Run on separate GPUs

## Files

- `verification_strategies.py` - Strategy implementations
- `server.py` - Target node with strategy support
- `quick_test.py` - Simple test script (recommended)
- `test_strategies.py` - Advanced comparison script
- `start_target_server.sh` - Server launcher with CLI args

## Next Steps After Testing

1. **Found a good strategy?**
   - Note the strategy name and parameters
   - Use it as default in your server configuration

2. **Still getting low acceptance?**
   - May indicate a bug in the verification logic
   - Check that logprobs are being extracted correctly
   - Verify token IDs match between draft and target

3. **Ready for production?**
   - Remove `--verbose` flag for better performance
   - Consider implementing batch verification
   - Add metrics/monitoring for acceptance rate tracking
