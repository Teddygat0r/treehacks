# Quick Reference Guide

## 🚀 Running the System

### 1. Start Target Server
```bash
./start_target_server.sh                          # Default: deterministic
./start_target_server.sh --strategy probabilistic # SLED paper algorithm
./start_target_server.sh --verbose                # See token comparisons
```

### 2. Test/Benchmark
```bash
# Quick test (3 prompts, ~30s)
python quick_test.py

# GSM8K benchmark (10 questions, ~2-3 min)
python benchmark_gsm8k.py

# Full benchmark (50 questions, ~10-15 min)
python benchmark_gsm8k.py --num-samples 50 --use-hf
```

## 📊 Available Tools

| Tool | Purpose | Runtime | Output |
|------|---------|---------|--------|
| `quick_test.py` | Quick acceptance check | ~30s | Console stats |
| `benchmark_gsm8k.py` | Math benchmark | ~2-15min | Console + JSON |
| `test_strategies.py` | Compare strategies | Manual | Console |

## 🎯 Verification Strategies

| Strategy | Description | Best For | Acceptance |
|----------|-------------|----------|------------|
| `deterministic` | Exact match only | Baseline testing | 85-95% |
| `probabilistic` | SLED paper (α=min(1, p_t/p_d)) | Research/paper replication | 75-90% |
| `threshold` | Accept if p_target > threshold | Tuning tradeoffs | 80-95% |
| `greedy` | Always use target | Debugging | Varies |

## 📈 Performance Targets

### OPT-350m → OPT-1.3b (Recommended)
- **Acceptance:** 85-95%
- **Speed:** 60-70 tokens/sec
- **Speedup:** 1.4-1.5x vs target alone

### OPT-125m → OPT-1.3b (Not Recommended)
- **Acceptance:** 40-50% ❌
- **Speed:** 35-45 tokens/sec
- **Speedup:** Minimal

## 🔧 Common Commands

### Basic Testing
```bash
# Test current setup
python quick_test.py

# Benchmark 10 questions
python benchmark_gsm8k.py
```

### Change Draft Model
```bash
# Edit draft_node/client.py, line 22:
draft_model="facebook/opt-350m"  # Change this
```

### Change Verification Strategy
```bash
# Restart server with different strategy
./start_target_server.sh --strategy probabilistic
python quick_test.py
```

### Debug Low Acceptance
```bash
# See token-by-token comparisons
./start_target_server.sh --strategy deterministic --verbose
python quick_test.py
```

## 📖 Documentation

- **README.md** - System overview and setup
- **TESTING_GUIDE.md** - Strategy testing details
- **BENCHMARK_USAGE.md** - GSM8K benchmark guide
- **QUICK_REFERENCE.md** - This file

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Low acceptance (<50%) | Use opt-350m instead of opt-125m |
| Out of memory | Reduce gpu_memory_utilization in server.py |
| Server won't start | Check port 50051 not in use |
| Connection refused | Start server first: `./start_target_server.sh` |

## 💡 Tips

1. **Always use greedy (temp=0)** for best acceptance
2. **Start with quick_test.py** before long benchmarks
3. **Use opt-350m** as draft model (good balance)
4. **Save benchmark results** for comparison
5. **Check verbose logs** if acceptance is low
