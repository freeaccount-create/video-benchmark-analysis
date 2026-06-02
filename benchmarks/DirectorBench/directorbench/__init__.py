"""
DirectorBench: Multi-Agent Evaluation Framework for Minute-Long Video Generation

A benchmark for evaluating AI-generated minute-long videos across five
evaluation dimensions, powered by a hierarchical multi-agent architecture:

    Phase 0: Orchestrator — preprocessing (shot detection, ASR, audio separation)
    Phase 1: Specialist Agents (Script / Video / Audio / Stability) — parallel
    Phase 2: Cross-Modal Alignment Agent — depends on Phase 1
    Phase 3: Diagnosis Synthesizer — aggregation, bottleneck detection, report
"""

__version__ = "0.1.0"
