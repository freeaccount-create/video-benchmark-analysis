"""
example_usage.py — Quick-start examples for DirectorBench.

Run with:
    export OPENAI_API_KEY="sk-..."
    # Or for Azure OpenAI:
    export AZURE_OPENAI_API_KEY="..."
    export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
    python example_usage.py
"""

from directorbench.config import EvalConfig
from directorbench.main import evaluate_video


def example_basic():
    """Minimal usage: evaluate a video with just a path and prompt."""
    report = evaluate_video(
        video_path="./sample_video.mp4",
        user_prompt="Create a 1-minute dramatic chase scene through a neon-lit cyberpunk city at night",
    )
    print(f"Grade: {report.overall_grade} | Score: {report.overall_score:.2f}")


def example_full():
    """Full usage: with script, storyboard, and custom user profile."""
    report = evaluate_video(
        video_path="./sample_video.mp4",
        user_prompt="Create a 1-minute dramatic chase scene through a neon-lit cyberpunk city at night",
        script_text="""
Scene 1 (0:00-0:15): Wide establishing shot of a rain-soaked cyberpunk city.
  Neon signs reflect off wet streets. A DETECTIVE (30s, trench coat) walks briskly.

Scene 2 (0:15-0:30): Close-up of the Detective's face as she spots a SUSPECT
  ducking into an alley. She breaks into a run.

Scene 3 (0:30-0:45): Fast-paced tracking shot through narrow alleys.
  The Suspect knocks over trash cans. The Detective leaps over them.

Scene 4 (0:45-1:00): The chase ends on a rooftop. The Suspect is cornered.
  Wide shot revealing the glowing city skyline behind them.
  The Detective draws her badge. Tension builds. Fade to black.
        """,
        storyboard=[
            {"shot": 1, "type": "wide", "description": "Establishing city shot", "duration": "15s"},
            {"shot": 2, "type": "close-up", "description": "Detective spots suspect", "duration": "15s"},
            {"shot": 3, "type": "tracking", "description": "Alley chase sequence", "duration": "15s"},
            {"shot": 4, "type": "wide", "description": "Rooftop confrontation", "duration": "15s"},
        ],
        user_profile={
            "weight_video": 2.0,        # prioritize visual quality
            "weight_audio": 1.5,        # audio is important for this scene
            "weight_crossmodal": 1.5,   # cross-modal consistency matters
            "weight_script": 0.8,       # script quality less critical
            "custom_requirements": [
                "Neon lighting must be visible in all outdoor shots",
                "Rain effects should be consistent",
                "Character should maintain consistent appearance across shots",
            ],
        },
        output_dir="./eval_outputs/chase_scene",
    )

    # Access detailed results
    print(f"\n{'='*50}")
    print(f"Grade: {report.overall_grade} | Score: {report.overall_score:.2f}")
    print(f"{'='*50}")

    print("\nDimension scores:")
    for dim, score in report.dimension_scores.items():
        print(f"  {dim}: {score:.2f}")

    print(f"\nBottlenecks: {len(report.bottlenecks)}")
    for b in report.bottlenecks:
        print(f"  ⚠ {b.metric_name}: {b.score:.2f}")
        for s in b.suggestions[:2]:
            print(f"    → {s}")

    print(f"\nNeeds human review: {len(report.needs_human_review)} items")

    # The full report is also saved to disk
    print(f"\nFull report saved to: ./eval_outputs/chase_scene/")


def example_custom_config():
    """Custom configuration: different model, thresholds, etc."""
    config = EvalConfig()
    config.llm.model = "gpt-4o-mini"          # cheaper model
    config.llm.temperature = 0.0               # deterministic
    config.confidence_threshold = 0.7          # stricter human review threshold
    config.bottleneck_threshold = 0.6          # stricter bottleneck detection
    config.preprocess.shot_detection_threshold = 30.0  # less sensitive shot detection

    report = evaluate_video(
        video_path="./sample_video.mp4",
        user_prompt="A peaceful nature documentary about forest wildlife",
        config=config,
    )
    return report


if __name__ == "__main__":
    # Run the basic example
    # Uncomment the one you want to run:

    # example_basic()
    example_full()
    # example_custom_config()
