"""
checkpoints.py — Checkpoint definitions (dynamic rubric registry).

Each sub-metric has a list of CheckpointDef objects.  Checkpoints that carry
an ``applicable_when`` condition are only activated when the video's
ContentProfile matches.  This prevents unfair penalisation (e.g. scoring
"character clothing consistency" on a landscape video with no characters).

Rubric anchors follow the Prometheus-Vision pattern: each score level (1-5)
has a concrete, observable description so the VLM uses the full range.

Organisation:
    CHECKPOINTS[<metric_name>] → list[CheckpointDef]
"""

from __future__ import annotations

from .schemas import CheckpointDef, CheckpointType, RubricAnchor

# ============================================================================
#  VIDEO AGENT
# ============================================================================

# ---------- temporal_coherence ----------

TEMPORAL_COHERENCE = [
    CheckpointDef(
        id="char_face_consistency",
        question="Do the characters' facial features (face shape, eye colour, skin tone, hairstyle) remain consistent across consecutive shots?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={"has_characters": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Face is pixel-level consistent; clearly the same person in every shot."),
            RubricAnchor(value=4, label="Good",        description="Minor shifts at extreme angles, but still clearly recognisable as the same person."),
            RubricAnchor(value=3, label="Noticeable",  description="Visible drift in proportions or skin tone; identity is plausible but questionable."),
            RubricAnchor(value=2, label="Poor",        description="Faces look like different people in some shots."),
            RubricAnchor(value=1, label="Broken",      description="Character face changes drastically between shots — effectively different characters."),
        ],
    ),
    CheckpointDef(
        id="char_clothing_consistency",
        question="Do characters' clothing, accessories, and body proportions stay consistent within the same scene (excluding intentional costume changes)?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.15,
        applicable_when={"has_characters": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Outfit colours, patterns, and fit are identical across shots in the same scene."),
            RubricAnchor(value=4, label="Good",        description="Very minor colour shade shift; overall outfit clearly the same."),
            RubricAnchor(value=3, label="Noticeable",  description="Some clothing details change (e.g. pattern disappears, sleeve length changes)."),
            RubricAnchor(value=2, label="Poor",        description="Outfit changes noticeably between shots with no narrative reason."),
            RubricAnchor(value=1, label="Broken",      description="Completely different outfits between consecutive shots in the same scene."),
        ],
    ),
    CheckpointDef(
        id="object_permanence",
        question="Do objects held by or near characters persist correctly? (e.g. a cup doesn't vanish/appear spontaneously)",
        checkpoint_type=CheckpointType.BINARY,
        weight=0.15,
        applicable_when={"has_held_objects": True},
    ),
    CheckpointDef(
        id="background_consistency",
        question="Is the background environment consistent within the same scene? (walls, furniture, landscape don't morph or change)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.15,
        applicable_when={},  # always applicable
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Background is perfectly stable — identical layout, colours, and objects across shots in the same scene."),
            RubricAnchor(value=4, label="Good",        description="Background is recognisably the same; very minor texture changes only."),
            RubricAnchor(value=3, label="Noticeable",  description="Background shifts noticeably (e.g. wall colour changes, furniture rearranges slightly)."),
            RubricAnchor(value=2, label="Poor",        description="Background is different enough that continuity feels broken."),
            RubricAnchor(value=1, label="Broken",      description="Background changes completely between shots in what should be the same scene."),
        ],
    ),
    CheckpointDef(
        id="scale_proportion",
        question="Do the relative sizes and proportions of people/objects remain physically plausible across shots?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.10,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="All sizes are physically consistent — no distortion."),
            RubricAnchor(value=4, label="Good",        description="Proportions are mostly correct; minor camera-angle-induced differences only."),
            RubricAnchor(value=3, label="Noticeable",  description="Some objects or characters appear to change size without justification."),
            RubricAnchor(value=2, label="Poor",        description="Obvious size inconsistencies (e.g. a person is much larger/smaller between cuts)."),
            RubricAnchor(value=1, label="Broken",      description="Severe proportion errors — objects or people resize dramatically."),
        ],
    ),
    CheckpointDef(
        id="motion_continuity",
        question="Is the direction and speed of motion continuous across cuts? (no teleportation or sudden direction reversal without cause)",
        checkpoint_type=CheckpointType.BINARY,
        weight=0.15,
        applicable_when={},
    ),
    CheckpointDef(
        id="temporal_logic",
        question="When time progression is implied (day→night, season change), is it logically justified by the narrative?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.10,
        applicable_when={"has_scene_changes": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="All time jumps are clearly signalled and narratively motivated."),
            RubricAnchor(value=4, label="Good",        description="Time jumps are plausible; minor ambiguity about duration."),
            RubricAnchor(value=3, label="Noticeable",  description="Some time shifts feel abrupt or under-motivated."),
            RubricAnchor(value=2, label="Poor",        description="Confusing temporal jumps that undermine coherence."),
            RubricAnchor(value=1, label="Broken",      description="Time progression is nonsensical (e.g. night→day→night in seconds with no reason)."),
        ],
    ),
]

# ---------- lighting_consistency ----------

LIGHTING_CONSISTENCY = [
    CheckpointDef(
        id="light_direction",
        question="Is the direction of the key light source consistent across shots within the same scene?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Shadows and highlights consistently indicate the same light source direction."),
            RubricAnchor(value=4, label="Good",        description="Light direction is mostly consistent; very minor shifts."),
            RubricAnchor(value=3, label="Noticeable",  description="Light direction shifts between some shots without justification."),
            RubricAnchor(value=2, label="Poor",        description="Light direction is frequently inconsistent."),
            RubricAnchor(value=1, label="Broken",      description="Light comes from obviously different directions in consecutive same-scene shots."),
        ],
    ),
    CheckpointDef(
        id="shadow_consistency",
        question="Are shadows physically plausible and consistent with the light sources shown?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Shadows are physically accurate and match the light source in every shot."),
            RubricAnchor(value=4, label="Good",        description="Shadows are mostly correct; minor inconsistencies."),
            RubricAnchor(value=3, label="Noticeable",  description="Some shadows are misaligned or missing where they should be."),
            RubricAnchor(value=2, label="Poor",        description="Shadows frequently contradict the visible light source."),
            RubricAnchor(value=1, label="Broken",      description="Shadows are absent, inverted, or nonsensical."),
        ],
    ),
    CheckpointDef(
        id="color_temperature",
        question="Is the colour temperature (warm/cool) consistent within the same scene, without unjustified sudden shifts?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Colour temperature is uniform throughout each scene."),
            RubricAnchor(value=4, label="Good",        description="Very slight colour temperature drift that most viewers wouldn't notice."),
            RubricAnchor(value=3, label="Noticeable",  description="Visible warm↔cool shift between shots in the same scene."),
            RubricAnchor(value=2, label="Poor",        description="Jarring colour temperature changes."),
            RubricAnchor(value=1, label="Broken",      description="Scene oscillates wildly between warm and cool tones."),
        ],
    ),
    CheckpointDef(
        id="exposure_stability",
        question="Is the exposure/brightness level stable, without sudden flashes or darkening?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Exposure is perfectly stable across all shots in each scene."),
            RubricAnchor(value=4, label="Good",        description="Minor exposure variation that doesn't distract."),
            RubricAnchor(value=3, label="Noticeable",  description="Some shots are noticeably brighter/darker than neighbours."),
            RubricAnchor(value=2, label="Poor",        description="Frequent brightness jumps."),
            RubricAnchor(value=1, label="Broken",      description="Severe flashing or extreme exposure changes between cuts."),
        ],
    ),
]

# ============================================================================
#  SCRIPT AGENT
# ============================================================================

SCRIPT_REASONABLENESS = [
    CheckpointDef(
        id="event_chain_logic",
        question="Does each story event logically follow from the previous one, with plausible cause-and-effect?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Every transition between events is causally motivated — no plot holes."),
            RubricAnchor(value=4, label="Good",        description="Cause-and-effect is mostly clear; one minor logical stretch."),
            RubricAnchor(value=3, label="Noticeable",  description="Some events feel disconnected or require viewer inference to connect."),
            RubricAnchor(value=2, label="Poor",        description="Multiple events lack causal connection; narrative feels random."),
            RubricAnchor(value=1, label="Broken",      description="Events are entirely disconnected — no coherent story."),
        ],
    ),
    CheckpointDef(
        id="character_motivation",
        question="Do characters act with plausible motivations? (actions make sense given what we know about them)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={"has_characters": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Every character action feels motivated and in-character."),
            RubricAnchor(value=4, label="Good",        description="Motivations are mostly clear; one action feels slightly arbitrary."),
            RubricAnchor(value=3, label="Noticeable",  description="Some character actions feel unmotivated or out-of-character."),
            RubricAnchor(value=2, label="Poor",        description="Characters frequently do things that contradict their established behaviour."),
            RubricAnchor(value=1, label="Broken",      description="Characters act randomly with no discernible motivation."),
        ],
    ),
    CheckpointDef(
        id="pacing_structure",
        question="Does the narrative have a clear beginning-middle-end structure with appropriate pacing for a ~1 minute video?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Clear setup→conflict→resolution arc; pacing feels natural for the duration."),
            RubricAnchor(value=4, label="Good",        description="Structure is recognisable; pacing is slightly uneven."),
            RubricAnchor(value=3, label="Noticeable",  description="Structure is loose — story meanders or rushes."),
            RubricAnchor(value=2, label="Poor",        description="No clear structure; feels like a random sequence of events."),
            RubricAnchor(value=1, label="Broken",      description="No narrative arc whatsoever."),
        ],
    ),
    CheckpointDef(
        id="internal_consistency",
        question="Are there any internal contradictions? (e.g. a character is said to be alone but others are present, time of day contradicts dialogue)",
        checkpoint_type=CheckpointType.BINARY,
        weight=0.20,
        applicable_when={},
    ),
    CheckpointDef(
        id="dialogue_naturalness",
        question="Is the dialogue (if any) natural, context-appropriate, and free of unintentional repetition?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.15,
        applicable_when={"has_dialogue": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Dialogue sounds natural and advances the story."),
            RubricAnchor(value=4, label="Good",        description="Dialogue is mostly natural; minor awkwardness."),
            RubricAnchor(value=3, label="Noticeable",  description="Some lines feel wooden, repetitive, or out of place."),
            RubricAnchor(value=2, label="Poor",        description="Dialogue frequently sounds unnatural or robotic."),
            RubricAnchor(value=1, label="Broken",      description="Dialogue is incomprehensible or completely unnatural."),
        ],
    ),
]

SCRIPT_NOVELTY = [
    CheckpointDef(
        id="premise_originality",
        question="Is the core premise/concept original or surprising, rather than a common cliché?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.35,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Highly original",   description="Premise is fresh, surprising, and unlikely to have been seen before."),
            RubricAnchor(value=4, label="Somewhat original",  description="Interesting twist on a known concept."),
            RubricAnchor(value=3, label="Average",             description="Standard premise — not cliché but not surprising."),
            RubricAnchor(value=2, label="Derivative",          description="Heavily borrows from well-known templates."),
            RubricAnchor(value=1, label="Cliché",              description="Extremely predictable, overused concept."),
        ],
    ),
    CheckpointDef(
        id="narrative_surprise",
        question="Does the story contain any unexpected turns, reveals, or creative choices that keep the viewer engaged?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.35,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Highly surprising", description="Multiple genuinely unexpected moments that feel earned."),
            RubricAnchor(value=4, label="Some surprise",      description="At least one notable surprise element."),
            RubricAnchor(value=3, label="Predictable",         description="Story proceeds as expected without surprises."),
            RubricAnchor(value=2, label="Very predictable",    description="Every beat is telegraphed far in advance."),
            RubricAnchor(value=1, label="No engagement",       description="Completely flat — no creative choices whatsoever."),
        ],
    ),
    CheckpointDef(
        id="visual_creativity",
        question="Are there creative visual storytelling techniques? (unusual angles, metaphorical imagery, creative transitions)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Highly creative",   description="Inventive visual language that enhances the narrative."),
            RubricAnchor(value=4, label="Some creativity",     description="A few interesting visual choices."),
            RubricAnchor(value=3, label="Standard",            description="Competent but conventional visual approach."),
            RubricAnchor(value=2, label="Uninspired",          description="Generic visual presentation."),
            RubricAnchor(value=1, label="Flat",                description="No visual storytelling — static and lifeless."),
        ],
    ),
]

# ============================================================================
#  AUDIO AGENT
# ============================================================================

BGM_CONSISTENCY = [
    CheckpointDef(
        id="bgm_mood_match",
        question="Does the background music mood match the visual/narrative mood of each scene?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={"has_background_music": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Music mood perfectly complements every scene's emotional tone."),
            RubricAnchor(value=4, label="Good",        description="Music mood mostly matches; minor misalignment in one scene."),
            RubricAnchor(value=3, label="Noticeable",  description="Music feels generic — not mismatched but not enhancing either."),
            RubricAnchor(value=2, label="Poor",        description="Music mood clashes with the visual mood in some scenes."),
            RubricAnchor(value=1, label="Broken",      description="Music is completely wrong for the content (e.g. upbeat during tragedy)."),
        ],
    ),
    CheckpointDef(
        id="bgm_transition_smoothness",
        question="Do music transitions between scenes feel smooth, without abrupt cuts or jarring volume changes?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={"has_background_music": True, "has_scene_changes": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Music transitions are seamless — crossfades or natural breakpoints."),
            RubricAnchor(value=4, label="Good",        description="Transitions are mostly smooth; one minor audible cut."),
            RubricAnchor(value=3, label="Noticeable",  description="Some music cuts are noticeable but not jarring."),
            RubricAnchor(value=2, label="Poor",        description="Music frequently cuts abruptly at scene changes."),
            RubricAnchor(value=1, label="Broken",      description="Harsh audio cuts or overlapping tracks at every transition."),
        ],
    ),
    CheckpointDef(
        id="bgm_tempo_pacing",
        question="Does the music tempo/rhythm match the pacing of the visual action?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={"has_background_music": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Tempo mirrors the action pacing — builds during intense moments, relaxes during calm."),
            RubricAnchor(value=4, label="Good",        description="Tempo is appropriate; one segment feels slightly mismatched."),
            RubricAnchor(value=3, label="Noticeable",  description="Tempo is constant and doesn't adapt to pacing changes."),
            RubricAnchor(value=2, label="Poor",        description="Tempo actively contradicts the visual pacing."),
            RubricAnchor(value=1, label="Broken",      description="Music rhythm is completely out of sync with the visual action."),
        ],
    ),
    CheckpointDef(
        id="bgm_volume_balance",
        question="Is the BGM volume balanced against dialogue/narration? (not drowning out speech, not inaudible)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={"has_background_music": True, "has_dialogue": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Music ducks appropriately under dialogue; perfectly balanced."),
            RubricAnchor(value=4, label="Good",        description="Balance is good; music occasionally slightly loud."),
            RubricAnchor(value=3, label="Noticeable",  description="Music sometimes competes with dialogue."),
            RubricAnchor(value=2, label="Poor",        description="Dialogue is hard to hear over the music in multiple scenes."),
            RubricAnchor(value=1, label="Broken",      description="Music completely drowns out or is entirely absent during dialogue."),
        ],
    ),
]

NARRATION_REASONABLENESS = [
    CheckpointDef(
        id="speech_timing",
        question="Does speech start and end at natural points? (no cut-off mid-word, no awkward silence gaps)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={"has_dialogue": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="All speech segments start and end cleanly at natural sentence/phrase boundaries."),
            RubricAnchor(value=4, label="Good",        description="Speech timing is mostly natural; one minor clipping."),
            RubricAnchor(value=3, label="Noticeable",  description="Some speech is cut off or has unnatural pauses."),
            RubricAnchor(value=2, label="Poor",        description="Frequent mid-word cuts or awkwardly long gaps."),
            RubricAnchor(value=1, label="Broken",      description="Speech is severely clipped, overlapping, or has nonsensical timing."),
        ],
    ),
    CheckpointDef(
        id="speech_emotion_fit",
        question="Does the vocal emotion/tone match the scene's mood? (e.g. not cheerful during a sad scene)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.35,
        applicable_when={"has_dialogue": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Vocal emotion precisely matches the scene's mood in every segment."),
            RubricAnchor(value=4, label="Good",        description="Emotion is mostly appropriate; one segment feels slightly off."),
            RubricAnchor(value=3, label="Noticeable",  description="Vocal emotion is flat/neutral regardless of scene mood."),
            RubricAnchor(value=2, label="Poor",        description="Vocal emotion clashes with the scene mood in some segments."),
            RubricAnchor(value=1, label="Broken",      description="Vocal emotion is completely wrong (e.g. laughing during crisis)."),
        ],
    ),
    CheckpointDef(
        id="speech_intelligibility",
        question="Is the speech clearly intelligible? (no garbling, mumbling, or AI-artefact distortion)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.35,
        applicable_when={"has_dialogue": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Every word is crystal clear and easily understood."),
            RubricAnchor(value=4, label="Good",        description="Almost all speech is clear; one or two words are slightly unclear."),
            RubricAnchor(value=3, label="Noticeable",  description="Some phrases are hard to understand without replaying."),
            RubricAnchor(value=2, label="Poor",        description="Significant portions are garbled or distorted."),
            RubricAnchor(value=1, label="Broken",      description="Speech is largely unintelligible."),
        ],
    ),
]

# ============================================================================
#  STABILITY AGENT
# ============================================================================

GENERATION_STABILITY = [
    CheckpointDef(
        id="visual_artifact_frequency",
        question="How frequently do visual artefacts appear? (flickering, warping, morphing, ghosting)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Clean",      description="No visible artefacts throughout the entire video."),
            RubricAnchor(value=4, label="Rare",        description="1-2 very brief artefacts that most viewers would miss."),
            RubricAnchor(value=3, label="Occasional",  description="Artefacts appear in a few scenes but don't dominate."),
            RubricAnchor(value=2, label="Frequent",    description="Artefacts in many scenes; distracting."),
            RubricAnchor(value=1, label="Pervasive",   description="Constant artefacts throughout — nearly unwatchable."),
        ],
    ),
    CheckpointDef(
        id="resolution_sharpness",
        question="Is the video sharp and well-resolved, or does it appear blurry/soft?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Sharp",      description="Crisp detail throughout; looks like a professional render."),
            RubricAnchor(value=4, label="Good",        description="Mostly sharp; minor softness in some areas."),
            RubricAnchor(value=3, label="Soft",        description="Noticeably softer than expected for the resolution."),
            RubricAnchor(value=2, label="Blurry",      description="Significant blur that obscures detail."),
            RubricAnchor(value=1, label="Very blurry", description="Extremely blurry — looks like a low-resolution upscale."),
        ],
    ),
    CheckpointDef(
        id="temporal_degradation",
        question="Does the video quality degrade over time? (e.g. second half is blurrier, more artefacts than first half)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Stable",     description="Quality is consistent from start to finish."),
            RubricAnchor(value=4, label="Slight",      description="Very minor quality drop in the last few seconds."),
            RubricAnchor(value=3, label="Noticeable",  description="Visible quality degradation in the second half."),
            RubricAnchor(value=2, label="Significant",  description="Second half is markedly worse than the first."),
            RubricAnchor(value=1, label="Severe",      description="Quality collapses dramatically — last portion is barely recognisable."),
        ],
    ),
    CheckpointDef(
        id="color_consistency",
        question="Are colours natural and consistent? (no random colour banding, posterisation, or saturation spikes)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.15,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Natural",    description="Colours are natural, well-graded, and consistent."),
            RubricAnchor(value=4, label="Good",        description="Colours are mostly natural; very minor inconsistency."),
            RubricAnchor(value=3, label="Noticeable",  description="Some colour banding or saturation issues visible."),
            RubricAnchor(value=2, label="Poor",        description="Frequent colour artefacts."),
            RubricAnchor(value=1, label="Broken",      description="Severe posterisation, banding, or colour corruption."),
        ],
    ),
    CheckpointDef(
        id="duration_completeness",
        question="Does the video reach the expected duration (~60 seconds) without cutting short or having frozen/repeated frames at the end?",
        checkpoint_type=CheckpointType.BINARY,
        weight=0.20,
        applicable_when={},
    ),
]

# ============================================================================
#  CROSS-MODAL AGENT
# ============================================================================

TEXT_VIDEO_CONSISTENCY = [
    CheckpointDef(
        id="scene_presence",
        question="Are the key scenes/events described in the script visually present in the video?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="All present",      description="Every scripted scene is visually realised."),
            RubricAnchor(value=4, label="Mostly present",    description="One minor scripted detail is missing or vague."),
            RubricAnchor(value=3, label="Partial",            description="Some scripted scenes are present; others are missing or replaced."),
            RubricAnchor(value=2, label="Sparse",             description="Only a few scripted scenes appear; much of the script is not reflected."),
            RubricAnchor(value=1, label="Absent",             description="The video bears almost no resemblance to the script."),
        ],
    ),
    CheckpointDef(
        id="scene_order",
        question="Do the scripted events appear in the correct temporal order in the video?",
        checkpoint_type=CheckpointType.BINARY,
        weight=0.15,
        applicable_when={},
    ),
    CheckpointDef(
        id="character_matching",
        question="Do the characters' visual appearance (gender, age, attire) match the script descriptions?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.25,
        applicable_when={"has_characters": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Characters perfectly match script descriptions in every detail."),
            RubricAnchor(value=4, label="Good",        description="Characters are recognisable from descriptions; minor deviations."),
            RubricAnchor(value=3, label="Partial",     description="Some character attributes match; others are noticeably different."),
            RubricAnchor(value=2, label="Poor",        description="Characters look quite different from descriptions."),
            RubricAnchor(value=1, label="Mismatched",  description="Characters bear no resemblance to script descriptions."),
        ],
    ),
    CheckpointDef(
        id="hallucinated_content",
        question="Does the video contain significant visual elements NOT mentioned in the script? (hallucination check)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.15,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Faithful",     description="No hallucinated content — everything shown is scripted."),
            RubricAnchor(value=4, label="Minor extras",  description="One or two extra details that don't contradict the script."),
            RubricAnchor(value=3, label="Noticeable",    description="Some unscripted elements appear but don't dominate."),
            RubricAnchor(value=2, label="Significant",    description="Major unscripted elements that confuse the narrative."),
            RubricAnchor(value=1, label="Hallucinated",   description="Video is full of content not in the script."),
        ],
    ),
    CheckpointDef(
        id="setting_accuracy",
        question="Do the visual settings/locations match the script descriptions? (indoor/outdoor, time of day, weather)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.15,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Every setting precisely matches script descriptions."),
            RubricAnchor(value=4, label="Good",        description="Settings are mostly accurate; minor environmental differences."),
            RubricAnchor(value=3, label="Partial",     description="Some settings match; others are noticeably different."),
            RubricAnchor(value=2, label="Poor",        description="Most settings don't match descriptions."),
            RubricAnchor(value=1, label="Wrong",       description="Settings are completely different from what the script describes."),
        ],
    ),
]

VIDEO_AUDIO_CONSISTENCY = [
    CheckpointDef(
        id="lip_sync_quality",
        question="When characters speak, do their lip movements match the audio timing?",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={"has_characters": True, "has_dialogue": True},
        rubric=[
            RubricAnchor(value=5, label="Perfect sync",   description="Lip movements precisely match audio — professional quality."),
            RubricAnchor(value=4, label="Good sync",       description="Lips are mostly in sync; very slight delay/advance."),
            RubricAnchor(value=3, label="Noticeable lag",  description="Visible but tolerable desynchronisation."),
            RubricAnchor(value=2, label="Poor sync",       description="Obvious mismatch between lip movement and audio."),
            RubricAnchor(value=1, label="No sync",         description="Lips move independently of audio — like a badly dubbed film."),
        ],
    ),
    CheckpointDef(
        id="sound_event_alignment",
        question="Do sound effects match the visual events that produce them? (e.g. door slam → bang sound at the right moment)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.30,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Every visual event has a matching, well-timed sound effect."),
            RubricAnchor(value=4, label="Good",        description="Most events are properly accompanied; one minor timing issue."),
            RubricAnchor(value=3, label="Partial",     description="Some events have sounds, others are silent or mistimed."),
            RubricAnchor(value=2, label="Poor",        description="Frequent mismatches between sounds and visual events."),
            RubricAnchor(value=1, label="Broken",      description="Sounds bear no relation to visual events."),
        ],
    ),
    CheckpointDef(
        id="audio_continuity",
        question="Is the audio continuous across shot transitions? (no abrupt cuts, pops, or silence gaps)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Seamless",   description="Audio flows naturally across all transitions."),
            RubricAnchor(value=4, label="Good",        description="One minor audio hiccup at a transition."),
            RubricAnchor(value=3, label="Noticeable",  description="A few audible cuts or pops at transitions."),
            RubricAnchor(value=2, label="Poor",        description="Frequent audio discontinuities."),
            RubricAnchor(value=1, label="Broken",      description="Audio cuts out, pops, or glitches at most transitions."),
        ],
    ),
    CheckpointDef(
        id="ambient_sound_match",
        question="Does the ambient audio match the visual environment? (e.g. outdoor sounds for outdoor scenes)",
        checkpoint_type=CheckpointType.LIKERT,
        weight=0.20,
        applicable_when={},
        rubric=[
            RubricAnchor(value=5, label="Perfect",    description="Ambient audio perfectly matches every visual environment."),
            RubricAnchor(value=4, label="Good",        description="Ambient is mostly appropriate; one scene feels slightly off."),
            RubricAnchor(value=3, label="Generic",     description="Ambient audio is present but generic — doesn't enhance immersion."),
            RubricAnchor(value=2, label="Mismatched",  description="Ambient audio clashes with the visual environment."),
            RubricAnchor(value=1, label="Absent/wrong", description="No ambient audio, or completely wrong ambience."),
        ],
    ),
]

# ============================================================================
#  Master registry: metric_name → list[CheckpointDef]
# ============================================================================

CHECKPOINTS: dict[str, list[CheckpointDef]] = {
    # Video agent
    "temporal_coherence": TEMPORAL_COHERENCE,
    "lighting_consistency": LIGHTING_CONSISTENCY,
    # Script agent
    "script_reasonableness": SCRIPT_REASONABLENESS,
    "script_novelty": SCRIPT_NOVELTY,
    # Audio agent
    "bgm_consistency": BGM_CONSISTENCY,
    "narration_reasonableness": NARRATION_REASONABLENESS,
    # Stability agent
    "generation_stability": GENERATION_STABILITY,
    # Cross-modal agent
    "text_video_consistency": TEXT_VIDEO_CONSISTENCY,
    "video_audio_consistency": VIDEO_AUDIO_CONSISTENCY,
}
