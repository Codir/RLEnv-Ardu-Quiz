"""vf-ardu-quiz: single-turn ArduPilot dataflash-log diagnosis environment.

The model reads a short description of symptoms observed in an ArduPilot
dataflash log and must answer with exactly one snake_case root-cause label
from a closed taxonomy, wrapped in <answer>...</answer> tags.

Design notes (anti-reward-hacking):
- The taxonomy is a closed set given in the system prompt, so exact string
  matching is a fair, deterministic verifier (no LLM judge).
- Every label appears exactly twice in the dataset, so blind constant
  guessing scores ~1/len(TAXONOMY) (~6.7%), well under the 20% red line.
- The format reward is weighted at 0.1 vs 1.0 for correctness, so a
  well-formatted but wrong answer scores <= ~0.09 of the max.
"""

import verifiers as vf
from datasets import Dataset

# Closed root-cause taxonomy. Kept sorted for a deterministic prompt.
TAXONOMY = [
    "barometer_interference",
    "battery_failsafe",
    "compass_calibration_bad",
    "excessive_vibration",
    "frame_resonance",
    "gps_glitch",
    "gps_multipath",
    "gyro_drift_temperature",
    "magnetic_interference_from_power",
    "motor_or_esc_failure",
    "pid_oscillation",
    "power_supply_sag",
    "props_imbalance",
    "radio_failsafe",
    "thrust_loss_overweight",
]

# (symptom description from the dataflash log, root-cause label)
# Exactly 2 cases per label -> balanced dataset, 30 cases total.
CASES = [
    # power_supply_sag
    ("POWR.Vcc drops from 5.0V to 4.3V exactly on current peaks during aggressive "
     "climbs; a brownout reset is recorded mid-flight; battery voltage itself stays healthy.",
     "power_supply_sag"),
    ("Flight controller reboots in flight. Log shows the 5V rail dipping to 4.2V whenever "
     "the camera gimbal slews; power module and battery readings are normal.",
     "power_supply_sag"),
    # magnetic_interference_from_power
    ("MAG.MagField rises from 420 to 780 in lockstep with throttle; compass health is "
     "perfect on the ground with motors off; EKF yaw variance spikes only under load.",
     "magnetic_interference_from_power"),
    ("Compass innovations grow proportionally to battery current draw; the mag interference "
     "test shows >60% throttle correlation; ground calibration passes every time.",
     "magnetic_interference_from_power"),
    # compass_calibration_bad
    ("Copter circles ('toilet-bowls') immediately after takeoff in Loiter regardless of "
     "throttle; COMPASS_OFS values exceed 550 on all axes; no correlation with current draw.",
     "compass_calibration_bad"),
    ("Heading in the log disagrees with the GPS ground track by ~40 degrees at constant "
     "cruise; offsets differ wildly between the two onboard compasses.",
     "compass_calibration_bad"),
    # excessive_vibration
    ("VIBE.Clip0-2 counters increase through the whole flight; Z-axis vibration averages "
     "45 m/s/s; the altitude estimate wobbles and the copter climbs on its own in AltHold.",
     "excessive_vibration"),
    ("Accelerometer clipping is recorded on every axis; the position estimate diverges from "
     "GPS during fast forward flight; VIBE values stay above 30 m/s/s on X and Y.",
     "excessive_vibration"),
    # frame_resonance
    ("FFT of gyro data shows a sharp 82 Hz peak whose amplitude is independent of "
     "throttle; the peak persists across two different prop sets; arms are long and thin.",
     "frame_resonance"),
    ("Post-flight FFT reveals one narrow frequency spike present even at idle hover; "
     "changing motors and balancing props did not move or reduce the peak.",
     "frame_resonance"),
    # gps_glitch
    ("Position jumps 35 m sideways in one EKF update; satellite count falls from 14 to 5 "
     "and HDOP spikes from 0.8 to 2.9 for three seconds; EKF reports a position reset.",
     "gps_glitch"),
    ("GPA.Delta shows a gap in GPS updates followed by a sudden reported position offset; "
     "the vehicle lurches to 'correct' toward the jumped position; sats recover afterwards.",
     "gps_glitch"),
    # gps_multipath
    ("Loiter position wanders in slow circles only when flying between tall buildings; "
     "satellite count stays at 15 and HDOP under 1.0 the whole time.",
     "gps_multipath"),
    ("Reported GPS position oscillates several meters near a metal-roofed warehouse "
     "while raw satellite count and DOP metrics remain excellent.",
     "gps_multipath"),
    # barometer_interference
    ("Barometric altitude drops 4 m every time the copter pitches into fast forward "
     "flight and recovers in hover; GPS altitude stays smooth throughout.",
     "barometer_interference"),
    ("CTUN.Alt spikes whenever the payload bay fan turns on; the foam over the barometer "
     "is missing; GPS and rangefinder altitude show no such jumps.",
     "barometer_interference"),
    # motor_or_esc_failure
    ("RCOU.C3 rails to maximum while the diagonally opposite motor drops to minimum; the "
     "copter yaws uncommanded and loses altitude; motor 3 is cold after landing.",
     "motor_or_esc_failure"),
    ("A sudden yaw twitch is followed by one ESC output saturating high for the rest of the "
     "flight; thrust asymmetry forces a descent despite pilot full throttle.",
     "motor_or_esc_failure"),
    # props_imbalance
    ("Vibration amplitude tracks motor RPM exactly (the FFT peak follows throttle changes); "
     "the IMU shows a strong once-per-revolution frequency; one prop has a visible chip.",
     "props_imbalance"),
    ("X-axis vibration doubles between hover and full throttle with the dominant FFT "
     "peak sliding up in frequency as RPM rises; bearings and frame check out fine.",
     "props_imbalance"),
    # radio_failsafe
    ("RC input channels flatline mid-flight; the EV log records FAILSAFE_RADIO ON and the "
     "mode switches to RTL without pilot input; RSSI decayed steadily before the event.",
     "radio_failsafe"),
    ("All RCIN channels freeze at last known values, then the throttle channel drops below "
     "FS_THR_VALUE; the vehicle enters failsafe RTL 1.2 km from the pilot.",
     "radio_failsafe"),
    # battery_failsafe
    ("BAT.Volt sags below the LOW_VOLT threshold for longer than the failsafe timer; the EV "
     "log shows FAILSAFE_BATT and an automatic LAND; cell voltage was 3.2V under load.",
     "battery_failsafe"),
    ("The vehicle initiates automatic RTL mid-mission; log events show battery failsafe "
     "stage 1 triggered at 21.0V on a 6S pack after eight minutes of aggressive flight.",
     "battery_failsafe"),
    # thrust_loss_overweight
    ("All eight motor outputs sit above 1900 us in a plain hover; altitude slowly drops "
     "even at full collective throttle; takeoff weight was increased by a new payload.",
     "thrust_loss_overweight"),
    ("The learned hover throttle reaches 0.82; the copter cannot climb out of ground "
     "effect with a full sensor package; motors and battery test healthy individually.",
     "thrust_loss_overweight"),
    # pid_oscillation
    ("Rate roll and pitch traces show a sustained 12 Hz oscillation that starts right "
     "after an aggressive ATC_RAT gain increase; motors come back too hot to touch.",
     "pid_oscillation"),
    ("Desired vs actual rate traces oscillate against each other with growing amplitude "
     "in fast descents; the oscillation disappears after halving the rate D gain.",
     "pid_oscillation"),
    # gyro_drift_temperature
    ("The attitude estimate slowly rolls several degrees while the vehicle sits armed and "
     "motionless on a cold morning; drift shrinks as the IMU heater reaches temperature.",
     "gyro_drift_temperature"),
    ("The horizon in the log tilts progressively during the first minutes of flight after "
     "a rapid move from a warm car to freezing air; the IMU temperature curve is still rising.",
     "gyro_drift_temperature"),
]

SYSTEM_PROMPT = (
    "You are an ArduPilot flight-log analyst. You will be shown symptoms extracted "
    "from a dataflash log. Diagnose the single most likely root cause.\n\n"
    "Choose EXACTLY ONE label from this list:\n"
    + "\n".join(f"- {label}" for label in TAXONOMY)
    + "\n\nReply with the label inside tags, e.g. <answer>gps_glitch</answer>. "
    "Think briefly before the tags if needed, but the tags must contain only the label."
)


# Module-level parser so reward functions are directly importable in tests.
PARSER = vf.XMLParser(fields=["answer"], answer_field="answer")


async def exact_match(completion, answer, **kw) -> float:
    """1.0 if the parsed <answer> label equals the ground truth, else 0.0."""
    parsed = PARSER.parse_answer(completion)
    if not parsed:
        return 0.0
    return 1.0 if parsed.strip().lower() == answer else 0.0


def load_environment(num_examples: int = -1, **kwargs) -> vf.Environment:
    """Build the single-turn ArduPilot diagnosis environment.

    Args:
        num_examples: truncate the dataset to the first N cases (-1 = all 30).
    """
    cases = CASES if num_examples == -1 else CASES[:num_examples]
    dataset = Dataset.from_list(
        [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": (
                            f"Symptoms from the dataflash log: {symptom}\n"
                            "Give the diagnosis as one snake_case label in "
                            "<answer>...</answer> tags."
                        ),
                    }
                ],
                "answer": label,
                "info": {"taxonomy_size": len(TAXONOMY)},
            }
            for symptom, label in cases
        ]
    )

    parser = PARSER

    rubric = vf.Rubric(
        funcs=[exact_match, parser.get_format_reward_func()],
        weights=[1.0, 0.1],
        parser=parser,
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )
