from __future__ import annotations


def build_remote_sensing_prompt(class_name: str, n: int) -> str:
    return f"""
You are a remote-sensing visual taxonomy annotator.

Generate exactly {n} candidate descriptions for the remote-sensing scene class:
"{class_name}".

Requirements:
1. Each description must be in English.
2. Each description must describe an overhead, aerial, nadir, or satellite-imagery view.
3. Focus on visible visual cues: objects, layout, geometry, texture, spatial arrangement.
4. Avoid ground-level or human-centric wording such as street-level, indoor, selfie, portrait, close-up, ground level.
5. Each description should contain 6 to 20 whitespace-delimited words if possible.
6. Avoid duplicate wording.
7. Return strict JSON only with this object shape:
{{
  "candidates": [
    {{
      "caption": "...",
      "viewpoint": "overhead|aerial|nadir|satellite",
      "visual_cues": ["...", "..."],
      "spatial_cues": ["...", "..."]
    }}
  ]
}}
""".strip()

