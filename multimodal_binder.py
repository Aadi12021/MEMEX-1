import json
import os
from openai import OpenAI
from percept_schemas import PerceptObject, RawPerceptInput
import config


class MultimodalBinder:
    """
    Stage 1 of PERCEPT-1.

    Takes raw heterogeneous input (text, JSON/CSV, image) and binds
    it into a single unified PerceptObject.

    Cognitive analog: sensory integration — the brain doesn't process
    vision, touch, and sound in isolation. It binds them into one
    coherent perceptual object before attention ever fires.
    """

    def __init__(self):
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)

    def _describe_image(self, image_base64: str, media_type: str) -> str:
        """
        Uses GPT-4o Vision to convert an image into a dense semantic
        description suitable for embedding and comparison.
        """
        response = self.client.chat.completions.create(
            model=config.MODEL_VISION,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_base64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Describe this image in dense, factual terms. "
                                "Focus on objects, relationships, spatial layout, and any text visible. "
                                "Output one concise paragraph. No fluff."
                            ),
                        },
                    ],
                }
            ],
        )
        return response.choices[0].message.content.strip()

    def _normalize_structured(self, data: dict) -> str:
        """Flattens structured data into a readable key-value summary."""
        lines = []
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                lines.append(f"{key}: {json.dumps(value)}")
            else:
                lines.append(f"{key}: {value}")
        return "Structured data — " + " | ".join(lines)

    def bind(self, raw: RawPerceptInput) -> PerceptObject:
        """
        Fuses all available modalities into a single PerceptObject.
        Modalities are concatenated with semantic delimiters so the
        downstream embedding can capture cross-modal relationships.
        """
        fragments = []
        modalities = []

        if raw.text:
            fragments.append(f"[TEXT] {raw.text.strip()}")
            modalities.append("text")

        if raw.structured:
            normalized = self._normalize_structured(raw.structured)
            fragments.append(f"[STRUCTURED] {normalized}")
            modalities.append("structured")

        if raw.image_base64:
            print("  👁️  [BINDER] Invoking GPT-4o Vision for image description...")
            image_description = self._describe_image(
                raw.image_base64, raw.image_media_type or "image/jpeg"
            )
            fragments.append(f"[IMAGE] {image_description}")
            modalities.append("image")

        if not fragments:
            raise ValueError("RawPerceptInput must contain at least one modality.")

        fused_text = "\n".join(fragments)

        return PerceptObject(
            fused_text=fused_text,
            modalities_present=modalities,
            raw_input=raw,
        )