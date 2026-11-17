from PIL import Image


class BackgroundRemover:
    def __init__(self, model_path):
        if not model_path.is_file():
            raise FileNotFoundError("Local U2-Net ONNX model is missing")
        import onnxruntime

        self.session = onnxruntime.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def apply(self, image):
        import numpy as np

        resized = image.convert("RGB").resize((320, 320), Image.Resampling.LANCZOS)
        pixels = np.asarray(resized, dtype=np.float32)
        pixels /= max(float(pixels.max()), 1e-6)
        mean = np.array((0.485, 0.456, 0.406), dtype=np.float32)
        std = np.array((0.229, 0.224, 0.225), dtype=np.float32)
        batch = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None])
        foreground = self.session.run([self.output_name], {self.input_name: batch})[0][0, 0]
        low, high = float(foreground.min()), float(foreground.max())
        if not np.isfinite(foreground).all() or high <= low:
            raise RuntimeError("Background model did not return a usable foreground mask")
        alpha = np.rint((foreground - low) * (255 / (high - low))).clip(0, 255).astype(np.uint8)
        mask = Image.fromarray(alpha).resize(image.size, Image.Resampling.LANCZOS)
        result = image.convert("RGBA")
        result.putalpha(mask)
        return result
