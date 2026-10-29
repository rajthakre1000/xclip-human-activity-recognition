
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from transformers import AutoProcessor, XCLIPModel


MODEL_ID = "microsoft/xclip-base-patch16-zero-shot"
NUM_FRAMES = 32

ACTIVITIES = [
    "a person is walking",
    "a person is running",
    "a person is jogging",
    "a person is sitting",
    "a person is standing",
    "a person is jumping",
    "a person is dancing",
    "a person is clapping",
    "a person is playing basketball",
    "a person is playing football",
    "a person is riding a bicycle",
    "a person is swimming",
    "a person is cooking",
    "a person is eating",
    "a person is exercising",
]


# -------------------- PAGE CONFIGURATION --------------------

st.set_page_config(
    page_title="Vision-Language Human Activity Recognition",
    layout="centered",
)

st.title("Vision-Language Human Activity Recognition")
st.markdown("### Raj Thakre")
st.caption("Scholar Number: 24U021008")
st.divider()


# -------------------- LOAD PRETRAINED X-CLIP MODEL --------------------

@st.cache_resource
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = XCLIPModel.from_pretrained(MODEL_ID)
    model.eval()

    return processor, model


# -------------------- EXTRACT VIDEO FRAMES --------------------

def extract_frames(video_path, count=NUM_FRAMES):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            "Unable to open the uploaded video. "
            "Please try another video."
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError("No readable frames were found in the video.")

    # Sample frames evenly throughout the video.
    indices = np.linspace(
        0,
        total_frames - 1,
        min(count, total_frames),
        dtype=int,
    )

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError("Could not extract frames from this video.")

    # X-CLIP requires 32 frames for this checkpoint.
    # Repeat the last available frame for shorter videos.
    while len(frames) < count:
        frames.append(frames[-1].copy())

    return frames[:count]


# -------------------- PREDICT HUMAN ACTIVITY --------------------

def predict_activity(frames, processor, model):
    if len(frames) != NUM_FRAMES:
        raise ValueError(
            f"Expected {NUM_FRAMES} frames, received {len(frames)}."
        )

    # One video containing 32 RGB frames.
    inputs = processor(
        text=ACTIVITIES,
        videos=[frames],
        return_tensors="pt",
        padding=True,
    )

    pixel_values = inputs["pixel_values"]

    # Diagnostic output in the VS Code terminal.
    print("X-CLIP pixel_values shape:", pixel_values.shape)

    if pixel_values.ndim != 5:
        raise ValueError(
            "Unexpected video tensor shape: "
            f"{tuple(pixel_values.shape)}. "
            "Expected (batch, frames, channels, height, width)."
        )

    if pixel_values.shape[0] != 1:
        raise ValueError(
            f"Expected batch size 1, got {pixel_values.shape[0]}."
        )

    if pixel_values.shape[1] != NUM_FRAMES:
        raise ValueError(
            f"Expected {NUM_FRAMES} frames in the tensor, "
            f"got {pixel_values.shape[1]}."
        )

    with torch.inference_mode():
        outputs = model(**inputs)

        scores = outputs.logits_per_video[0]
        probabilities = torch.softmax(scores, dim=-1)

    top_indices = torch.argsort(
        probabilities,
        descending=True,
    )[:5]

    results = []

    for index in top_indices.tolist():
        results.append(
            (
                ACTIVITIES[index],
                probabilities[index].item() * 100,
            )
        )

    return results


# -------------------- VIDEO UPLOAD --------------------

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:

    st.video(uploaded_file)

    if st.button("Recognize Activity", type="primary"):

        try:
            with st.spinner("Loading pretrained X-CLIP model..."):
                processor, model = load_model()

            device_name = (
                "CUDA GPU"
                if torch.cuda.is_available()
                else "CPU"
            )

            st.caption(f"Device: {device_name}")

            with tempfile.TemporaryDirectory() as temp_dir:

                video_path = Path(temp_dir) / "input_video.mp4"
                video_path.write_bytes(uploaded_file.getvalue())

                with st.spinner("Extracting 32 video frames..."):
                    frames = extract_frames(
                        video_path,
                        count=NUM_FRAMES,
                    )

                st.write(f"Frames processed: {len(frames)}")

                with st.spinner("Recognizing human activity..."):
                    results = predict_activity(
                        frames,
                        processor,
                        model,
                    )

            # -------------------- DISPLAY RESULTS --------------------

            st.success("Activity analysis completed.")

            st.subheader("Predicted Activity")
            st.markdown(f"## {results[0][0].capitalize()}")

            st.metric(
                "Relative score",
                f"{results[0][1]:.2f}%",
            )

            st.subheader("Top 5 Activity Predictions")

            for label, score in results:
                st.write(f"{label.capitalize()} — {score:.2f}%")
                st.progress(min(score / 100, 1.0))

            st.caption(
                "Scores are normalized across the candidate activity "
                "labels. They are not calibrated probabilities or "
                "a measure of model accuracy."
            )

        except Exception as error:
            st.error("Unable to analyze this video.")
            st.exception(error)

            st.info(
                "Check the video format, model download, and "
                "VS Code terminal output. If the problem persists, "
                "share the complete traceback and the printed "
                "X-CLIP pixel_values shape."
            )
