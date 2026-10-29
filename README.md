# X-CLIP Based Human Activity Recognition

## About the Project

This project identifies human activities from video clips using Python, PyTorch, and the X-CLIP vision-language model. A user can upload a video through a Streamlit interface, and the application predicts the activity based on the visual content.

The aim is to make video-based activity recognition simple to use without requiring model training every time a video is analysed.

## Model Used

A pretrained vision-language model designed for video understanding.

The model analyses sampled video frames and compares their visual features with text descriptions of possible activities. It selects the activity with the strongest matching score.

The selected checkpoint was pretrained using the **Kinetics-400** action-recognition dataset. The current application uses its pretrained weights for zero-shot inference; it does not train a new model from scratch.

## Dataset

**Dataset:** Kinetics-400

Kinetics-400 contains videos labelled with 400 human action categories and is used in the pretraining of the selected X-CLIP checkpoint.

For future fine-tuning or independent evaluation, datasets such as UCF101 and HMDB51 can also be considered.

## Performance

| Metric | Result |
|---|---|
| Reported average accuracy | 87% |
| Training videos used in this project | Not applicable — pretrained model |
| Test videos evaluated | To be measured and documented |

The 87% figure should be described as a measured result only if it comes from an actual evaluation. The pretrained model's original training data and this project's test videos are separate.

## Technologies Used

- Python
- Streamlit
- PyTorch
- Hugging Face Transformers
- OpenCV
- NumPy

## Run the Project

Clone the repository:

```bash
git clone https://github.com/rajthakre1000/xclip-human-activity-recognition.git
cd xclip-human-activity-recognition
```

Install the dependencies and start the application:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a video in the browser and view the predicted activity.

## Limitations

Prediction quality depends on video quality, frame sampling, and how closely an activity matches the model's learned concepts. Similar activities may be confused, and results can vary across videos.

## Reference
- [X-CLIP research paper](https://arxiv.org/abs/2208.02816)

## Author

**Rajkumar Thakre**  
Computer Science and Engineering, IIIT Bhopal
