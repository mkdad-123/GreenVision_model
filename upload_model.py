from huggingface_hub import HfApi

api = HfApi()

HF_TOKEN = "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"   # استبدل بالـ token حقك
REPO_ID = "makdadTaleb/plant-disease-cnn"  # استبدل باسم حسابك واسم الريبو
MODEL_FILE = "best_model.pth"

# يرفع الملف إلى الريبو (في المسار الجذري داخل الريبو)
api.upload_file(
    path_or_fileobj=MODEL_FILE,
    path_in_repo="best_model.pth",
    repo_id=REPO_ID,
    token=HF_TOKEN
)
print("Upload finished")
