FROM python:3.12-slim

# HF Spaces UID 1000 foydalanuvchi bilan ishlaydi
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU uchun yengil torch
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Faqat kerakli fayllar
COPY --chown=user app ./app
COPY --chown=user deep_learning/__init__.py ./deep_learning/__init__.py
COPY --chown=user deep_learning/src ./deep_learning/src
COPY --chown=user deep_learning/models ./deep_learning/models

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]