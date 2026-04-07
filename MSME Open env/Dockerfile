FROM python:3.11-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy project
COPY --chown=user . .

# HF Spaces uses port 7860
EXPOSE 7860

# Environment variables (set as HF Secrets)
ENV API_BASE_URL=""
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV ENV_URL="http://localhost:7860"

CMD ["uvicorn", "env.server:app", "--host", "0.0.0.0", "--port", "7860"]
