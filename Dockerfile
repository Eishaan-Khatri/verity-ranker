# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Set default command (run validation script as sanity check, or the ranking script)
CMD ["python", "rank.py", "--jd", "data/hackathon/jd.txt", "--candidates", "candidates.jsonl", "--cache-dir", "cache", "--output", "submission/ranked_output.csv"]
