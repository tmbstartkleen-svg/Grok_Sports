FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV NETWORK_PORT=8787
EXPOSE 8787

# Default: API + dashboard. Override with `desk` or `network` etc.
CMD ["python3", "-m", "sports_network.api.server"]
