This is stock ai hope you like it

Here is how to get started:

```yaml
services:
  backend:
    image: waffle12/stock-trader-backend:latest
    container_name: stock_backend
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./.documents:/app/documents
      - ./.chroma:/app/chroma
      - ./.models:/app/models
      - ./.data:/app/data
    environment:
      FINNHUB_API_KEY:
      GROQ_API_KEY:
      SIGNUP_CODE:

  frontend:
    image: waffle12/stock-trader-frontend:latest
    container_name: stock-trader-frontend
    restart: unless-stopped
    ports:
      - "8080:80"
    networks:
      - trader

networks:
  trader:
    external: true
```