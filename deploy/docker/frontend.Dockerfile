FROM node:24-bookworm-slim AS build

ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --ignore-scripts --no-audit --no-fund
COPY frontend/ ./
COPY pyproject.toml /build/pyproject.toml
RUN npm run build:demo

FROM nginx:stable-alpine
COPY deploy/docker/nginx-static.conf /etc/nginx/nginx.conf
COPY --from=build /build/frontend/out/ /usr/share/nginx/html/
USER nginx
EXPOSE 8080
ENTRYPOINT ["nginx", "-g", "daemon off;"]
