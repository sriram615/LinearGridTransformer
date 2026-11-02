FROM node:22.12.0-alpine3.20 AS build

WORKDIR /app
RUN npm install --global pnpm@10.34.3

COPY frontend/package.json frontend/pnpm-lock.yaml ./
COPY frontend/.figma/ ./.figma/
RUN pnpm install --frozen-lockfile

COPY frontend/ ./
ARG VITE_API_BASE_URL=
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}
RUN pnpm build

FROM nginx:1.27.4-alpine AS runtime
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/ /usr/share/nginx/html/
EXPOSE 80
