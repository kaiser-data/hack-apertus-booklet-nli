// Proxies the live demo's API calls (/api/booklets, /api/health, /api/verify) to the Modal
// backend, so the static pages and the API share one origin. The visitor's address travels as
// x-demo-client for the demo's per-person daily limit.
const BACKEND = "https://kaiser-data--booklet-nli-demo-web.us-east.modal.run";

export async function onRequest({ request }) {
  const url = new URL(request.url);
  const headers = new Headers({ "content-type": request.headers.get("content-type") || "application/json" });
  headers.set("x-demo-client", request.headers.get("cf-connecting-ip") || "");
  return fetch(BACKEND + url.pathname + url.search, {
    method: request.method,
    headers,
    body: ["GET", "HEAD"].includes(request.method) ? undefined : await request.text(),
  });
}
