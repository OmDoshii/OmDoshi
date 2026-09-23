// functions/api/chat.js
//
// Cloudflare Pages Function -> POST /api/chat
//
// Replaces the old Streamlit + Chroma + HuggingFaceEmbeddings RAG pipeline.
// The portfolio page is short enough (a few thousand words) that we don't
// need a vector store at all: we just read the deployed index.html at
// request time via the Pages ASSETS binding, strip it to plain text, and
// hand the whole thing to Groq as context. This is what removes the RAM
// pressure that was crashing the Streamlit app - there's no embeddings
// model loaded in memory, nothing cached across requests, nothing to leak.

const GROQ_MODEL = "openai/gpt-oss-120b";
const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

const MAX_MESSAGE_LEN = 800;      // chars, per user message
const MAX_HISTORY_TURNS = 12;     // user+assistant messages kept from client
const MAX_TOKENS = 500;

// In-memory per-isolate cache of the extracted portfolio text. Cloudflare
// Workers reuse an isolate across multiple requests when there's traffic,
// so this avoids re-fetching/re-stripping index.html on every single
// message. It's just an optimization - a cold isolate simply re-fetches.
let cachedContext = null;
let cachedAt = 0;
const CONTEXT_TTL_MS = 5 * 60 * 1000; // 5 min - picks up redeploys reasonably fast

const SYSTEM_PROMPT = `You are the AI assistant embedded on Om Doshi's portfolio site, answering
questions from recruiters, hiring managers, and site visitors on Om's behalf.

STRICT GROUNDING RULES - follow these exactly:
1. Only answer using the portfolio content provided below. Never invent, assume, or guess facts
   about Om (skills, dates, companies, degrees, projects, etc.) that are not present in it.
2. If a question asks about something not covered in the content:
   - If it's a reasonable question about Om himself that just isn't on the site (e.g. salary
     expectations, notice period, visa status), say it isn't covered here and suggest reaching
     out to Om directly via the contact links on the site.
   - If it's unrelated to Om entirely (general knowledge, other people, anything with no
     connection to his portfolio), say this assistant only answers questions about Om Doshi's
     background and portfolio, and doesn't have information on that topic. Don't speculate.
3. If you're unsure whether the content fully answers the question, say what you do know and
   flag what you don't, rather than filling the gap.
4. Keep answers concise (2-5 sentences unless genuinely more detail is needed), professional,
   and in third person about Om (e.g. "Om worked on...").
5. Format plainly - no markdown headers, this renders in a small chat panel.

Portfolio content:
---
{context}
---`;

function corsHeaders(origin) {
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
  };
}

// Icon-only links (mailto/tel/social) have no visible text, so htmlToText's
// tag-stripping below would otherwise erase them entirely — the AI would
// have no way to know Om's email or socials even though they're on the page.
// Pull them out as explicit plain-text lines first.
function extractContactLinks(html) {
  const patterns = [
    { label: "Email", re: /href=["']mailto:([^"'?]+)/i },
    { label: "Phone", re: /href=["']tel:([^"']+)/i },
    { label: "LinkedIn", re: /href=["'](https?:\/\/(?:www\.)?linkedin\.com[^"']*)["']/i },
    { label: "GitHub", re: /href=["'](https?:\/\/(?:www\.)?github\.com[^"']*)["']/i },
    { label: "LeetCode", re: /href=["'](https?:\/\/(?:www\.)?leetcode\.com[^"']*)["']/i },
    { label: "Instagram", re: /href=["'](https?:\/\/(?:www\.)?instagram\.com[^"']*)["']/i },
    { label: "TUF", re: /href=["'](https?:\/\/(?:www\.)?takeuforward\.org[^"']*)["']/i },
    { label: "Resume", re: /href=["'](https?:\/\/(?:www\.)?drive\.google\.com[^"']*)["']/i },
  ];
  const lines = [];
  for (const { label, re } of patterns) {
    const m = html.match(re);
    if (m) lines.push(`${label}: ${m[1]}`);
  }
  return lines.length ? `Om's contact links (use these when asked how to reach him):\n${lines.join("\n")}\n` : "";
}

function htmlToText(html) {
  return html
    // drop non-content elements entirely
    .replace(/<script[\s\S]*?<\/script>/gi, " ")
    .replace(/<style[\s\S]*?<\/style>/gi, " ")
    .replace(/<svg[\s\S]*?<\/svg>/gi, " ")
    // keep some structure: turn block-ish tag boundaries into line breaks
    .replace(/<\/(h1|h2|h3|h4|li|p|div|section|footer|header)>/gi, "\n")
    .replace(/<(br|hr)\s*\/?>/gi, "\n")
    // strip all remaining tags
    .replace(/<[^>]+>/g, " ")
    // decode the handful of entities we actually use
    .replace(/&nbsp;/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&#39;|&rsquo;/g, "'")
    .replace(/&mdash;/g, "-")
    // collapse whitespace
    .replace(/[ \t]+/g, " ")
    .replace(/\n\s*\n+/g, "\n")
    .trim();
}

async function getPortfolioContext(env, request) {
  const now = Date.now();
  if (cachedContext && now - cachedAt < CONTEXT_TTL_MS) {
    return cachedContext;
  }

  const origin = new URL(request.url).origin;
  const assetRes = await env.ASSETS.fetch(new URL("/", origin));
  const html = await assetRes.text();
  const contactLines = extractContactLinks(html);
  const text = htmlToText(html);

  // Hard cap so a single request can never balloon token usage even if the
  // page grows a lot in the future.
  cachedContext = (contactLines + text).slice(0, 12000);
  cachedAt = now;
  return cachedContext;
}

function badRequest(message, origin) {
  return new Response(JSON.stringify({ error: message }), {
    status: 400,
    headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
  });
}

export async function onRequestGet({ request, env }) {
  const origin = new URL(request.url).origin;
  return new Response(
    JSON.stringify({
      status: "ok",
      message: "This endpoint accepts POST requests from the chat widget.",
      groq_key_configured: Boolean(env.GROQ_API_KEY),
    }),
    {
      status: 200,
      headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
    }
  );
}

export async function onRequestOptions({ request }) {
  const origin = new URL(request.url).origin;
  return new Response(null, { status: 204, headers: corsHeaders(origin) });
}

export async function onRequestPost({ request, env }) {
  const origin = new URL(request.url).origin;

  if (!env.GROQ_API_KEY) {
    return new Response(JSON.stringify({ error: "Server not configured." }), {
      status: 500,
      headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
    });
  }

  let body;
  try {
    body = await request.json();
  } catch {
    return badRequest("Invalid JSON body.", origin);
  }

  const question = typeof body.message === "string" ? body.message.trim() : "";
  const history = Array.isArray(body.history) ? body.history : [];

  if (!question) return badRequest("Message is required.", origin);
  if (question.length > MAX_MESSAGE_LEN) {
    return badRequest(`Message too long (max ${MAX_MESSAGE_LEN} characters).`, origin);
  }

  // Sanitize/trim client-supplied history: only role+content, only the last
  // N turns, each capped in length. Never trust the shape of client input.
  const safeHistory = history
    .filter(
      (m) =>
        m &&
        (m.role === "user" || m.role === "assistant") &&
        typeof m.content === "string"
    )
    .slice(-MAX_HISTORY_TURNS)
    .map((m) => ({
      role: m.role,
      content: m.content.slice(0, MAX_MESSAGE_LEN),
    }));

  let context;
  try {
    context = await getPortfolioContext(env, request);
  } catch (err) {
    return new Response(
      JSON.stringify({ error: "Could not load portfolio content." }),
      {
        status: 500,
        headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
      }
    );
  }

  const messages = [
    { role: "system", content: SYSTEM_PROMPT.replace("{context}", context) },
    ...safeHistory,
    { role: "user", content: question },
  ];

  let groqRes;
  try {
    groqRes = await fetch(GROQ_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages,
        temperature: 0.3,
        max_tokens: MAX_TOKENS,
      }),
    });
  } catch (err) {
    return new Response(JSON.stringify({ error: "Upstream request failed." }), {
      status: 502,
      headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
    });
  }

  if (!groqRes.ok) {
    const detail = await groqRes.text().catch(() => "");
    console.error("Groq error", groqRes.status, detail);
    return new Response(JSON.stringify({ error: "Assistant is unavailable right now." }), {
      status: 502,
      headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
    });
  }

  const data = await groqRes.json();
  const reply = data?.choices?.[0]?.message?.content?.trim() || "Sorry, I couldn't generate a response.";

  return new Response(JSON.stringify({ reply }), {
    status: 200,
    headers: { "Content-Type": "application/json", ...corsHeaders(origin) },
  });
}
