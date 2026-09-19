/**
 * Generator entrypoint. Read a toolkit catalog, infer its dependencies, write a graph.
 *
 * Approach
 * --------
 * Composio tool schemas are self-describing enough to infer most "precursor" edges
 * without any hardcoded toolkit knowledge:
 *
 *   1. Every tool's `outputParameters` is a JSON Schema whose `$defs` describe the
 *      shape of the data it returns (e.g. `GITHUB_LIST_REPOSITORY_ISSUES` returns an
 *      array of `Issue` objects, each with a `number` field). We flatten those defs
 *      into a field index: fieldName -> [{ tool, defTitle }, ...].
 *   2. Every tool's `inputParameters.required` lists the fields it needs to run. We
 *      scan those for "foreign key"-shaped names (`issue_number`, `pull_number`,
 *      `gist_id`, `sha`, ...), split off the identifier suffix (`number`, `id`,
 *      `sha`, ...) and the resource "stem" (`issue`, `pull`, `gist`, ...).
 *   3. We look up candidate producers in the field index by the identifier field,
 *      then rank them by how well the stem (+ the param's own description) matches
 *      the producing type's name, preferring read-only tools (list/get/search) as
 *      the natural precursor step.
 *   4. Anything left unresolved by the heuristic (rare — a handful of parameters
 *      whose name gives no hint) is optionally handed to an LLM, batched, to pick a
 *      producer from the shortlist of tools that share its resource tag. This step
 *      is best-effort: if no API credentials are configured, or the call fails, we
 *      just skip it and keep the deterministic graph.
 *
 * This is intentionally toolkit-agnostic: nothing below is GitHub-specific, it all
 * flows from the shapes of the schemas in whatever catalog is passed in.
 *
 * How we run it:
 *   - The path to a toolkit's catalog JSON is passed as a CLI ARGUMENT, e.g.
 *     `node --import tsx src/generate.ts path/to/catalog.json`.
 *   - Writes `dependency_graph.json` in the working directory.
 *   - For LLM access, the OpenAI SDK reads OPENAI_API_KEY / OPENAI_BASE_URL from the
 *     environment. Use an OpenRouter-style model id such as `openai/gpt-4o`.
 */
import { readFileSync, writeFileSync } from "fs";

type JSONSchema = Record<string, any>;
type Tool = Record<string, any>;

interface Node {
  id: string;
  service?: string;
}
interface Edge {
  from: string;
  to: string;
  label?: string;
}
interface Graph {
  nodes: Node[];
  edges: Edge[];
}

const CATALOG_PATH = process.argv.length > 2 ? process.argv[process.argv.length - 1] : undefined;
const OUT_PATH = "dependency_graph.json";

// ---------------------------------------------------------------------------
// Catalog loading
// ---------------------------------------------------------------------------

function loadCatalog(): Tool[] {
  if (!CATALOG_PATH) {
    throw new Error("pass the toolkit catalog path as the first argument");
  }
  const data = JSON.parse(readFileSync(CATALOG_PATH, "utf-8"));
  return Array.isArray(data) ? data : (data.tools ?? data.items ?? []);
}

function slugOf(tool: Tool): string | undefined {
  return tool.slug ?? tool.name ?? tool.function?.name;
}

// ---------------------------------------------------------------------------
// Small text helpers
// ---------------------------------------------------------------------------

/** Split "PullRequestNumber", "pull_request_number", "pull-request" into tokens. */
function tokenize(s: string | undefined | null): string[] {
  if (!s) return [];
  const withSpaces = s
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2") // e.g. "APull" -> "A Pull", "XMLParser" -> "XML Parser"
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2") // e.g. "getPull" -> "get Pull"
    .replace(/[_\-./]+/g, " ")
    .toLowerCase();
  return withSpaces
    .split(/\s+/)
    .map((t) => t.trim())
    .filter((t) => t.length > 1 && !STOPWORDS.has(t));
}

const STOPWORDS = new Set([
  "the",
  "a",
  "an",
  "of",
  "to",
  "for",
  "this",
  "that",
  "is",
  "are",
  "or",
  "and",
  "response",
  "request",
  "wrapper",
  "object",
  "data",
  "id",
  "identifier",
  "identifying",
  "unique",
  // identifier-suffix words are structural, not topical - they'd otherwise make
  // every "*_number"/"*_id" param spuriously overlap every other one.
  "number",
  "sha",
  "key",
  "ref",
  "node",
  "slug",
  "login",
]);

/** Very small English singularizer: "issues" -> "issue", "labels" -> "label". */
function singular(word: string): string {
  if (word.endsWith("ies") && word.length > 3) return word.slice(0, -3) + "y";
  if (word.endsWith("ses")) return word.slice(0, -2);
  if (word.endsWith("s") && !word.endsWith("ss") && word.length > 3) return word.slice(0, -1);
  return word;
}

function tokenSet(s: string | undefined | null): Set<string> {
  return new Set(tokenize(s).map(singular));
}


// ---------------------------------------------------------------------------
// Output schema flattening
// ---------------------------------------------------------------------------

interface FieldOrigin {
  toolSlug: string;
  defTitle: string;
  defTokens: Set<string>;
  isArrayItem: boolean;
}

/** fieldName (lowercase) -> producers that expose that field somewhere in their output */
type FieldIndex = Map<string, FieldOrigin[]>;

function resolveRef(ref: string, defs: Record<string, JSONSchema>): JSONSchema | undefined {
  const name = ref.split("/").pop();
  return name ? defs[name] : undefined;
}

/**
 * Walk a tool's outputParameters schema and record every leaf field name it can
 * produce, tagged with the $defs title it came from (so we can later match that
 * title against a parameter's resource "stem", e.g. "Issue" <-> "issue_number").
 */
function flattenOutputFields(toolSlug: string, outputSchema: JSONSchema | undefined, fieldIndex: FieldIndex) {
  if (!outputSchema || typeof outputSchema !== "object") return;
  const defs: Record<string, JSONSchema> = outputSchema.$defs ?? outputSchema.definitions ?? {};

  const visited = new Set<string>();

  function walk(schema: JSONSchema | undefined, defTitle: string | undefined, isArrayItem: boolean, depth: number) {
    if (!schema || depth > 6) return;
    if (schema.$ref) {
      const key = `${schema.$ref}|${isArrayItem}`;
      if (visited.has(key)) return;
      visited.add(key);
      const target = resolveRef(schema.$ref, defs);
      if (target) walk(target, target.title ?? defTitle, isArrayItem, depth + 1);
      return;
    }
    if (schema.type === "array" && schema.items) {
      walk(schema.items, defTitle, true, depth + 1);
      return;
    }
    if (schema.type === "object" || schema.properties) {
      const title = schema.title ?? defTitle;
      const titleTokens = tokenSet(title);
      for (const [propName, propSchemaRaw] of Object.entries<JSONSchema>(schema.properties ?? {})) {
        const propSchema = propSchemaRaw as JSONSchema;
        if (propSchema.$ref || propSchema.type === "object" || propSchema.type === "array") {
          walk(propSchema, title, isArrayItem, depth + 1);
        } else {
          // leaf field
          const key = propName.toLowerCase();
          const arr = fieldIndex.get(key) ?? [];
          arr.push({ toolSlug, defTitle: title ?? "", defTokens: titleTokens, isArrayItem });
          fieldIndex.set(key, arr);
        }
        // some schemas nest object shape under anyOf/oneOf; handle shallowly
        for (const branch of propSchema.anyOf ?? propSchema.oneOf ?? []) {
          walk(branch, title, isArrayItem, depth + 1);
        }
      }
      return;
    }
  }

  // Entry point: the actual payload usually lives under outputSchema.properties.data
  const dataProp = outputSchema.properties?.data;
  if (dataProp) {
    walk(dataProp, dataProp.title, false, 0);
  } else {
    walk(outputSchema, outputSchema.title, false, 0);
  }
}

// ---------------------------------------------------------------------------
// Input parameter -> identifier detection
// ---------------------------------------------------------------------------

const ID_SUFFIXES: Array<{ suffix: string; field: string }> = [
  { suffix: "_node_id", field: "node_id" },
  { suffix: "_number", field: "number" },
  { suffix: "_id", field: "id" },
  { suffix: "_sha", field: "sha" },
  { suffix: "_key", field: "key" },
  { suffix: "_ref", field: "ref" },
  { suffix: "_slug", field: "slug" },
  { suffix: "_login", field: "login" },
];
const BARE_ID_FIELDS = new Set(["id", "number", "sha", "sha1", "ref", "node_id", "key", "slug"]);

// Parameters that are almost always supplied directly by the caller (a repo
// coordinate, pagination, free text) rather than looked up from another tool.
const SKIP_PARAMS = new Set([
  "owner",
  "repo",
  "org",
  "username",
  "page",
  "per_page",
  "perpage",
  "sort",
  "direction",
  "state",
  "type",
  "since",
  "q",
  "query",
  "body",
  "title",
  "message",
  "content",
  "path",
  "name",
  "visibility",
]);

interface IdParam {
  name: string;
  description: string;
  stem: string; // resource hint, e.g. "issue" from "issue_number"
  field: string; // "number" | "id" | "sha" | ...
}

function detectIdParam(paramName: string, description: string): IdParam | undefined {
  const lower = paramName.toLowerCase();
  if (SKIP_PARAMS.has(lower)) return undefined;

  for (const { suffix, field } of ID_SUFFIXES) {
    if (lower.endsWith(suffix) && lower.length > suffix.length) {
      const stem = lower.slice(0, -suffix.length);
      return { name: paramName, description, stem, field };
    }
  }
  if (BARE_ID_FIELDS.has(lower)) {
    return { name: paramName, description, stem: "", field: lower === "sha1" ? "sha" : lower };
  }
  return undefined;
}

// ---------------------------------------------------------------------------
// Tags / service grouping
// ---------------------------------------------------------------------------

const NON_RESOURCE_TAGS = new Set([
  "openWorldHint",
  "mcpIgnore",
  "readOnlyHint",
  "idempotentHint",
  "updateHint",
  "destructiveHint",
  "createHint",
  "important",
  "deprecated",
  "GraphQL",
]);

function serviceOf(tool: Tool, slug: string): string {
  const tags: string[] = Array.isArray(tool.tags) ? tool.tags : [];
  const resourceTag = tags.find((t) => !NON_RESOURCE_TAGS.has(t));
  if (resourceTag) return resourceTag.toLowerCase();
  // fallback: second underscore-delimited token, e.g. GITHUB_LIST_ISSUES -> "list"
  const parts = slug.split("_");
  return parts.length > 1 ? parts[1].toLowerCase() : "general";
}

function isReadOnly(tool: Tool): boolean {
  const tags: string[] = Array.isArray(tool.tags) ? tool.tags : [];
  if (tags.includes("readOnlyHint")) return true;
  const slug = String(slugOf(tool) ?? "").toUpperCase();
  return /_(LIST|GET|SEARCH|FIND)_/.test(`_${slug}_`) || /^(LIST|GET|SEARCH|FIND)_/.test(slug);
}

// ---------------------------------------------------------------------------
// Core inference
// ---------------------------------------------------------------------------

interface UnresolvedParam {
  consumerSlug: string;
  param: IdParam;
}

interface Scored {
  c: FieldOrigin;
  score: number;
  groupKey: string;
}

function inferEdges(tools: Tool[]): { edges: Edge[]; unresolved: UnresolvedParam[] } {
  const fieldIndex: FieldIndex = new Map();
  for (const tool of tools) {
    const slug = slugOf(tool);
    if (!slug) continue;
    flattenOutputFields(slug, tool.outputParameters, fieldIndex);
  }

  const readOnlyBySlug = new Map<string, boolean>();
  const requiredParamsBySlug = new Map<string, Set<string>>();
  for (const tool of tools) {
    const slug = slugOf(tool);
    if (!slug) continue;
    readOnlyBySlug.set(slug, isReadOnly(tool));
    requiredParamsBySlug.set(slug, new Set<string>(tool.inputParameters?.required ?? []));
  }

  const edgeKeySeen = new Set<string>();
  const edges: Edge[] = [];
  const unresolved: UnresolvedParam[] = [];

  for (const tool of tools) {
    const consumerSlug = slugOf(tool);
    if (!consumerSlug) continue;
    const required: string[] = tool.inputParameters?.required ?? [];
    const props: Record<string, JSONSchema> = tool.inputParameters?.properties ?? {};

    for (const paramName of required) {
      const description = String(props[paramName]?.description ?? "");
      const idParam = detectIdParam(paramName, description);
      if (!idParam) continue;

      const candidates: FieldOrigin[] = (fieldIndex.get(idParam.field) ?? []).filter(
        (c) => c.toolSlug !== consumerSlug,
      );
      if (candidates.length === 0) {
        unresolved.push({ consumerSlug, param: idParam });
        continue;
      }

      // Topical tokens (the resource stem + the param's own name) are the strong
      // signal; the free-text description is a weaker, secondary one (it can
      // mention multiple resources, e.g. "issue or pull request").
      const primaryTokens = new Set<string>([...tokenSet(idParam.stem), ...tokenSet(idParam.name)]);
      const descTokens = tokenSet(description);

      const scored: Scored[] = candidates.map((c) => {
        const primaryMatch = [...primaryTokens].filter((t) => c.defTokens.has(t));
        const descMatch = [...descTokens].filter((t) => c.defTokens.has(t) && !primaryTokens.has(t));
        return {
          c,
          score: primaryMatch.length * 2 + descMatch.length,
          groupKey: primaryMatch.length > 0 ? primaryMatch.sort().join(",") : descMatch.sort().join(","),
        };
      });

      const distinctDefTitles = new Set(candidates.map((c) => c.defTitle));
      let chosen: Scored[] = scored.filter((s) => s.score > 0);
      let confident = chosen.length > 0;
      if (chosen.length === 0) {
        if (distinctDefTitles.size === 1) {
          // Unambiguous even without a token match (only one kind of thing has this field).
          chosen = scored.map((s) => ({ ...s, groupKey: "unambiguous" }));
          confident = true;
        } else {
          // A bare field (e.g. plain "id") with no textual hint and several unrelated
          // producer types is too ambiguous to guess at without noise. Leave it to the
          // optional LLM pass instead of emitting a low-confidence, likely-wrong edge.
          unresolved.push({ consumerSlug, param: idParam });
          continue;
        }
      }

      const rank = (s: Scored) => {
        const ro = readOnlyBySlug.get(s.c.toolSlug) ? 1 : 0;
        return { ro, score: s.score };
      };
      const byRank = (a: Scored, b: Scored) => {
        const ra = rank(a);
        const rb = rank(b);
        if (rb.ro !== ra.ro) return rb.ro - ra.ro;
        if (rb.score !== ra.score) return rb.score - ra.score;
        return a.c.toolSlug.localeCompare(b.c.toolSlug);
      };

      // Diversity-aware selection: when a param plausibly matches several distinct
      // resource types (e.g. an id that is both an Issue's and a PullRequest's),
      // keep the best producer(s) from each matched group rather than letting one
      // group's alphabetically-early tools crowd out the others.
      const groups = new Map<string, Scored[]>();
      for (const s of chosen) {
        const arr = groups.get(s.groupKey) ?? [];
        arr.push(s);
        groups.set(s.groupKey, arr);
      }

      const perGroupCap = confident ? 2 : 3;
      const overallCap = 4;
      const picked: Scored[] = [];
      for (const arr of groups.values()) {
        arr.sort(byRank);
        const seenProducers = new Set<string>();
        for (const s of arr) {
          if (picked.filter((p) => p.groupKey === s.groupKey).length >= perGroupCap) break;
          if (seenProducers.has(s.c.toolSlug)) continue;
          seenProducers.add(s.c.toolSlug);
          picked.push(s);
        }
      }
      picked.sort(byRank);

      const seenProducers = new Set<string>();
      let added = 0;
      for (const { c } of picked) {
        if (added >= overallCap) break;
        if (seenProducers.has(c.toolSlug)) continue;
        seenProducers.add(c.toolSlug);
        const key = `${c.toolSlug}->${consumerSlug}:${paramName}`;
        if (edgeKeySeen.has(key)) continue;
        edgeKeySeen.add(key);
        edges.push({ from: c.toolSlug, to: consumerSlug, label: paramName });
        added++;
      }
    }
  }

  return { edges, unresolved };
}

// ---------------------------------------------------------------------------
// Optional LLM fallback for parameters the heuristic couldn't resolve at all
// ---------------------------------------------------------------------------

async function llmResolveUnresolved(
  tools: Tool[],
  unresolved: UnresolvedParam[],
  existingEdgeKeys: Set<string>,
): Promise<Edge[]> {
  if (unresolved.length === 0) return [];
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return [];

  let OpenAI: any;
  try {
    ({ default: OpenAI } = await import("openai"));
  } catch {
    return [];
  }

  const client = new OpenAI({ apiKey, baseURL: process.env.OPENAI_BASE_URL });
  const model = process.env.LITMUS_MODEL || "openai/gpt-4o";

  const bySlug = new Map(tools.map((t) => [slugOf(t), t] as const));
  const readOnlySlugs = tools.filter(isReadOnly).map(slugOf).filter(Boolean) as string[];

  const extraEdges: Edge[] = [];
  const BATCH = 8;
  // De-duplicate identical (consumerSlug, param.name) pairs across tools first.
  const uniq = new Map<string, UnresolvedParam>();
  for (const u of unresolved) uniq.set(`${u.consumerSlug}:${u.param.name}`, u);
  const items = [...uniq.values()].slice(0, 60); // keep token usage bounded

  for (let i = 0; i < items.length; i += BATCH) {
    const batch = items.slice(i, i + BATCH);
    const prompt = batch
      .map((u, idx) => {
        const tool = bySlug.get(u.consumerSlug);
        return `${idx}. consumer_tool=${u.consumerSlug}\n   consumer_description=${String(tool?.description ?? "").slice(0, 200)}\n   param=${u.param.name}\n   param_description=${u.param.description.slice(0, 200)}`;
      })
      .join("\n");

    const candidateList = readOnlySlugs.slice(0, 400).join(", ");

    try {
      const resp = await client.chat.completions.create({
        model,
        messages: [
          {
            role: "system",
            content:
              "You map API tool parameters to the read-only/list/get tool that would supply that value as a precursor call. " +
              "Given a numbered list of (consumer tool, required parameter), pick the single best producer tool slug from the CANDIDATES list for each, " +
              'or null if none clearly supplies it. Respond ONLY with strict JSON: {"answers": [{"index": 0, "producer": "SLUG_OR_NULL"}, ...]}.',
          },
          {
            role: "user",
            content: `CANDIDATES: ${candidateList}\n\nITEMS:\n${prompt}`,
          },
        ],
        temperature: 0,
      });

      const text = resp.choices?.[0]?.message?.content ?? "{}";
      const jsonMatch = text.match(/\{[\s\S]*\}/);
      const parsed = jsonMatch ? JSON.parse(jsonMatch[0]) : { answers: [] };
      for (const ans of parsed.answers ?? []) {
        const item = batch[ans.index];
        if (!item || !ans.producer || ans.producer === "null") continue;
        if (!bySlug.has(ans.producer)) continue;
        if (ans.producer === item.consumerSlug) continue;
        const key = `${ans.producer}->${item.consumerSlug}:${item.param.name}`;
        if (existingEdgeKeys.has(key)) continue;
        existingEdgeKeys.add(key);
        extraEdges.push({ from: ans.producer, to: item.consumerSlug, label: item.param.name });
      }
    } catch (err) {
      console.error("LLM fallback batch failed, skipping:", (err as Error).message);
    }
  }

  return extraEdges;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

async function generate(tools: Tool[]): Promise<Graph> {
  const nodes: Node[] = tools
    .map((t) => {
      const id = slugOf(t);
      if (!id) return undefined;
      return { id, service: serviceOf(t, id) };
    })
    .filter((n): n is Node => !!n);

  const { edges, unresolved } = inferEdges(tools);

  const edgeKeySeen = new Set(edges.map((e) => `${e.from}->${e.to}:${e.label}`));
  const llmEdges = await llmResolveUnresolved(tools, unresolved, edgeKeySeen);

  return { nodes, edges: [...edges, ...llmEdges] };
}

async function main() {
  const tools = loadCatalog();
  const graph = await generate(tools);
  writeFileSync(OUT_PATH, JSON.stringify(graph, null, 2), "utf-8");
  console.error(`wrote ${graph.nodes.length} nodes, ${graph.edges.length} edges to ${OUT_PATH}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
