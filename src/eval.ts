/**
 * Algorithmic eval for dependency_graph.json (no LLM / no network).
 *
 * Usage:
 *   npm run eval                  # reads existing dependency_graph.json
 *   npm run eval -- --regen       # regenerate first via generate.ts
 *
 * Prints % coverage: tools placed, consumers covered, id-params covered,
 * structurally-valid edges, plus README gold + hygiene checks.
 */
import { readFileSync, existsSync } from "fs";
import { execFileSync } from "child_process";

const CATALOG = "github_catalog.json";
const OUT = "dependency_graph.json";

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

const USERISH = new Set([
  "owner",
  "repo",
  "org",
  "username",
  "body",
  "title",
  "message",
  "content",
  "name",
  "page",
  "per_page",
  "perpage",
  "query",
  "q",
  "sort",
  "direction",
  "state",
  "type",
  "since",
  "path",
  "visibility",
]);

const ID_SUFFIXES = ["_node_id", "_number", "_id", "_sha", "_key", "_ref", "_slug", "_login"];
const BARE_ID = new Set(["id", "number", "sha", "sha1", "ref", "node_id", "key", "slug"]);

const README_GOLD: Array<{ from: string; to: string; label: string }> = [
  {
    from: "GITHUB_LIST_REPOSITORY_ISSUES",
    to: "GITHUB_CREATE_AN_ISSUE_COMMENT",
    label: "issue_number",
  },
  {
    from: "GITHUB_LIST_PULL_REQUESTS",
    to: "GITHUB_MERGE_A_PULL_REQUEST",
    label: "pull_number",
  },
];

function slugOf(tool: Tool): string | undefined {
  return tool.slug ?? tool.name ?? tool.function?.name;
}

function isIdParam(name: string): boolean {
  const lower = name.toLowerCase();
  if (USERISH.has(lower)) return false;
  if (BARE_ID.has(lower)) return true;
  return ID_SUFFIXES.some((s) => lower.endsWith(s) && lower.length > s.length);
}

function loadTools(): Tool[] {
  if (!existsSync(CATALOG)) throw new Error(`missing ${CATALOG}`);
  const raw = JSON.parse(readFileSync(CATALOG, "utf-8"));
  return Array.isArray(raw) ? raw : (raw.tools ?? raw.items ?? []);
}

function loadGraph(): Graph {
  if (!existsSync(OUT)) throw new Error(`missing ${OUT} — run the generator first or pass --regen`);
  return JSON.parse(readFileSync(OUT, "utf-8"));
}

function pct(num: number, den: number): number {
  if (!den) return 0;
  return Number(((100 * num) / den).toFixed(1));
}

function main() {
  const regen = process.argv.includes("--regen");
  if (regen) {
    console.error("regenerating graph…");
    execFileSync("node", ["--import", "tsx", "src/generate.ts", CATALOG], { stdio: "inherit" });
  }

  const tools = loadTools();
  const graph = loadGraph();
  const nodes = graph.nodes ?? [];
  const edges = graph.edges ?? [];

  const bySlug = new Map<string, Tool>();
  for (const t of tools) {
    const s = slugOf(t);
    if (s) bySlug.set(s, t);
  }
  const catalogSlugs = [...bySlug.keys()];
  const nodeIds = new Set(nodes.map((n) => n.id));

  // ---- per-tool required id params ----
  type Need = { slug: string; param: string };
  const needs: Need[] = [];
  for (const slug of catalogSlugs) {
    const t = bySlug.get(slug)!;
    const req: string[] = t.inputParameters?.required ?? [];
    for (const p of req) {
      if (isIdParam(p)) needs.push({ slug, param: p });
    }
  }
  const consumersNeedingId = new Set(needs.map((n) => n.slug));

  // edge index: consumer|param -> edges
  const edgesByNeed = new Map<string, Edge[]>();
  for (const e of edges) {
    if (!e.label) continue;
    const key = `${e.to}|${e.label}`;
    const arr = edgesByNeed.get(key) ?? [];
    arr.push(e);
    edgesByNeed.set(key, arr);
  }

  // ---- coverage counts ----
  const toolsPlaced = catalogSlugs.filter((s) => nodeIds.has(s)).length;

  let paramsCovered = 0;
  const uncoveredParams: string[] = [];
  for (const n of needs) {
    const hits = edgesByNeed.get(`${n.slug}|${n.param}`) ?? [];
    if (hits.length > 0) paramsCovered++;
    else uncoveredParams.push(`${n.slug}.${n.param}`);
  }

  let consumersCovered = 0;
  for (const slug of consumersNeedingId) {
    const mine = needs.filter((n) => n.slug === slug);
    const anyHit = mine.some((n) => (edgesByNeed.get(`${n.slug}|${n.param}`) ?? []).length > 0);
    if (anyHit) consumersCovered++;
  }

  // structurally "correct" edges (algorithmic, not semantic gold)
  let edgesValid = 0;
  let chicken = 0;
  let userish = 0;
  let badEndpoint = 0;
  let labelNotRequired = 0;
  for (const e of edges) {
    const prod = bySlug.get(e.from);
    const cons = bySlug.get(e.to);
    if (!prod || !cons) {
      badEndpoint++;
      continue;
    }
    const consReq: string[] = cons.inputParameters?.required ?? [];
    const prodReq: string[] = prod.inputParameters?.required ?? [];
    const label = String(e.label ?? "");
    if (!label || USERISH.has(label.toLowerCase())) {
      userish++;
      continue;
    }
    if (!consReq.includes(label)) {
      labelNotRequired++;
      continue;
    }
    if (prodReq.includes(label)) {
      chicken++;
      continue;
    }
    edgesValid++;
  }

  // README gold
  let goldHits = 0;
  const goldLines: string[] = [];
  for (const g of README_GOLD) {
    const hit = edges.some((e) => e.from === g.from && e.to === g.to && e.label === g.label);
    if (hit) goldHits++;
    goldLines.push(`${hit ? "✓" : "✗"} ${g.from} --${g.label}--> ${g.to}`);
  }

  // hygiene
  const selfLoops = edges.filter((e) => e.from === e.to).length;
  const dupKeys = edges.map((e) => `${e.from}->${e.to}:${e.label}`);
  const dups = dupKeys.length - new Set(dupKeys).size;
  const withService = nodes.filter((n) => n.service).length;
  const listishProducers = edges.filter(
    (e) => e.from.includes("LIST") || e.from.includes("FIND") || e.from.includes("SEARCH"),
  ).length;

  const coverage = {
    tools_in_catalog: catalogSlugs.length,
    tools_placed_as_nodes: toolsPlaced,
    tools_placed_pct: pct(toolsPlaced, catalogSlugs.length),

    tools_needing_id_param: consumersNeedingId.size,
    tools_with_at_least_one_precursor: consumersCovered,
    tools_with_precursor_pct: pct(consumersCovered, consumersNeedingId.size),

    id_params_total: needs.length,
    id_params_with_producer: paramsCovered,
    id_params_covered_pct: pct(paramsCovered, needs.length),

    edges_total: edges.length,
    edges_structurally_valid: edgesValid,
    edges_valid_pct: pct(edgesValid, edges.length),
    edges_rejected: {
      chicken_egg: chicken,
      userish_label: userish,
      label_not_required_on_consumer: labelNotRequired,
      unknown_endpoint: badEndpoint,
    },

    readme_gold_hits: goldHits,
    readme_gold_total: README_GOLD.length,
    readme_gold_pct: pct(goldHits, README_GOLD.length),

    list_find_search_producer_edges: listishProducers,
    list_find_search_producer_pct: pct(listishProducers, edges.length),

    nodes_with_service_pct: pct(withService, nodes.length),
    self_loops: selfLoops,
    duplicate_edges: dups,
  };

  // overall = weighted blend of the main % metrics
  const overall = Number(
    (
      coverage.tools_placed_pct * 0.15 +
      coverage.tools_with_precursor_pct * 0.25 +
      coverage.id_params_covered_pct * 0.25 +
      coverage.edges_valid_pct * 0.2 +
      coverage.readme_gold_pct * 0.15
    ).toFixed(1),
  );

  const report = {
    overall_pct: overall,
    coverage,
    readme_gold: goldLines,
    sample_uncovered_params: uncoveredParams.slice(0, 15),
    uncovered_params_count: uncoveredParams.length,
  };

  console.log(JSON.stringify(report, null, 2));

  // Human-readable % board
  console.error("\n════════ COVERAGE % ════════");
  console.error(
    `Tools placed:           ${coverage.tools_placed_as_nodes}/${coverage.tools_in_catalog}  =  ${coverage.tools_placed_pct}%`,
  );
  console.error(
    `Tools w/ precursor:     ${coverage.tools_with_at_least_one_precursor}/${coverage.tools_needing_id_param}  =  ${coverage.tools_with_precursor_pct}%`,
  );
  console.error(
    `ID-params covered:      ${coverage.id_params_with_producer}/${coverage.id_params_total}  =  ${coverage.id_params_covered_pct}%`,
  );
  console.error(
    `Edges structurally OK:  ${coverage.edges_structurally_valid}/${coverage.edges_total}  =  ${coverage.edges_valid_pct}%`,
  );
  console.error(
    `  rejected → chicken-egg=${chicken}, userish=${userish}, label≠required=${labelNotRequired}, bad-endpoint=${badEndpoint}`,
  );
  console.error(
    `README gold edges:      ${coverage.readme_gold_hits}/${coverage.readme_gold_total}  =  ${coverage.readme_gold_pct}%`,
  );
  console.error(
    `LIST/FIND/SEARCH share: ${listishProducers}/${edges.length}  =  ${coverage.list_find_search_producer_pct}%`,
  );
  console.error(`\nOVERALL (weighted):     ${overall}%`);
  for (const line of goldLines) console.error(`  ${line}`);

  if (overall < 70 || goldHits < README_GOLD.length) process.exit(1);
}

main();
