import { createWriteStream, readFileSync } from "node:fs";
import { once } from "node:events";
import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite/vector";

const [snapshotPath, wasmPath, fsBundlePath, outputPath] = process.argv.slice(2);
if (!snapshotPath || !wasmPath || !fsBundlePath || !outputPath) {
  throw new Error("usage: export.mjs SNAPSHOT WASM FS_BUNDLE OUTPUT");
}

const snapshot = new Blob([readFileSync(snapshotPath)], { type: "application/gzip" });
const wasm = await WebAssembly.compile(readFileSync(wasmPath));
const fsBundle = new Blob([readFileSync(fsBundlePath)]);
const database = await PGlite.create({
  loadDataDir: snapshot,
  pgliteWasmModule: wasm,
  fsBundle,
  extensions: { vector },
  relaxedDurability: true,
});
const output = createWriteStream(outputPath, { flags: "wx", mode: 0o600 });

function encoded(value) {
  if (typeof value === "bigint") return { $bigint: value.toString() };
  if (value instanceof Float32Array || value instanceof Float64Array) return { $vector: Array.from(value) };
  if (value instanceof Uint8Array) return { $bytes: Buffer.from(value).toString("base64") };
  if (value instanceof Date) return { $date: value.toISOString() };
  return value;
}

async function write(value) {
  const line = JSON.stringify(value, (_key, item) => encoded(item)) + "\n";
  if (!output.write(line)) await once(output, "drain");
}

try {
  const tables = await database.query(`
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    ORDER BY table_name
  `);
  await write({ type: "manifest", format: 1, tables: tables.rows.map((row) => row.table_name) });
  for (const { table_name: table } of tables.rows) {
    const columns = await database.query(
      `SELECT column_name, data_type, udt_name FROM information_schema.columns WHERE table_schema='public' AND table_name=$1 ORDER BY ordinal_position`,
      [table],
    );
    const primary = await database.query(
      `SELECT a.attname AS column_name
       FROM pg_index i
       JOIN pg_attribute a ON a.attrelid=i.indrelid AND a.attnum=ANY(i.indkey)
       WHERE i.indrelid=($1::text)::regclass AND i.indisprimary
       ORDER BY array_position(i.indkey, a.attnum)`,
      [`public.${table}`],
    );
    const quotedTable = `"${table.replaceAll('"', '""')}"`;
    const order = primary.rows.length
      ? ` ORDER BY ${primary.rows.map(({ column_name }) => `"${column_name.replaceAll('"', '""')}"`).join(",")}`
      : "";
    await write({
      type: "table",
      table,
      columns: columns.rows,
      primaryKey: primary.rows.map((row) => row.column_name),
    });
    const rows = await database.query(`SELECT * FROM ${quotedTable}${order}`);
    for (const row of rows.rows) await write({ type: "row", table, row });
  }
  await write({ type: "complete" });
} finally {
  await database.close();
  output.end();
  await once(output, "close");
}
