#!/usr/bin/env node
// newton-claw MCP stdio server: exposes the Newton Control Server (Jetson Thor) as
// dedicated named tools (`newton__*`) the local model can call cleanly, instead of
// the generic exec/code bridge that small models mangle.
//
// NOTHING IS HARDCODED: valid task ids / robot names are fetched LIVE from the
// control server (GET /envs, /robots) and baked into the tool schemas as enums.
//
// TRANSPORT NOTE: the gateway spawns this MCP server WITHOUT http_proxy/https_proxy
// set (only NEMOCLAW_PROXY_HOST/PORT), and in a network namespace where a DIRECT
// connection to host.openshell.internal:5561 may be firewall-dropped. So we do our
// own transport with raw node:http (not fetch — fetch is proxy-patched/guarded and
// flaky here): TRY THE EGRESS PROXY FIRST (built from NEMOCLAW_PROXY_HOST/PORT),
// then FALL BACK TO DIRECT. Whichever path the serving netns allows wins.
import http from "node:http";
import fs from "node:fs";

const HOST = process.env.NEWTON_CONTROL_HOST || "host.openshell.internal";
const PORT = parseInt(process.env.NEWTON_CONTROL_PORT || "5561", 10);
const DIRECT_ONLY = !!process.env.NEWTON_CONTROL_HOST;   // host-side testing: skip the sandbox proxy
const PROXY_HOST = process.env.NEMOCLAW_PROXY_HOST
  || (process.env.http_proxy||process.env.HTTP_PROXY||"").replace(/^https?:\/\//,"").split(":")[0]
  || "10.200.0.1";
const PROXY_PORT = parseInt(process.env.NEMOCLAW_PROXY_PORT
  || (process.env.http_proxy||process.env.HTTP_PROXY||"").split(":").pop()
  || "3128", 10);

function logErr(o){ try{ fs.appendFileSync("/sandbox/.openclaw/newton-mcp-err.log", new Date().toISOString()+" "+JSON.stringify(o)+"\n"); }catch{} }

// One raw node:http request. useProxy=true → connect to the egress proxy and send
// an absolute-form request (origin-style forward proxy for http URLs).
function once(useProxy, method, path, data){
  return new Promise((resolve, reject)=>{
    const target = `http://${HOST}:${PORT}${path}`;
    const opts = useProxy
      ? { host: PROXY_HOST, port: PROXY_PORT, method, path: target,
          agent: new http.Agent({}),
          headers: { Host: `${HOST}:${PORT}`, "Content-Type":"application/json",
                     ...(data ? { "Content-Length": Buffer.byteLength(data) } : {}) } }
      : { host: HOST, port: PORT, method, path,
          agent: new http.Agent({}),
          headers: { "Content-Type":"application/json",
                     ...(data ? { "Content-Length": Buffer.byteLength(data) } : {}) } };
    const req = http.request(opts, (res)=>{
      let b=""; res.on("data",d=>b+=d);
      res.on("end",()=>{ (res.statusCode>=200 && res.statusCode<500 && b.trim().startsWith("{")) ? resolve(b)
                         : reject(new Error(`HTTP ${res.statusCode} via ${useProxy?"proxy":"direct"}`)); });
    });
    req.setTimeout(8000, ()=>req.destroy(new Error("timeout")));
    req.on("error", reject);
    if (data) req.write(data);
    req.end();
  });
}

async function fetchCS(method, path, body){
  const data = body ? JSON.stringify(body) : null;
  if (DIRECT_ONLY) { try { return await once(false, method, path, data); } catch (e) { return JSON.stringify({error:"unreachable", detail:String(e&&(e.message||e))}); } }
  try { return await once(true, method, path, data); }        // proxy first
  catch (e1) {
    try { return await once(false, method, path, data); }     // then direct
    catch (e2) {
      const info = { error:"unreachable", proxyErr:String(e1&&(e1.message||e1)), directErr:String(e2&&(e2.message||e2)),
                     url:`http://${HOST}:${PORT}${path}`, proxy:`${PROXY_HOST}:${PROXY_PORT}` };
      logErr(info); return JSON.stringify(info);
    }
  }
}

// ---- LIVE catalog (fetched once per process/session; MCP restarts per session) ----
let _cat;
async function catalog(){
  if (_cat) return _cat;
  const getj = async (p)=>{ try{ const o=JSON.parse(await fetchCS("GET",p)); return (o&&o.error)?null:o; }catch{ return null; } };
  const [rob, env] = await Promise.all([getj("/robots"), getj("/envs")]);
  _cat = {
    robots: rob && Array.isArray(rob.robots) ? rob.robots.map(r=>r.name).filter(Boolean) : null,
    tasks:  env && Array.isArray(env.environments) ? env.environments.map(e=>e.id).filter(Boolean) : null,
  };
  return _cat;
}
const S = (description, list)=> (list && list.length) ? { type:"string", description, enum:list } : { type:"string", description };
const NONE = { type:"object", properties:{} };

async function buildTools(){
  const c = await catalog();
  const policyProps = {
    task: S("task id (live list from newton__list_tasks)", c.tasks),
    pretrained:{ type:"boolean", description:"use the reference policy shipped with the robot assets (Go2 / G1 / ANYmal only) instead of the latest one trained on this Thor" },
    num_envs:{ type:"integer", description:"number of robots (default 16 for play, 1024 for eval)" },
  };
  return [
    { name:"list_tasks", description:"List the robot tasks available on this Jetson Thor. ONLY for questions like 'what can I train' / 'how many tasks'. NOT needed before training or playing — the task ids are already in those tools' schemas.", inputSchema:NONE },
    { name:"list_robots", description:"List robots that can be opened in the Newton viewer.", inputSchema:NONE },
    { name:"list_runs", description:"List policies already trained on this Thor (task, run, checkpoint, exported ONNX).", inputSchema:NONE },
    { name:"train_task", description:"START RL TRAINING of a robot in Newton physics (rsl_rl PPO) on the Thor GPU. Use for 'train the Go2', 'train a G1', 'train the robot'. 4096 parallel robots; by default a window on the Thor's monitor shows 36 of them. ONE GPU job at a time: if it answers 'a GPU job is already running', call newton__stop first.",
      inputSchema:{ type:"object", properties:{
        task: S("task id (live list from newton__list_tasks)", c.tasks),
        num_envs:{ type:"integer", description:"parallel robots (default 4096; lower if memory is tight)" },
        max_iterations:{ type:"integer", description:"training iterations (omit for the task default)" },
        seed:{ type:"integer" },
        resume:{ type:"boolean", description:"continue from the latest checkpoint of this task" },
        view:{ type:"string", enum:["gui","web","headless"], description:"where to show it. Default gui = a window on the Thor's monitor. web = browser link. headless = no display (fastest) — use only if the user says headless / no window / in the background." },
        params:{ type:"object", description:"hyperparameters / reward weights, e.g. {learning_rate:5e-4, entropy_coef:0.005, \"reward.action_rate_l2\":-0.02}. Take names and ranges from newton__tuning_guide — never invent them." } },
        required:["task"] } },
    { name:"training_status", description:"Training / job progress: running or finished, iteration, mean reward, mean episode length, steps per second, ETA, per-term rewards, and (after an evaluation) the result metrics. Use for 'how is the training going', 'is it done', 'status'.", inputSchema:NONE },
    { name:"play_policy", description:"Run INFERENCE with a trained policy and show the robot in a window on the Thor's monitor (or view:web for a browser link). Use for 'show me the Go2 walking', 'play the policy', 'run inference'.",
      inputSchema:{ type:"object", properties:{ ...policyProps,
        command:{ type:"array", items:{type:"number"}, description:"walking command [forward m/s, sideways m/s, turn rad/s], e.g. [1.0,0,0]" },
        view:{ type:"string", enum:["gui","web"], description:"gui (default) = window on the Thor's monitor; web = browser link (returns view_url)" } },
        required:["task"] } },
    { name:"eval_policy", description:"Measure a trained policy headless (falls, velocity-tracking error, speed). Takes ~1 minute; then read the numbers with newton__training_status (field result).",
      inputSchema:{ type:"object", properties: policyProps, required:["task"] } },
    { name:"open_robot", description:"Show a robot in the Newton viewer (window on the Thor's monitor by default) (no training, no policy unless the robot entry has one).",
      inputSchema:{ type:"object", properties:{ robot: S("robot name (live list from newton__list_robots)", c.robots), view:{ type:"string", enum:["gui","web"], description:"gui (default) = window on the Thor's monitor; web = browser link (returns view_url)" } }, required:["robot"] } },
    { name:"stop", description:"Stop the running GPU job (training, inference, evaluation or viewer) and free the GPU. Use for 'stop training', 'close it', 'cancel'.", inputSchema:NONE },
    { name:"tuning_guide", description:"Hyperparameter + reward-weight guide (safe ranges, symptom -> fix). Read BEFORE choosing params for a tuned training run.", inputSchema:NONE },
    { name:"device_status", description:"Jetson Thor snapshot: model, free memory, free disk.", inputSchema:NONE },
  ];
}

const VIEW = { gui:"gl", web:"viser", headless:"none" };   // omitted -> server default (gui)
async function call(name, a){
  a = a || {};
  if (name==="list_tasks") return fetchCS("GET","/envs");
  if (name==="list_robots") return fetchCS("GET","/robots");
  if (name==="list_runs") return fetchCS("GET","/runs");
  if (name==="tuning_guide") return fetchCS("GET","/tuning");
  if (name==="device_status") return fetchCS("GET","/device");
  if (name==="stop") return fetchCS("POST","/stop",{});
  if (name==="train_task"){
    const b={task:a.task};
    if (VIEW[a.view]) b.viewer=VIEW[a.view];
    for (const k of ["num_envs","max_iterations","seed"]) if (a[k]!=null) b[k]=a[k];
    if (a.resume) b.resume=true;
    if (a.params && typeof a.params==="object") b.params=a.params;
    return fetchCS("POST","/train",b);
  }
  if (name==="play_policy" || name==="eval_policy"){
    const b={task:a.task};
    if (a.pretrained) b.pretrained=true;
    if (a.num_envs!=null) b.num_envs=a.num_envs;
    if (Array.isArray(a.command)) b.command=a.command;
    if (name==="play_policy" && VIEW[a.view]) b.viewer=VIEW[a.view];
    return fetchCS("POST", name==="play_policy"?"/play":"/eval", b);
  }
  if (name==="open_robot"){ const b={robot:a.robot}; if (VIEW[a.view]) b.viewer=VIEW[a.view]; return fetchCS("POST","/open",b); }
  if (name==="training_status"){
    const pj = (t)=>{ try{ return JSON.parse(t); }catch{ return t; } };
    const [st, lg] = await Promise.all([ fetchCS("GET","/status"), fetchCS("GET","/logs?lines=60") ]);
    const status = pj(st), logObj = pj(lg);
    const text = (logObj && typeof logObj.lines === "string") ? logObj.lines : "";
    const rewards = {};
    for (const l of text.split(/\r?\n/)){ const m=l.match(/Episode_Reward\/(\S+):\s*(-?\d[\d.eE+-]*)/); if(m) rewards[m[1]]=parseFloat(m[2]); }
    return JSON.stringify({ status, reward_terms: rewards }, null, 2);
  }
  return JSON.stringify({error:"unknown tool "+name});
}

const send = o => process.stdout.write(JSON.stringify(o)+"\n");
let buf="";
process.stdin.on("data", async d => {
  buf += d.toString(); let i;
  while((i = buf.indexOf("\n")) >= 0){
    const line = buf.slice(0,i).trim(); buf = buf.slice(i+1);
    if(!line) continue;
    let m; try{ m = JSON.parse(line); }catch{ continue; }
    const { id, method, params } = m;
    if(method==="initialize") send({jsonrpc:"2.0",id,result:{protocolVersion:"2024-11-05",capabilities:{tools:{}},serverInfo:{name:"newton",version:"1.0.0"}}});
    else if(method==="tools/list") send({jsonrpc:"2.0",id,result:{tools: await buildTools()}});
    else if(method==="tools/call"){ const text = await call(params&&params.name, params&&params.arguments); send({jsonrpc:"2.0",id,result:{content:[{type:"text",text}]}}); }
    else if(method==="ping") send({jsonrpc:"2.0",id,result:{}});
    else if(method && method.startsWith("notifications/")){ /* no response */ }
    else if(id!==undefined) send({jsonrpc:"2.0",id,error:{code:-32601,message:"method not found: "+method}});
  }
});
process.stdin.resume();
