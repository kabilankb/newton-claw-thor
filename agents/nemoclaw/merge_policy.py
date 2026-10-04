import subprocess, sys, yaml
sb, extra_path, out = sys.argv[1:4]
txt = subprocess.check_output(["openshell", "policy", "get", sb, "--full"], text=True)
pol = yaml.safe_load(txt.split("---", 1)[1])
extra = yaml.safe_load(open(extra_path))["network_policies"]
pol.setdefault("network_policies", {}).update(extra)
yaml.safe_dump(pol, open(out, "w"), sort_keys=False)
print("merged:", list(extra))
