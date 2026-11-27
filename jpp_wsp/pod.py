"""Drive the scoring pod through Jupyter on RunPod's HTTP proxy: create it, upload the code and inputs, start score_pod2.sh, follow
its log and GPU use, fetch the outputs, terminate it. The pod needs no GitHub or Hugging Face token: the lenses come as signed links
made just before upload (valid about an hour), and everything else goes over Jupyter.

    python jpp_wsp/pod.py create            # an H200 with Jupyter; writes the pod id and its Jupyter token to --state
    python jpp_wsp/pod.py start             # upload, then run score_pod2.sh in the background on the pod
    python jpp_wsp/pod.py status            # the end of the pod's log, its GPU use, whether it is done
    python jpp_wsp/pod.py fetch --dest DIR  # the outputs
    python jpp_wsp/pod.py terminate

--state (default: the scratch file wsp_pod_state.json) holds the token; it is never committed.
"""

import argparse
import base64
import json
import os
import secrets
import sys
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
GRAPHQL = "https://api.runpod.io/graphql"
IMAGE = "runpod/pytorch:1.4.1-cu1281-torch280-ubuntu2404"  # the image of the earlier jspace pods
HF = "https://huggingface.co/debanshd/jspace-omni-jpp-runs/resolve"
WORK = "workspace/wsp"  # relative to Jupyter's root, /
# the session's HTTPS proxy, read at run time: its port changes between sessions
_proxy = os.environ.get("HTTPS_PROXY", "http://127.0.0.1:36687").rsplit("/", 1)[-1].rsplit("@", 1)[-1]
PROXY = (_proxy.rsplit(":", 1)[0], int(_proxy.rsplit(":", 1)[1]))
CA = "/root/.ccr/ca-bundle.crt"


def gql(query: str, variables: dict | None = None) -> dict:
    r = requests.post(GRAPHQL, json={"query": query, "variables": variables or {}}, timeout=60)
    r.raise_for_status()
    out = r.json()
    if out.get("errors"):
        raise SystemExit(f"RunPod: {out['errors']}")
    return out["data"]


class Jupyter:
    def __init__(self, state: dict):
        self.base = f"https://{state['pod']}-8888.proxy.runpod.net"
        self.h = {"Authorization": f"token {state['token']}"}

    def get(self, path: str, **kw):
        return requests.get(f"{self.base}{path}", headers=self.h, timeout=120, **kw)

    def put_file(self, remote: str, data: bytes) -> None:
        parts = remote.split("/")
        for i in range(1, len(parts)):  # the directories first
            d = "/".join(parts[:i])
            if self.get(f"/api/contents/{d}?content=0").status_code == 404:
                requests.put(f"{self.base}/api/contents/{d}", headers=self.h, json={"type": "directory"}, timeout=60).raise_for_status()
        body = {"type": "file", "format": "base64", "content": base64.b64encode(data).decode()}
        requests.put(f"{self.base}/api/contents/{remote}", headers=self.h, json=body, timeout=120).raise_for_status()

    def read(self, remote: str) -> bytes | None:
        r = self.get(f"/files/{remote}")
        return r.content if r.status_code == 200 else None

    def run(self, command: str) -> None:
        """type a command into a new Jupyter terminal"""
        import websocket

        name = requests.post(f"{self.base}/api/terminals", headers=self.h, timeout=60).json()["name"]
        ws = websocket.create_connection(
            f"wss://{self.base.removeprefix('https://')}/terminals/websocket/{name}",
            header=[f"Authorization: {self.h['Authorization']}"],
            http_proxy_host=PROXY[0],
            http_proxy_port=PROXY[1],
            proxy_type="http",
            sslopt={"ca_certs": CA},
            timeout=60,
        )
        ws.send(json.dumps(["stdin", command + "\r"]))
        time.sleep(3)
        ws.close()

    def ls(self, remote: str) -> list[dict]:
        r = self.get(f"/api/contents/{remote}")
        return r.json()["content"] if r.status_code == 200 else []


def signed(path: str, rev: str) -> str:
    """the CDN link Hugging Face redirects a private file to; the proxy adds the credential to this request only"""
    r = requests.get(f"{HF}/{rev}/{path}", allow_redirects=False, timeout=60)
    if r.status_code != 302:
        raise SystemExit(f"{path}: no signed link ({r.status_code})")
    return r.headers["location"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["create", "start", "status", "fetch", "terminate"])
    ap.add_argument("--state", type=Path, default=Path(os.environ.get("WSP_STATE", "wsp_pod_state.json")))
    ap.add_argument("--gpu", default="NVIDIA H200")
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--pr18", type=Path, help="a checkout of jpp_evals/ at PR 18's 3d8ccd6")
    ap.add_argument("--urls", type=Path, help="JSON {file name: [hf path, revision] in the private repo, or a public URL}")
    ap.add_argument("--script", default="score_pod2.sh", help="the pod-side script in jpp_wsp/")
    ap.add_argument("--code", nargs="*", default=["frozen.json", "frozen2.json"], help="files of jpp_wsp/ the script reads")
    ap.add_argument("--inputs", default="inputs", help="the folder of jpp_wsp/ holding the correctness files")
    args = ap.parse_args()

    if args.cmd == "create":
        token = secrets.token_urlsafe(24)
        inp = {
            "cloudType": "SECURE",
            "gpuCount": 1,
            "volumeInGb": 100,
            "containerDiskInGb": 50,
            "gpuTypeId": args.gpu,
            "name": "jspace-olmo-wsp2",
            "imageName": IMAGE,
            "ports": "8888/http",
            "volumeMountPath": "/workspace",
            "env": [{"key": "JUPYTER_PASSWORD", "value": token}],
        }
        q = "mutation($i: PodFindAndDeployOnDemandInput) { podFindAndDeployOnDemand(input: $i) { id costPerHr machine { gpuDisplayName } } }"
        pod = gql(q, {"i": inp})["podFindAndDeployOnDemand"]
        args.state.write_text(json.dumps({"pod": pod["id"], "token": token}))
        print(json.dumps(pod))
        return

    state = json.loads(args.state.read_text())
    jp = Jupyter(state)

    if args.cmd == "start":
        for _ in range(60):  # Jupyter comes up a minute or two after the pod
            try:
                if jp.get("/api/status").status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(10)
        else:
            raise SystemExit("Jupyter did not answer in 10 minutes")
        for f in sorted(args.pr18.glob("*.py")):
            jp.put_file(f"{WORK}/code/jpp_evals/{f.name}", f.read_bytes())
        for f in [*args.code, args.script]:
            jp.put_file(f"{WORK}/code/{f}", (HERE / f).read_bytes())
        for f in ("model_correctness.csv", "correctness.csv"):
            jp.put_file(f"{WORK}/inputs/{f}", (HERE / args.inputs / f).read_bytes())
        spec = json.loads(args.urls.read_text())
        urls = {name: src if isinstance(src, str) else signed(*src) for name, src in spec.items()}
        jp.put_file(f"{WORK}/urls.json", json.dumps(urls).encode())
        jp.run(f"cd /{WORK} && nohup bash code/{args.script} > run.log 2>&1 &")
        print("started")
    elif args.cmd == "status":
        log = jp.read(f"{WORK}/run.log") or b"(no log yet)"
        gpu = gql("query($p: String!) { pod(input: {podId: $p}) { runtime { uptimeInSeconds gpus { gpuUtilPercent memoryUtilPercent } } } }", {"p": state["pod"]})
        done = "DONE" if jp.read(f"{WORK}/DONE") is not None else "FAILED" if jp.read(f"{WORK}/FAILED") is not None else "running"
        print("\n".join(log.decode(errors="replace").splitlines()[-15:]))
        print(json.dumps(gpu["pod"]["runtime"]), done)
    elif args.cmd == "fetch":

        def pull(remote: str, local: Path) -> None:
            for item in jp.ls(remote):
                if item["type"] == "directory":
                    pull(item["path"], local / item["name"])
                elif not item["name"].endswith(".pt"):
                    local.mkdir(parents=True, exist_ok=True)
                    (local / item["name"]).write_bytes(jp.read(item["path"]))

        pull(f"{WORK}/out", args.dest)
        (args.dest / "run.log").write_bytes(jp.read(f"{WORK}/run.log") or b"")
        print("fetched to", args.dest)
    elif args.cmd == "terminate":
        gql("mutation($p: String!) { podTerminate(input: {podId: $p}) }", {"p": state["pod"]})
        print("terminated", state["pod"])


if __name__ == "__main__":
    sys.exit(main())
