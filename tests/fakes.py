"""In-memory stand-ins for the Docker SDK so the heal path can be unit-tested without a daemon."""
import itertools
import time
from types import SimpleNamespace

_ids = itertools.count(1)

PROC_DUMP = ("=== PID 1\n--- cmdline\n/usr/sbin/sshd -D \n--- status\nName:\tsshd\n--- maps\n55a-55b r-xp /usr/sbin/sshd\n"
             "=== PID 77\n--- cmdline\nstress-ng --cpu 2 \n--- status\nName:\tstress-ng\n--- maps\n7f1-7f2 r-xp /usr/bin/stress-ng\n")
NET_TCP = ("  sl  local_address rem_address   st\n   0: 00000000:0016 00000000:0000 0A 0 0 0\n"
           "   1: 0200A8C0:0016 0500A8C0:D2F1 01 0 0 0\n")


class FakeNetwork:
    def __init__(self, name):
        self.name, self.members = name, set()

    def connect(self, c):
        self.members.add(c.id)
        c.attrs["NetworkSettings"]["Networks"][self.name] = {}

    def disconnect(self, c, force=False):
        self.members.discard(c.id)
        c.attrs["NetworkSettings"]["Networks"].pop(self.name, None)


class FakeContainer:
    def __init__(self, name, image, ports=None, cpu=0.0):
        self.id = f"{next(_ids):064x}"
        self.name, self.status, self.cpu = name, "running", cpu
        self.calls = []
        self.attrs = {"Name": "/" + name, "State": {"Pid": 4242},
                      "Config": {"Image": image, "Env": ["A=1"], "Labels": {}},
                      "HostConfig": {"PortBindings": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "2222"}]} if ports else {},
                                     "RestartPolicy": {"Name": "", "MaximumRetryCount": 0}},
                      "NetworkSettings": {"Networks": {"bridge": {}}}}

    def reload(self): pass
    def pause(self): self.calls.append("pause")
    def kill(self): self.calls.append("kill")
    def diff(self): return [{"Path": "/tmp/x", "Kind": 1}, {"Path": "/etc/shadow", "Kind": 0}]
    def logs(self, tail=200): return b"sshd started\n"

    def remove(self, force=False):
        self.calls.append("remove")
        self.status = "removed"

    def exec_run(self, cmd, demux=False):
        out = ""
        if cmd[:2] == ["sh", "-c"]:
            out = PROC_DUMP
        elif cmd == ["cat", "/proc/net/tcp"]:
            out = NET_TCP
        elif cmd[0] == "tail":
            out = "1000.0|START|s1|demoUser|10.0.0.9|sshd\n1001.0|CMD|s1|demoUser|10.0.0.9|sshd>bash>cat|cat /etc/shadow\n"
        return SimpleNamespace(exit_code=0, output=out.encode())


class FakeDocker:
    def __init__(self):
        self.created = []
        self.containers = SimpleNamespace(get=self._get, run=self._run)
        self.networks = SimpleNamespace(get=self._net_get, create=self._net_create)
        self.api = SimpleNamespace(stats=self._stats)
        self._by_id, self._nets = {}, {"bridge": FakeNetwork("bridge")}

    def add(self, c):
        self._by_id[c.id] = c
        return c

    def _get(self, key):
        for c in self._by_id.values():
            if (c.id == key or c.name == key) and c.status != "removed":
                return c
        raise KeyError(key)

    def _run(self, image, **kw):
        c = FakeContainer(kw["name"], image, ports=kw.get("ports"))
        c.run_kwargs = kw
        self.created.append(c)
        return self.add(c)

    def _net_get(self, name):
        if name not in self._nets:
            raise KeyError(name)
        return self._nets[name]

    def _net_create(self, name, **kw):
        self._nets[name] = FakeNetwork(name)
        return self._nets[name]

    def _stats(self, cid, stream=False):
        c = self._by_id[cid]
        return {"cpu_stats": {"cpu_usage": {"total_usage": 1000 + int(c.cpu * 10000)}, "system_cpu_usage": 1_000_000, "online_cpus": 1},
                "precpu_stats": {"cpu_usage": {"total_usage": 1000}, "system_cpu_usage": 0},
                "memory_stats": {"usage": 10_000_000, "limit": 1_000_000_000, "stats": {}}, "networks": {}}
