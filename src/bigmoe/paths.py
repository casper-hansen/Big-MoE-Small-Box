import os
from pathlib import Path


def default_home() -> Path:
    if os.environ.get("BIGMOE_HOME"):
        return Path(os.environ["BIGMOE_HOME"]).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "bigmoe"


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "--")


class Home:
    def __init__(self, root=None):
        self.root = (Path(root).expanduser() if root else default_home()).absolute()

    def __repr__(self):
        return f"Home({str(self.root)!r})"

    def __truediv__(self, other):
        return self.root / other

    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def stores(self) -> Path:
        return self.root / "stores"

    @property
    def envs(self) -> Path:
        return self.root / "envs"

    @property
    def kernels(self) -> Path:
        return self.root / "kernels"

    @property
    def wheels(self) -> Path:
        return self.root / "wheels"

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def triton(self) -> Path:
        return self.root / "triton"

    @property
    def results(self) -> Path:
        return self.root / "results"

    def model_dir(self, repo_id: str, commit: str) -> Path:
        return self.models / f"{repo_slug(repo_id)}@{commit[:12]}"

    def store_dir(self, repo_id: str, commit: str) -> Path:
        return self.stores / f"{repo_slug(repo_id)}@{commit[:12]}"

    def run_dir(self, port: int) -> Path:
        return self.runs / str(int(port))
