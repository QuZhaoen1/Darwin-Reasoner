from darwin_reasoner.storage import RunStore


def test_resume_completed_key(tmp_path):
    run = tmp_path / "run"
    store = RunStore(tmp_path, "x", resume_dir=str(run))
    store.append("abc", {"reward": 1.0})
    assert store.is_completed("abc")

    store2 = RunStore(tmp_path, "x", resume_dir=str(run))
    assert store2.is_completed("abc")
