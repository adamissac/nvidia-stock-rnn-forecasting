from nvquant.experiments.registry import Registry, Trial


def test_registry_roundtrip(tmp_path):
    reg = Registry(tmp_path / "r" / "trials.jsonl")
    assert list(reg.rows()) == [] and reg.latest_run_id("model") is None
    t = reg.log(
        Trial(kind="model", name="ridge", config={"a": 1}, git_sha="x", data_hash="d", run_id="r1")
    )
    reg.log(
        Trial(
            kind="strategy",
            name="ridge__sign",
            config={"a": 2},
            git_sha="x",
            data_hash="d",
            run_id="r2",
        )
    )
    rows = list(reg.rows())
    assert (
        len(rows) == 2
        and rows[0]["trial_id"] == t.trial_id
        and rows[0]["config_hash"] == t.config_hash
    )
    assert [r["name"] for r in reg.select(kind="strategy")] == ["ridge__sign"]
    assert reg.select(run_id="r1")[0]["name"] == "ridge"
    assert reg.latest_run_id("strategy") == "r2"
