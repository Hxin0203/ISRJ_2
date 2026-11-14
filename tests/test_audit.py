import pytest
from tools.run_evaluation import validate_record


def test_audit_rejects_changed_frozen_config_and_sources():
    source={'paper2/network.py':'abc'}
    record=dict(config_sha256='config',source=source,steps=1200)
    validate_record(record,'config',source,1200)
    with pytest.raises(RuntimeError,match='config'):
        validate_record(record,'changed',source,1200)
    with pytest.raises(RuntimeError,match='source'):
        validate_record(record,'config',{'paper2/network.py':'changed'},1200)
    with pytest.raises(RuntimeError,match='steps'):
        validate_record(record,'config',source,100)
