from scripts.run_codex_bigrasp import write_comparison


def test_summary_compares_both_current_references(tmp_path):
    metrics = ('hoi_err_fusion', 'obj_trans_err_fusion', 'obj_jitter', 'mpjpe', 'root_trans_err')
    manifest = {'dataset': 'behave', 'runs': {
        name: {'metrics': dict.fromkeys(metrics, value)}
        for name, value in (('source_current', 12.0), ('current', 11.0), ('bigrasp', 9.0))
    }}
    write_comparison(tmp_path, manifest)
    assert manifest['comparison']['bigrasp_minus_current']['hoi_err_fusion'] == -2.0
    assert manifest['comparison']['bigrasp_minus_source_current']['hoi_err_fusion'] == -3.0
    text = (tmp_path / 'summary.md').read_text()
    assert '| bigrasp_minus_current | -2.0000 |' in text
    assert '| source_current | 12.0000 |' in text


def test_summary_preserves_missing_metrics(tmp_path):
    manifest = {'dataset': 'omomo', 'runs': {
        name: {'metrics': {'hoi_err_fusion': value}}
        for name, value in (('source_current', 12.0), ('current', 11.0), ('bigrasp', 9.0))
    }}
    write_comparison(tmp_path, manifest)
    assert manifest['comparison']['bigrasp_minus_current']['obj_jitter'] is None
    assert 'N/A' in (tmp_path / 'summary.md').read_text()