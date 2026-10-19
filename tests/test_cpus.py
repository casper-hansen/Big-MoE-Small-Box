from bigmoe.cpus import CpuLayout, format_cpus, parse_cpus, plan_layout
from bigmoe.system import Core


def ryzen_7600x():
    return [Core((i, i + 6)) for i in range(6)]


def test_7600x_matches_the_measured_layout():
    lay = plan_layout(ryzen_7600x(), cpu_lane=True)
    assert lay.main == (0,) and lay.controller == 6
    assert format_cpus(lay.lane) == "1-5"
    assert format_cpus(lay.readers) == "7-11"


def test_exact_mode_has_no_cpu_lane():
    lay = plan_layout(ryzen_7600x(), cpu_lane=False)
    assert lay.lane == () and lay.main == (0,) and lay.controller == 6 and format_cpus(lay.readers) == "7-11"


def test_no_smt():
    lay = plan_layout([Core((i,)) for i in range(8)], cpu_lane=True)
    assert lay.main == (0,) and lay.controller == 1
    assert format_cpus(lay.lane) == "2-7" and format_cpus(lay.readers) == "2-7"


def test_parse_format_roundtrip():
    assert parse_cpus("1-5,7, 9-10") == [1, 2, 3, 4, 5, 7, 9, 10]
    assert format_cpus([7, 1, 2, 3, 10, 9]) == "1-3,7,9-10"
    assert parse_cpus(3) == [3]
    lay = CpuLayout.from_dict({"main": "0", "controller": 6, "lane": "1-5", "readers": [7, 8, 9, 10, 11]})
    assert lay == plan_layout(ryzen_7600x(), True)
