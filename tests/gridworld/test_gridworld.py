from unittest.mock import mock_open, patch

import pytest

from gridworld.gridworld import Action, GridWorldSim, State


class TestGridWorldSim:

    DEFAULT_WORLD_SCHEME_FILE = """\
        [["o","o","o","o","e"],\
        ["o","o","o","o","o"],\
        ["o","o","o","o","o"],\
        ["o","s","o","o","o"],]"""
    DEFAULT_WORLD_SCHEME = [
        ["o", "o", "o", "o", "e"],
        ["o", "o", "o", "o", "o"],
        ["o", "o", "o", "o", "o"],
        ["o", "s", "o", "o", "o"],
    ]

    @pytest.fixture
    def gridworld_sim_fixture(self) -> GridWorldSim:
        with patch(
            "builtins.open",
            mock_open(read_data=TestGridWorldSim.DEFAULT_WORLD_SCHEME_FILE),
        ):
            test_path_to_file = "path/to/file"
            return GridWorldSim(test_path_to_file)

    def test_init(self):
        """Automatically loads world schema and sets initial states"""
        with patch(
            "builtins.open",
            mock_open(read_data=TestGridWorldSim.DEFAULT_WORLD_SCHEME_FILE),
        ) as mock_file:
            test_path_to_file = "path/to/file"
            test_gridworld = GridWorldSim(test_path_to_file)

        assert test_gridworld.world_schema == TestGridWorldSim.DEFAULT_WORLD_SCHEME
        assert test_gridworld.world_height == len(TestGridWorldSim.DEFAULT_WORLD_SCHEME)
        assert test_gridworld.world_width == len(
            TestGridWorldSim.DEFAULT_WORLD_SCHEME[0]
        )
        assert test_gridworld.start == (3, 1)
        assert test_gridworld.end == (0, 4)

    def test_is_valid_position(self, gridworld_sim_fixture):
        """Based on DEFAULT_WORLD_SCHEME dimensions"""
        assert gridworld_sim_fixture._is_valid_position((0, 4)) is True
        assert gridworld_sim_fixture._is_valid_position((0, 5)) is False
        assert gridworld_sim_fixture._is_valid_position((3, 0)) is True
        assert gridworld_sim_fixture._is_valid_position((4, 0)) is False

    def test_move(self, gridworld_sim_fixture):
        # All directions
        gridworld_sim_fixture.position = (2, 1)
        assert gridworld_sim_fixture.move(Action.UP) == (0, State((1, 1)))
        assert gridworld_sim_fixture.move(Action.DOWN) == (0, State((2, 1)))
        assert gridworld_sim_fixture.move(Action.LEFT) == (0, State((2, 0)))
        assert gridworld_sim_fixture.move(Action.RIGHT) == (0, State((2, 1)))

        # Out of bounds
        gridworld_sim_fixture.position = (3, 1)
        assert gridworld_sim_fixture.move(Action.DOWN) == (-1, State((3, 1)))

        # To goal state
        gridworld_sim_fixture.position = (0, 3)
        assert gridworld_sim_fixture.move(Action.RIGHT) == (10, State((0, 4)))

        # Try to get out of goal state
        gridworld_sim_fixture.position = (0, 4)
        assert gridworld_sim_fixture.move(Action.LEFT) == (0, State((0, 4)))

    def test_get_position(self, gridworld_sim_fixture):
        assert gridworld_sim_fixture.get_position() == (3, 1)
        gridworld_sim_fixture.move(Action.UP)
        assert gridworld_sim_fixture.get_position() == (2, 1)

    def test_reset(self, gridworld_sim_fixture):
        gridworld_sim_fixture.position = (0, 0)
        gridworld_sim_fixture.reset()
        assert gridworld_sim_fixture.position == (3, 1)
