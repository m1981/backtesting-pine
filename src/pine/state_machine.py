"""
State machine infrastructure for strategy state management.

Provides a simple but powerful state machine for managing strategy logic.
"""

from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum

from .core.types import State


@dataclass
class StateTransition:
    """Represents a state transition."""
    from_state: State
    to_state: State
    trigger: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[Any] = None


class StateMachine:
    """
    Simple state machine for strategy logic.

    Tracks current state, handles transitions, and maintains state history.

    Example:
        sm = StateMachine(initial_state=State.IDLE)
        sm.transition_to(State.WAITING_CONFIRMATION, trigger="setup_detected")
        print(sm.current_state)  # State.WAITING_CONFIRMATION
        print(sm.bars_in_state())  # 0
    """

    def __init__(self, initial_state: State = State.IDLE):
        """
        Initialize state machine.

        Args:
            initial_state: Starting state
        """
        self._current_state = initial_state
        self._previous_state: Optional[State] = None
        self._state_entry_bar: int = 0
        self._current_bar: int = 0

        # History
        self._state_history: List[StateTransition] = []
        self._state_metadata: Dict[str, Any] = {}

        # Callbacks (optional)
        self._on_enter_callbacks: Dict[State, List[Callable]] = {}
        self._on_exit_callbacks: Dict[State, List[Callable]] = {}

    @property
    def current_state(self) -> State:
        """Get current state."""
        return self._current_state

    @property
    def previous_state(self) -> Optional[State]:
        """Get previous state."""
        return self._previous_state

    @property
    def state_metadata(self) -> Dict[str, Any]:
        """Get metadata for current state."""
        return self._state_metadata

    def is_state(self, state: State) -> bool:
        """
        Check if current state matches given state.

        Args:
            state: State to check

        Returns:
            True if current state matches
        """
        return self._current_state == state

    def transition_to(
        self,
        new_state: State,
        trigger: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Transition to a new state.

        Args:
            new_state: Target state
            trigger: Name/description of what triggered the transition
            metadata: Optional metadata to attach to this transition
        """
        if new_state == self._current_state:
            return  # Already in this state

        # Call exit callbacks for current state
        if self._current_state in self._on_exit_callbacks:
            for callback in self._on_exit_callbacks[self._current_state]:
                callback()

        # Record transition
        transition = StateTransition(
            from_state=self._current_state,
            to_state=new_state,
            trigger=trigger,
            metadata=metadata or {},
            timestamp=self._current_bar
        )
        self._state_history.append(transition)

        # Update state
        self._previous_state = self._current_state
        self._current_state = new_state
        self._state_entry_bar = self._current_bar

        # Store metadata for new state
        if metadata:
            self._state_metadata = metadata.copy()
        else:
            self._state_metadata = {}

        # Call enter callbacks for new state
        if new_state in self._on_enter_callbacks:
            for callback in self._on_enter_callbacks[new_state]:
                callback()

    def advance_bar(self) -> None:
        """
        Advance to next bar.

        Should be called by the backtesting engine at each bar.
        """
        self._current_bar += 1

    def bars_in_state(self) -> int:
        """
        Get number of bars spent in current state.

        Returns:
            Number of bars since entering current state
        """
        return self._current_bar - self._state_entry_bar

    def bars_since_state(self, state: State) -> Optional[int]:
        """
        Get number of bars since last time in given state.

        Args:
            state: State to check

        Returns:
            Number of bars since state, or None if never in that state
        """
        # Search backwards through history
        for transition in reversed(self._state_history):
            if transition.from_state == state:
                return self._current_bar - transition.timestamp

        return None

    def was_in_state(self, state: State) -> bool:
        """
        Check if state machine was ever in given state.

        Args:
            state: State to check

        Returns:
            True if state was visited
        """
        return any(
            t.from_state == state or t.to_state == state
            for t in self._state_history
        )

    def on_enter(self, state: State, callback: Callable) -> None:
        """
        Register callback to be called when entering a state.

        Args:
            state: State to watch
            callback: Function to call on entry
        """
        if state not in self._on_enter_callbacks:
            self._on_enter_callbacks[state] = []
        self._on_enter_callbacks[state].append(callback)

    def on_exit(self, state: State, callback: Callable) -> None:
        """
        Register callback to be called when exiting a state.

        Args:
            state: State to watch
            callback: Function to call on exit
        """
        if state not in self._on_exit_callbacks:
            self._on_exit_callbacks[state] = []
        self._on_exit_callbacks[state].append(callback)

    def reset(self, initial_state: State = State.IDLE) -> None:
        """
        Reset state machine to initial state.

        Args:
            initial_state: State to reset to
        """
        self._current_state = initial_state
        self._previous_state = None
        self._state_entry_bar = 0
        self._current_bar = 0
        self._state_history = []
        self._state_metadata = {}

    def get_history(self) -> List[StateTransition]:
        """
        Get state transition history.

        Returns:
            List of StateTransition objects
        """
        return self._state_history.copy()

    def __repr__(self) -> str:
        return (
            f"StateMachine(current={self._current_state.name}, "
            f"bars_in_state={self.bars_in_state()})"
        )
